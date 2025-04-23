/*
Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/


#ifndef MIVISIONX_HIP_HOST_DECLS_H
#define MIVISIONX_HIP_HOST_DECLS_H
#include "hip/hip_runtime.h"
#include <VX/vx.h>
#include "ago_haf_cpu.h"

typedef struct AgoConfigScaleMatrix ago_scale_matrix_t;

// arithmetic_kernels

int HipExec_AbsDiff_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_AbsDiff_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_U8_U8U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_U8_U8U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_S16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_S16_S16U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_S16_S16U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_S16_S16S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Add_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_U8_U8U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_U8_U8U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_S16U8_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_S16U8_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_U8S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_U8S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_S16S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Sub_S16_S16S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Mul_U8_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_U8_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_U8_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_U8_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_U8U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_U8U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_U8U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_U8U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16U8_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16U8_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16U8_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16U8_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16S16_Wrap_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16S16_Wrap_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16S16_Sat_Trunc(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Mul_S16_S16S16_Sat_Round(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 scale);
int HipExec_Magnitude_S16_S16S16(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Phase_U8_S16S16(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_int16 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_WeightedAverage_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        vx_float32 alpha);

// logical_kernels

int HipExec_And_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_And_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Or_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U8_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U8_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U8_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U8_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U1_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U1_U8U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U1_U1U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Xor_U1_U1U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_Not_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_Not_U8_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_Not_U1_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_Not_U1_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);

// color_kernels

int HipExec_Lut_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_uint8 *lut);
int HipExec_ChannelCopy_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelCopy_U8_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelCopy_U1_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelCopy_U1_U1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ColorDepth_U8_S16_Wrap(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift);
int HipExec_ColorDepth_U8_S16_Sat(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift);
int HipExec_ColorDepth_S16_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_int32 shift);
int HipExec_ChannelExtract_U8_U16_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ChannelExtract_U8_U16_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ChannelExtract_U8_U24_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ChannelExtract_U8_U24_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ChannelExtract_U8_U24_Pos2(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ChannelExtract_U8_U32_Pos0(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_df_image srcType);
int HipExec_ChannelExtract_U8_U32_Pos1(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_df_image srcType);
int HipExec_ChannelExtract_U8_U32_Pos2(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_df_image srcType);
int HipExec_ChannelExtract_U8_U32_Pos3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        vx_df_image srcType);
int HipExec_ChannelExtract_U8U8U8_U24(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelExtract_U8U8U8_U32(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelExtract_U8U8U8U8_U32(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage0, vx_uint8 *pHipDstImage1, vx_uint8 *pHipDstImage2, vx_uint8 *pHipDstImage3,
        vx_uint32 dstImageStrideInBytes, const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ChannelCombine_U16_U8U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes);
int HipExec_ChannelCombine_U24_U8U8U8_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes);
int HipExec_ChannelCombine_U32_U8U8U8_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes);
int HipExec_ChannelCombine_U32_U8U8U8_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes);
int HipExec_ChannelCombine_U32_U8U8U8U8_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes,
        const vx_uint8 *pHipSrcImage2, vx_uint32 srcImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage3, vx_uint32 srcImage3StrideInBytes,
        const vx_uint8 *pHipSrcImage4, vx_uint32 srcImage4StrideInBytes);
int HipExec_ColorConvert_RGBX_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGB_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGB_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGB_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGBX_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGBX_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGBX_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_RGB_IYUV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes);
int HipExec_ColorConvert_RGB_NV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes);
int HipExec_ColorConvert_RGB_NV21(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes);
int HipExec_ColorConvert_RGBX_IYUV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcYImage, vx_uint32 srcYImageStrideInBytes,
        const vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        const vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes);
int HipExec_ColorConvert_RGBX_NV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes);
int HipExec_ColorConvert_RGBX_NV21(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcLumaImage, vx_uint32 srcLumaImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes);
int HipExec_ColorConvert_NV12_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
        vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_NV12_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImageLuma, vx_uint32 dstImageLumaStrideInBytes,
        vx_uint8 *pHipDstImageChroma, vx_uint32 dstImageChromaStrideInBytes,
        const vx_uint8 *pHipSrcImage1, vx_uint32 srcImage1StrideInBytes);
int HipExec_ColorConvert_IYUV_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ColorConvert_IYUV_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_NV12_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
        vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
        const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_NV12_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pDstLumaImage, vx_uint32 dstLumaImageStrideInBytes,
        vx_uint8 *pDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
        const vx_uint8 *pSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_IYUV_UYVY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_IYUV_YUYV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ColorConvert_YUV4_RGB(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_IUV_UV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcChromaImage, vx_uint32 srcChromaImageStrideInBytes);
int HipExec_FormatConvert_UV12_IUV(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstChromaImage, vx_uint32 dstChromaImageStrideInBytes,
        vx_uint8 *pHipSrcUImage, vx_uint32 srcUImageStrideInBytes,
        vx_uint8 *pHipSrcVImage, vx_uint32 srcVImageStrideInBytes);
int HipExec_ColorConvert_YUV4_RGBX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstYImage, vx_uint32 dstYImageStrideInBytes,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_FormatConvert_UV_UV12(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstUImage, vx_uint32 dstUImageStrideInBytes,
        vx_uint8 *pHipDstVImage, vx_uint32 dstVImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleUp2x2_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);

// filter_kernels

int HipExec_Box_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Dilate_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Erode_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Median_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Gaussian_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Convolve_U8_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
        vx_uint32 srcImageBufferOffset, float *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight);
int HipExec_Convolve_S16_U8(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
        vx_uint32 srcImageBufferOffset, float *conv, vx_uint32 convolutionWidth, vx_uint32 convolutionHeight);
int HipExec_Sobel_S16S16_U8_3x3_GXY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage1, vx_uint32 dstImage1StrideInBytes,
        vx_int16 *pHipDstImage2, vx_uint32 dstImage2StrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Sobel_S16_U8_3x3_GX(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_Sobel_S16_U8_3x3_GY(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_int16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleGaussianHalf_U8_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleGaussianHalf_U8_U8_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize);

// statistical_kernels

int HipExec_Threshold_U8_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int32 thresholdValue);
int HipExec_Threshold_U8_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper);
int HipExec_Threshold_U1_U8_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int32 thresholdValue);
int HipExec_Threshold_U1_U8_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int32 thresholdLower, vx_int32 thresholdUpper);
int HipExec_Threshold_U8_S16_Binary(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int16 thresholdValue);
int HipExec_Threshold_U8_S16_Range(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_int16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_int16 thresholdLower, vx_int16 thresholdUpper);


// geometric_kernels

int HipExec_ScaleImage_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleImage_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleImage_U8_U8_Bilinear_Replicate(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_ScaleImage_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        const vx_uint8 borderValue);
int HipExec_ScaleImage_U8_U8_Area(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_WarpAffine_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint32 srcImageBufferSize, ago_affine_matrix_t *affineMatrix);
int HipExec_WarpAffine_U8_U8_Nearest_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix,
        vx_uint8 borderValue, vx_rectangle_t rect_valid);
int HipExec_WarpAffine_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint32 srcImageBufferSize, ago_affine_matrix_t *affineMatrix);
int HipExec_WarpAffine_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_affine_matrix_t *affineMatrix,
        vx_uint8 borderValue);
int HipExec_WarpPerspective_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint32 srcImageBufferSize, ago_perspective_matrix_t *perspectiveMatrix);
int HipExec_WarpPerspective_U8_U8_Nearest_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix,
        vx_uint8 borderValue);
int HipExec_WarpPerspective_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint32 srcImageBufferSize, ago_perspective_matrix_t *perspectiveMatrix);
int HipExec_WarpPerspective_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_perspective_matrix_t *perspectiveMatrix,
        vx_uint8 borderValue);
int HipExec_Remap_U8_U8_Nearest(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes, vx_uint32 srcImageBufferSize,
        ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes);
int HipExec_Remap_U8_U8_Nearest_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,vx_uint32 srcImageBufferSize,
        ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes, const vx_uint8 borderValue);
int HipExec_Remap_U8_U8_Bilinear(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_uint32 srcImageBufferSize, ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes);
int HipExec_Remap_U8_U8_Bilinear_Constant(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        ago_coord2d_ushort_t *remap, vx_uint32 remapStrideInBytes, const vx_uint8 borderValue);

// vision_kernels

int HipExec_CannySobel_U16_U8_3x3_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySobel_U16_U8_5x5_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySobel_U16_U8_7x7_L1NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySobel_U16_U8_3x3_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySobel_U16_U8_5x5_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySobel_U16_U8_7x7_L2NORM(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_uint16 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_CannySuppThreshold_U8XY_U16_3x3(hipStream_t stream,
    vx_uint32 dstWidth, vx_uint32 dstHeight,
    vx_uint8 *pHipDstImage, vx_uint32 dstImageStrideInBytes,
    const vx_uint16 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
    vx_uint8* xyStack, vx_uint32 xyStackOffset, vx_uint32 capacityOfXY,
    vx_uint16 hyst_lower, vx_uint16 hyst_upper);
int HipExec_FastCorners_XY_U8_NoSupression(
        hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_uint8 *pDstCorner,
        vx_uint32 cornerBufferOffset, vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_float32 strength_threshold);
int HipExec_FastCorners_XY_U8_Supression(
        hipStream_t stream, vx_uint32 capacityOfDstCorner, vx_uint8 *pDstCorner,
        vx_uint32 cornerBufferOffset, vx_uint32 srcWidth, vx_uint32 srcHeight,
        const vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes,
        vx_float32 strength_threshold);
int HipExec_HarrisSobel_HG3_U8_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_HarrisSobel_HG3_U8_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_HarrisSobel_HG3_U8_7x7(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstGxy, vx_uint32 dstGxyStrideInBytes,
        vx_uint8 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);
int HipExec_HarrisScore_HVC_HG3_3x3(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes,
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_int32 border, vx_float32 normFactor);
int HipExec_HarrisScore_HVC_HG3_5x5(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes, vx_int32 srcImageBufferSize, 
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_int32 border, vx_float32 normFactor);
int HipExec_HarrisScore_HVC_HG3_7x7(
        hipStream_t stream, vx_uint32 dstWidth, vx_uint32 dstHeight,
        vx_float32 *pHipDstVc, vx_uint32 dstVcStrideInBytes,
        vx_float32 *pHipSrcGxy, vx_uint32 srcGxyStrideInBytes,
        vx_float32 sensitivity, vx_float32 strength_threshold,
        vx_int32 border, vx_float32 normFactor);
int HipExec_NonMaxSupp_XY_ANY_3x3(
        hipStream_t stream, vx_uint32 capacityOfList, vx_uint8 *pHipDstList,
        vx_uint32 dstListOffset, vx_uint32 srcWidth, vx_uint32 srcHeight,
        vx_float32 *pHipSrcImage, vx_uint32 srcImageStrideInBytes);

#endif //MIVISIONX_HIP_HOST_DECLS_H
