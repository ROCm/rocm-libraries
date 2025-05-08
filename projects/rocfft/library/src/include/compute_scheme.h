// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef COMPUTE_SCHEME_H
#define COMPUTE_SCHEME_H

#include <string>

// TODO-
// (PROB_DESCRIPTION << 24) | (KERNEL_ALGO << 16 ) |
// (KERNEL_OR_DECOMPOSITION_TYPE << 8) | (KERNEL_LAYOUT) ????
enum ComputeScheme
{
    CS_NONE,
    CS_KERNEL_STOCKHAM,
    CS_KERNEL_STOCKHAM_BLOCK_CC,
    CS_KERNEL_STOCKHAM_BLOCK_RC,
    CS_KERNEL_STOCKHAM_BLOCK_CR,
    CS_KERNEL_TRANSPOSE,
    CS_KERNEL_TRANSPOSE_XY_Z,
    CS_KERNEL_TRANSPOSE_Z_XY,

    CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z,
    CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY,
    CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY,

    CS_REAL_TRANSFORM_USING_CMPLX,
    CS_KERNEL_COPY_R_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_HERM,
    CS_KERNEL_COPY_HERM_TO_CMPLX,
    CS_KERNEL_COPY_CMPLX_TO_R,

    CS_REAL_TRANSFORM_EVEN,
    CS_KERNEL_R_TO_CMPLX,
    CS_KERNEL_R_TO_CMPLX_TRANSPOSE,
    CS_KERNEL_CMPLX_TO_R,
    CS_KERNEL_TRANSPOSE_CMPLX_TO_R,
    CS_REAL_2D_EVEN,
    CS_REAL_3D_EVEN,

    CS_BLUESTEIN,
    CS_KERNEL_CHIRP,
    CS_KERNEL_PAD_MUL,
    CS_KERNEL_FFT_MUL,
    CS_KERNEL_RES_MUL,
    CS_KERNEL_BLUESTEIN_SINGLE,

    CS_L1D_TRTRT,
    CS_L1D_CC,
    CS_L1D_CRT,

    CS_2D_STRAIGHT, // not implemented yet
    CS_2D_RTRT,
    CS_2D_RC,
    CS_KERNEL_2D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_2D_SINGLE,

    CS_3D_STRAIGHT, // not implemented yet
    CS_3D_TRTRTR,
    CS_3D_RTRT,
    CS_3D_BLOCK_RC,
    CS_3D_BLOCK_CR,
    CS_3D_RC,
    CS_KERNEL_3D_STOCKHAM_BLOCK_CC, // not implemented yet
    CS_KERNEL_3D_SINGLE // not implemented yet
};

// print abbreviation for kernel scheme
std::string   PrintKernelSchemeAbbr(ComputeScheme cs);
std::string   PrintScheme(ComputeScheme cs);
ComputeScheme StrToComputeScheme(const std::string& str);
bool          ComputeSchemeIsAProblem(ComputeScheme cs);

#endif
