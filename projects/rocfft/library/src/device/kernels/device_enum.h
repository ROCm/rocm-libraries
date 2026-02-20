// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef DEVICE_ENUM_H
#define DEVICE_ENUM_H

enum StrideBin
{
    SB_UNIT,
    SB_NONUNIT,
};

enum class EmbeddedType : int
{
    NONE        = 0, // Works as the regular complex to complex FFT kernel
    Real2C_POST = 1, // Works with even-length real2complex post-processing
    C2Real_PRE  = 2, // Works with even-length complex2real pre-processing
};

// TODO: rework this
//
//
// NB:
// SBRC kernels can be used in various scenarios. Instead of tmeplate all
// combinations, we define/enable the cases in using only. In this way,
// the logic in POWX_LARGE_SBRC_GENERATOR() would be simple. People could
// add more later or find a way to simply POWX_LARGE_SBRC_GENERATOR().
enum SBRC_TYPE
{
    SBRC_2D = 2, // for one step in 1D middle size decomposition

    SBRC_3D_FFT_TRANS_XY_Z = 3, // for 3D C2C middle size fused kernel
    SBRC_3D_FFT_TRANS_Z_XY = 4, // for 3D R2C middle size fused kernel
    SBRC_3D_TRANS_XY_Z_FFT = 5, // for 3D C2R middle size fused kernel

    // for 3D R2C middle size, to fuse FFT, Even-length real2complex, and Transpose_Z_XY
    SBRC_3D_FFT_ERC_TRANS_Z_XY = 6,

    // for 3D C2R middle size, to fuse Transpose_XY_Z, Even-length complex2real, and FFT
    SBRC_3D_TRANS_XY_Z_ECR_FFT = 7,
};

enum SBRC_TRANSPOSE_TYPE
{
    NONE, // indicating this is a non-sbrc type, an SBRC kernel shouldn't have this
    DIAGONAL, // best, but requires cube sizes
    TILE_ALIGNED, // OK, doesn't require handling unaligned corner case
    TILE_UNALIGNED,
};

enum DirectRegType
{
    // the direct-to-from-reg codes are not even generated from generator
    // or is generated but we don't want to use it in some arch
    FORCE_OFF_OR_NOT_SUPPORT,
    TRY_ENABLE_IF_SUPPORT, // Use the direct-to-from-reg function
};

enum IntrinsicAccessType
{
    DISABLE_BOTH, // turn-off intrinsic buffer load/store
    ENABLE_LOAD_ONLY, // turn-on intrinsic buffer load only
    ENABLE_BOTH, // turn-on both intrinsic buffer load/store
};

enum BluesteinType
{
    BT_NONE,
    BT_SINGLE_KERNEL, // implementation for small lengths (that fit in LDS)
    BT_MULTI_KERNEL, // large lengths
    BT_MULTI_KERNEL_FUSED, // large lengths with fused intermediate Bluestein operations
};

enum BluesteinFuseType
{ // Fused operation types for multi-kernel Bluestein
    BFT_NONE,
    BFT_FWD_CHIRP, // fused chirp + padding + forward fft
    BFT_FWD_CHIRP_MUL, // fused chirp / input Hadamard product + padding + forward fft
    BFT_INV_CHIRP_MUL, // fused convolution Hadamard product + inverse fft + chirp Hadamard product
};

enum PartialPassType
{
    PPT_NONE,
    PPT_SBCC,
    PPT_SBRR,
};

#endif
