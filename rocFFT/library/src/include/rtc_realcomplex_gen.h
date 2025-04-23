// Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef RTC_REAL2COMPLEX_EMBED_GEN
#define RTC_REAL2COMPLEX_EMBED_GEN

#include "compute_scheme.h"
#include "load_store_ops.h"
#include "rocfft/rocfft.h"
#include "rtc_kernel.h"

#include "../device/kernels/common.h"

#include <vector>

struct RealComplexSpecs
{
    ComputeScheme     scheme;
    size_t            dim;
    rocfft_precision  precision;
    rocfft_array_type inArrayType;
    rocfft_array_type outArrayType;
    CallbackType      cbtype;
    LoadOps           loadOps;
    StoreOps          storeOps;
    bool              grid3D;
};

struct RealComplexEvenSpecs : public RealComplexSpecs
{
    RealComplexEvenSpecs(RealComplexSpecs&& baseSpecs, bool Ndiv4)
        : RealComplexSpecs(baseSpecs)
        , Ndiv4(Ndiv4)
    {
    }
    bool Ndiv4;
};

struct RealComplexEvenTransposeSpecs : public RealComplexSpecs
{
    RealComplexEvenTransposeSpecs(RealComplexSpecs&& baseSpecs)
        : RealComplexSpecs(baseSpecs)
    {
    }

    static unsigned int TileX(ComputeScheme scheme)
    {
        // r2c uses 16x16 tiles, c2r uses 32x16
        return scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE ? 16 : 32;
    }
    unsigned int TileX() const
    {
        return TileX(scheme);
    }
    static unsigned int TileY()
    {
        return 16;
    }
};

// generate name for RTC realcomplex kernel
std::string realcomplex_rtc_kernel_name(const RealComplexSpecs& specs);
std::string realcomplex_even_rtc_kernel_name(const RealComplexEvenSpecs& specs);
std::string realcomplex_even_transpose_rtc_kernel_name(const RealComplexEvenTransposeSpecs& specs);

// generate source for RTC realcomplex kernel.
std::string realcomplex_rtc(const std::string& kernel_name, const RealComplexSpecs& specs);
std::string realcomplex_even_rtc(const std::string& kernel_name, const RealComplexEvenSpecs& specs);
std::string realcomplex_even_transpose_rtc(const std::string&                   kernel_name,
                                           const RealComplexEvenTransposeSpecs& specs);

#endif
