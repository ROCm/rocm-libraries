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

#ifndef RTC_TRANSPOSE_GEN
#define RTC_TRANSPOSE_GEN

#include "load_store_ops.h"
#include "rocfft/rocfft.h"
#include "rtc_kernel.h"

#include "../device/kernels/common.h"

struct TransposeSpecs
{
    unsigned int      tileX;
    unsigned int      tileY;
    size_t            dim;
    rocfft_precision  precision;
    rocfft_array_type inArrayType;
    rocfft_array_type outArrayType;
    size_t            largeTwdSteps;
    int               largeTwdDirection;
    bool              diagonal;
    bool              tileAligned;
    CallbackType      cbtype;
    LoadOps           loadOps;
    StoreOps          storeOps;
    bool              grid3D;
};

// generate name for RTC transpose kernel
std::string transpose_rtc_kernel_name(const TransposeSpecs& specs);

// generate source for RTC transpose kernel.
std::string transpose_rtc(const std::string& kernel_name, const TransposeSpecs& specs);

#endif
