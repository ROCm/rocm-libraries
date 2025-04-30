// Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef RTC_BLUESTEIN_GEN
#define RTC_BLUESTEIN_GEN

#include "../device/kernels/callback.h"
#include "compute_scheme.h"
#include "load_store_ops.h"
#include "rocfft/rocfft.h"
#include <vector>

// single kernel bluestein
struct BluesteinSingleSpecs
{
    unsigned int              length;
    unsigned int              dim;
    std::vector<unsigned int> factors;
    unsigned int              threads_per_block;
    unsigned int              threads_per_transform;
    int                       direction;
    rocfft_precision          precision;
    rocfft_result_placement   placement;
    rocfft_array_type         inArrayType;
    rocfft_array_type         outArrayType;
    CallbackType              cbtype;
    LoadOps                   loadOps;
    StoreOps                  storeOps;
};
std::string bluestein_single_rtc_kernel_name(const BluesteinSingleSpecs& specs);
std::string bluestein_single_rtc(const std::string& kernel_name, const BluesteinSingleSpecs& specs);

static const unsigned int LAUNCH_BOUNDS_BLUESTEIN_MULTI_KERNEL = 64;

// multi-kernel bluestein
struct BluesteinMultiSpecs
{
    ComputeScheme     scheme;
    rocfft_precision  precision;
    rocfft_array_type inArrayType;
    rocfft_array_type outArrayType;
    CallbackType      cbtype;
    LoadOps           loadOps;
    StoreOps          storeOps;
};

std::string bluestein_multi_rtc_kernel_name(const BluesteinMultiSpecs& specs);

std::string bluestein_multi_rtc(const std::string& kernel_name, const BluesteinMultiSpecs& specs);

#endif
