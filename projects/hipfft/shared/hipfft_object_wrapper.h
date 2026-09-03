// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPFFT_OBJECT_WRAPPER_H
#define HIPFFT_OBJECT_WRAPPER_H

#include "hip_object_wrapper.h"
#include "hipfft/hipfft.h"
#include "hipfft/hipfftXt.h"

// plan handles are pointers for rocFFT backend, and ints for cuFFT
#ifdef __HIP_PLATFORM_AMD__
static constexpr hipfftHandle INVALID_HIPFFT_PLAN_HANDLE = nullptr;
#else
static constexpr hipfftHandle INVALID_HIPFFT_PLAN_HANDLE = -1;
#endif

// hipfftXtMalloc takes (plan, &desc, format) but hip_object_wrapper_t expects TCreate(&obj, ...).
// This adapter reorders the arguments to match.
inline hipfftResult
    hipfftXtMalloc_adapted(hipLibXtDesc** desc, hipfftHandle plan, hipfftXtSubFormat fmt)
{
    return hipfftXtMalloc(plan, desc, fmt);
}

// RAII wrappers for hipFFT handles and Xt descriptors
typedef hip_object_wrapper_t<hipfftHandle,
                             hipfftCreate,
                             hipfftDestroy,
                             HIPFFT_SUCCESS,
                             INVALID_HIPFFT_PLAN_HANDLE>
    hipfftHandle_wrapper_t;
typedef hip_object_wrapper_t<hipLibXtDesc*, hipfftXtMalloc_adapted, hipfftXtFree, HIPFFT_SUCCESS>
    hipfftLibXtDesc_wrapper_t;

#endif // HIPFFT_OBJECT_WRAPPER_H
