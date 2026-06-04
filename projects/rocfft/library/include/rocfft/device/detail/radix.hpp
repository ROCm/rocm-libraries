// Copyright (C) 2016 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

/*! @file radix.hpp
 *  @brief Radix butterfly functions and constants for the rocFFT device library.
 *  @details Consolidated from
 *  library/src/device/generator/rtc_radix_functions/radix_*.h and
 *  library/src/device/kernels/butterfly_constant.h. Provides
 *  `FwdRad{N}B1` / `InvRad{N}B1` for radices 2, 4, 8, and 16 (the radices used by
 *  the pre-generated Stockham kernels). Higher radices (3, 5, 6, 7, 9, 10, 11,
 *  13, 17) can be added when needed by including additional radix headers.
 */

#pragma once

#ifndef ROCFFT_DEVICE_RADIX_HPP
#define ROCFFT_DEVICE_RADIX_HPP

#include "complex.hpp"

// ---- butterfly constants ----

#ifndef BUTTERFLY_CONSTANT_H
#define C8Q  static_cast<real_type_t<T>>(0.70710678118654752440084436210485)
#define C16A static_cast<real_type_t<T>>(0.923879532511286738)
#define C16B static_cast<real_type_t<T>>(0.382683432365089837)
#endif

// ---- radix-2 ----

template <typename T>
__device__ void FwdRad2B1(T* R0, T* R1)
{
    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

template <typename T>
__device__ void InvRad2B1(T* R0, T* R1)
{
    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
}

// ---- radix-4 ----

template <typename T>
__device__ void FwdRad4B1(T* R0, T* R2, T* R1, T* R3)
{
    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);

    (*R3) = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

template <typename T>
__device__ void InvRad4B1(T* R0, T* R2, T* R1, T* R3)
{
    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);

    res   = (*R1);
    (*R1) = (*R2);
    (*R2) = res;
}

// ---- radix-8 ----

template <typename T>
__device__ void FwdRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{
    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T(-(*R3).y, (*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + T(-(*R7).y, (*R7).x);

    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) - C8Q * T((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + T(-(*R6).y, (*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) - C8Q * T((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

template <typename T>
__device__ void InvRad8B1(T* R0, T* R4, T* R2, T* R6, T* R1, T* R5, T* R3, T* R7)
{
    T res;

    (*R1) = (*R0) - (*R1);
    (*R0) = 2.0 * (*R0) - (*R1);
    (*R3) = (*R2) - (*R3);
    (*R2) = 2.0 * (*R2) - (*R3);
    (*R5) = (*R4) - (*R5);
    (*R4) = 2.0 * (*R4) - (*R5);
    (*R7) = (*R6) - (*R7);
    (*R6) = 2.0 * (*R6) - (*R7);

    (*R2) = (*R0) - (*R2);
    (*R0) = 2.0 * (*R0) - (*R2);
    (*R3) = (*R1) + T((*R3).y, -(*R3).x);
    (*R1) = 2.0 * (*R1) - (*R3);
    (*R6) = (*R4) - (*R6);
    (*R4) = 2.0 * (*R4) - (*R6);
    (*R7) = (*R5) + T((*R7).y, -(*R7).x);
    (*R5) = 2.0 * (*R5) - (*R7);

    (*R4) = (*R0) - (*R4);
    (*R0) = 2.0 * (*R0) - (*R4);
    (*R5) = ((*R1) - C8Q * (*R5)) + C8Q * T((*R5).y, -(*R5).x);
    (*R1) = 2.0 * (*R1) - (*R5);
    (*R6) = (*R2) + T((*R6).y, -(*R6).x);
    (*R2) = 2.0 * (*R2) - (*R6);
    (*R7) = ((*R3) + C8Q * (*R7)) + C8Q * T((*R7).y, -(*R7).x);
    (*R3) = 2.0 * (*R3) - (*R7);

    res   = (*R1);
    (*R1) = (*R4);
    (*R4) = res;
    res   = (*R3);
    (*R3) = (*R6);
    (*R6) = res;
}

// ---- radix-16: include directly from upstream to ensure exact correctness ----

#include "../../../../src/device/generator/rtc_radix_functions/radix_16.h"

#endif // ROCFFT_DEVICE_RADIX_HPP
