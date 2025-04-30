/* ************************************************************************
 * Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include <cstdint>
#include <rocblas.h>

#if defined(ROCBLAS_INTERNAL_ILP64)
#define ROCBLAS_API(_f) _f##_64
#else
#define ROCBLAS_API(_f) _f
#endif

#if defined(ROCBLAS_INTERNAL_ILP64)
// this define is tied to enum rocblas_client_api (only supports C and C_64 API for bench)
#define ROCBLAS_API_BENCH "./rocblas-bench --api 1"
#else
#define ROCBLAS_API_BENCH "./rocblas-bench"
#endif

#if defined(ROCBLAS_INTERNAL_ILP64)
#define ROCBLAS_API_STR(_f) #_f "_64"
#else
#define ROCBLAS_API_STR(_f) #_f
#endif

// vs.

template <typename>
inline constexpr char rocblas_api_suffix[] = "";
template <>
inline constexpr char rocblas_api_suffix<int64_t>[] = "_64";

constexpr int64_t c_i32_max = int64_t(std::numeric_limits<int32_t>::max());
constexpr int64_t c_i32_min = int64_t(std::numeric_limits<int32_t>::min());

#ifndef ROCBLAS_DEV_TEST_ILP64

// constants for production

constexpr int64_t c_ILP64_i32_max = c_i32_max;

constexpr int64_t c_i64_grid_X_chunk = 1ULL << 28;
constexpr int64_t c_i64_grid_YZ_chunk
    = int64_t((std::numeric_limits<uint16_t>::max() & ~0xf)); // % 16 == 0

#else

// constants for developer testing using small sizes

constexpr int64_t c_ILP64_i32_max = int64_t(0); // bypass int32 API use case for values < i32_max

// forced testing with small sizes for loop coverage
constexpr int64_t c_i64_grid_X_chunk  = int64_t(512);
constexpr int64_t c_i64_grid_YZ_chunk = int64_t(512);

#endif

// int64 outer loop helpers

// For device pointers (used by non-batched and _strided_batched functions)
template <typename T>
__forceinline__ __device__ __host__ auto
    adjust_ptr_batch(T const* p, int64_t block, rocblas_stride stride)
{
    return p + block * stride;
}

template <typename T>
__forceinline__ __device__ __host__ auto
    adjust_ptr_batch(T* p, int64_t block, rocblas_stride stride)
{
    return p + block * stride;
}

// For device array of device pointers (used by _batched functions)

template <typename T>
__forceinline__ __device__ __host__ auto
    adjust_ptr_batch(T const* const* p, int64_t block, rocblas_stride stride)
{
    return p + block;
}

template <typename T>
__forceinline__ __device__ __host__ auto
    adjust_ptr_batch(T* const* p, int64_t block, rocblas_stride stride)
{
    return p + block;
}

template <typename T>
__forceinline__ __device__ __host__ auto
    adjust_ptr_batch(T** p, int64_t block, rocblas_stride stride)
{
    return p + block;
}
