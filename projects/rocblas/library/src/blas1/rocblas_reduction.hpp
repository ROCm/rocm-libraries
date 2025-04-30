/* ************************************************************************
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "device_macros.hpp"
#include "int64_helpers.hpp"
#include "utility.hpp"
#include <hip/hip_runtime.h>

static constexpr int rocblas_log2ui(int x)
{
    unsigned int ax = (unsigned int)x;
    int          v  = 0;
    while(ax >>= 1)
    {
        v++;
    }
    return v;
}

template <int N, typename T>
__inline__ __device__ T rocblas_wavefront_reduce(T val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
#pragma unroll
    for(int i = 0; i < WFBITS; i++)
    {
        val += __shfl_down(val, offset);
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_float_complex rocblas_wavefront_reduce(rocblas_float_complex val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
#pragma unroll
    for(int i = 0; i < WFBITS; i++)
    {
        val.real(val.real() + __shfl_down(val.real(), offset));
        val.imag(val.imag() + __shfl_down(val.imag(), offset));
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_double_complex rocblas_wavefront_reduce(rocblas_double_complex val)
{
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
#pragma unroll
    for(int i = 0; i < WFBITS; i++)
    {
        val.real(val.real() + __shfl_down(val.real(), offset));
        val.imag(val.imag() + __shfl_down(val.imag(), offset));
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_bfloat16 rocblas_wavefront_reduce(rocblas_bfloat16 val)
{
    union
    {
        int              i;
        rocblas_bfloat16 h;
    } tmp;
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
#pragma unroll
    for(int i = 0; i < WFBITS; i++)
    {
        tmp.h = val;
        tmp.i = __shfl_down(tmp.i, offset);
        val += tmp.h;
        offset >>= 1;
    }
    return val;
}

template <int N>
__inline__ __device__ rocblas_half rocblas_wavefront_reduce(rocblas_half val)
{
    union
    {
        int          i;
        rocblas_half h;
    } tmp;
    constexpr int WFBITS = rocblas_log2ui(N);
    int           offset = 1 << (WFBITS - 1);
#pragma unroll
    for(int i = 0; i < WFBITS; i++)
    {
        tmp.h = val;
        tmp.i = __shfl_down(tmp.i, offset);
        val += tmp.h;
        offset >>= 1;
    }
    return val;
}

template <int WARP, int NB, typename T>
__inline__ __device__ T rocblas_dot_block_reduce(T val)
{
    __shared__ T psums[WARP];

    rocblas_int wavefront = threadIdx.x / WARP;
    rocblas_int wavelet   = threadIdx.x % WARP;

    if(wavefront == 0)
        psums[wavelet] = T(0);
    __syncthreads();

    val = rocblas_wavefront_reduce<WARP>(val); // sum over wavefront
    if(wavelet == 0)
        psums[wavefront] = val; // store sum for wavefront

    __syncthreads(); // Wait for all wavefront reductions

    // ensure wavefront was run
    static constexpr rocblas_int num_wavefronts = NB / WARP;

    // single warp of either size will doing final sum so test with smaller constraint
    static_assert(num_wavefronts <= WARP_32);

    val = (threadIdx.x < num_wavefronts) ? psums[wavelet] : T(0);
    if(wavefront == 0)
        val = rocblas_wavefront_reduce<num_wavefronts>(val); // sum wavefront sums

    return val;
}

template <typename API_INT>
inline size_t rocblas_reduction_kernel_block_count(API_INT n, int NB)
{
    if(n <= 0)
        n = 1; // avoid sign loss issues
    return size_t(n - 1) / NB + 1;
}

inline size_t rocblas_reduction_kernel_pass_count(int64_t n)
{
    if(n <= 0)
        n = 0;
    int64_t passes = (n - 1) / c_i64_grid_X_chunk + 1;
    return size_t(passes);
}

template <typename API_INT, int NB, typename To>
size_t rocblas_single_pass_reduction_workspace_size(API_INT n, API_INT batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;

    API_INT n_chunked = n;
    if constexpr(std::is_same_v<API_INT, int64_t>)
    {
        // algorithm in launcher required to always chunk, no 32bit pass-through for n < int32 max
        n_chunked = std::min(n_chunked, c_i64_grid_X_chunk);
    }
    auto blocks = rocblas_reduction_kernel_block_count<API_INT>(n_chunked, NB);

    if constexpr(std::is_same_v<API_INT, int64_t>)
    {
        int64_t batches = std::min(batch_count, c_i64_grid_YZ_chunk);

        return sizeof(To) * (blocks + 1) * batches;
    }
    else
    {
        // original API
        return sizeof(To) * (blocks + 1) * batch_count;
    }
}

/*! \brief rocblas_multi_pass_reduction_workspace_size
    Work area where each chunk is fully reduced to single value
    ********************************************************************/
template <typename API_INT, int NB, typename To>
size_t rocblas_multi_pass_reduction_workspace_size(API_INT n, API_INT batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;

    API_INT n_chunked = n;
    if constexpr(std::is_same_v<API_INT, int64_t>)
    {
        // algorithm in launcher required to always chunk, no 32bit pass-through for n < int32 max
        n_chunked = std::min(n_chunked, c_i64_grid_X_chunk);
    }
    auto blocks = rocblas_reduction_kernel_block_count<API_INT>(n_chunked, NB);

    if constexpr(std::is_same_v<API_INT, int64_t>)
    {
        auto    passes  = rocblas_reduction_kernel_pass_count(n);
        int64_t batches = std::min(batch_count, c_i64_grid_YZ_chunk);

        // each pass is reduced so only addition of passes * batches
        return sizeof(To) * ((blocks + 1) * batches + passes * batches);
    }
    else
    {
        // original API
        return sizeof(To) * (blocks + 1) * batch_count;
    }
}

/*! \brief rocblas_reduction_workspace_non_chunked_size
    Work area for reduction must be at lease sizeof(To) * (blocks + 1) * batch_count.

    @param[in]
    outputType To
        Type of output values
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <typename API_INT, int NB, typename To>
size_t rocblas_reduction_workspace_non_chunked_size(API_INT n, API_INT batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;

    auto blocks = rocblas_reduction_kernel_block_count<API_INT>(n, NB);

    // original API

    return sizeof(To) * (blocks + 1) * batch_count;
}

/*! \brief rocblas_reduction_kernel_part2
    gathers all the partial results in workspace and finishes the final reduction.
    ********************************************************************/
template <int NB, int WIN, typename FINALIZE, typename V, typename T = V>
ROCBLAS_KERNEL(NB)
rocblas_reduction_kernel_part2(int n_sums, V* __restrict__ in, T* __restrict__ out)
{
    V sum = 0;

    size_t offset = size_t(blockIdx.x) * n_sums;
    in += offset;

    int inc = NB * WIN;

    int i         = threadIdx.x * WIN;
    int remainder = n_sums % WIN;
    int end       = n_sums - remainder;
    for(; i < end; i += inc) // cover all sums as 1 block
    {
        for(int j = 0; j < WIN; j++)
            sum += in[i + j];
    }
    if(threadIdx.x < remainder)
    {
        sum += in[n_sums - 1 - threadIdx.x];
    }

    if(warpSize == WARP_32)
        sum = rocblas_dot_block_reduce<WARP_32, NB>(sum);
    else
        sum = rocblas_dot_block_reduce<WARP_64, NB>(sum);

    if(threadIdx.x == 0)
        out[blockIdx.x] = T(FINALIZE{}(sum));
}

/*! \brief rocblas_reduction_batched_kernel_workspace_size
    Work area for reductions where full reduction to single value occurs
    Additional passes add to workspace requirement for ILP64 subdivisions but size limited to chunks

    @param[in]
    outputType To
        Type of output values
    @param[in]
    batch_count rocblas_int
        Number of batches
    ********************************************************************/
template <typename API_INT, int NB, typename To>
size_t
    rocblas_reduction_workspace_size(API_INT n, API_INT incx, API_INT incy, API_INT batch_count = 1)
{
    if(n <= 0)
        n = 1; // allow for return value of empty set
    if(batch_count <= 0)
        batch_count = 1;

    if constexpr(std::is_same_v<API_INT, int64_t>)
    {
        // must track _64 kernel decision code

        if(std::abs(incx) <= c_i32_max && std::abs(incy) <= c_i32_max && n <= c_i32_max
           && batch_count < c_i64_grid_YZ_chunk)
        {
            // reusing 32- bit API
            return rocblas_reduction_workspace_non_chunked_size<API_INT, NB, To>(n, batch_count);
        }
        else
        {
            // algorithm in launcher required to always chunk for these sizes or workspace too small
            return rocblas_multi_pass_reduction_workspace_size<API_INT, NB, To>(n, batch_count);
        }
    }
    else
    {
        // original API
        return rocblas_reduction_workspace_non_chunked_size<API_INT, NB, To>(n, batch_count);
    }
}

template <typename API_INT, int NB, typename To>
size_t rocblas_reduction_workspace_size(
    API_INT n, API_INT incx, API_INT incy, API_INT batch_count, To* output_type)
{
    return rocblas_reduction_workspace_size<API_INT, NB, To>(n, incx, incy, batch_count);
}

template <typename API_INT, int NB>
size_t rocblas_reduction_workspace_size(
    API_INT n, API_INT incx, API_INT incy, API_INT batch_count, rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return rocblas_reduction_workspace_size<API_INT, NB, rocblas_half>(
            n, incx, incy, batch_count);
    case rocblas_datatype_bf16_r:
        return rocblas_reduction_workspace_size<API_INT, NB, rocblas_bfloat16>(
            n, incx, incy, batch_count);
    case rocblas_datatype_f32_r:
        return rocblas_reduction_workspace_size<API_INT, NB, float>(n, incx, incy, batch_count);
    case rocblas_datatype_f64_r:
        return rocblas_reduction_workspace_size<API_INT, NB, double>(n, incx, incy, batch_count);
    case rocblas_datatype_f32_c:
        return rocblas_reduction_workspace_size<API_INT, NB, rocblas_float_complex>(
            n, incx, incy, batch_count);
    case rocblas_datatype_f64_c:
        return rocblas_reduction_workspace_size<API_INT, NB, rocblas_double_complex>(
            n, incx, incy, batch_count);
    default:
        return 0;
    }
}
