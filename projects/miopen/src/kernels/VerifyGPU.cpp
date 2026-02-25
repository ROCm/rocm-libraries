/*******************************************************************************
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

#include "verify_float_types.hpp"

#include <type_traits>

template <int BlockSize>
__device__ __forceinline__ void CalculateRMS(const FLOAT_UUT* __restrict__ uut,
                                             const FLOAT_REF* __restrict__ ref,
                                             size_t sz,
                                             float* rms,
                                             float* max)
{
    __shared__ float shared_data[BlockSize];
    __shared__ float shared_max_ref[BlockSize];
    __shared__ float shared_max_uut[BlockSize];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * (blockDim.x * 2) + tid;

    float uut1 = gid < sz ? CVT_FLOAT_UUT2ACCUM(uut[gid]) : 0;
    float uut2 = (gid + blockDim.x) < sz ? CVT_FLOAT_UUT2ACCUM(uut[gid + blockDim.x]) : 0;
    float ref1 = gid < sz ? CVT_FLOAT_REF2ACCUM(ref[gid]) : 0;
    float ref2 = (gid + blockDim.x) < sz ? CVT_FLOAT_REF2ACCUM(ref[gid + blockDim.x]) : 0;

    shared_data[tid]    = powf(uut1 - ref1, 2) + powf(uut2 - ref2, 2);
    shared_max_ref[tid] = fmaxf(fabsf(ref1), fabsf(ref2));
    shared_max_uut[tid] = fmaxf(fabsf(uut1), fabsf(uut2));

    __syncthreads();

    for(int i = blockDim.x / 2; i >= warpSize; i >>= 1)
    {
        if(tid < i)
        {
            shared_data[tid]    = shared_data[tid] + shared_data[tid + i];
            shared_max_ref[tid] = fmaxf(shared_max_ref[tid], shared_max_ref[tid + i]);
            shared_max_uut[tid] = fmaxf(shared_max_uut[tid], shared_max_uut[tid + i]);
        }
        __syncthreads();
    }

    float local_res     = shared_data[tid];
    float local_max_ref = shared_max_ref[tid];
    float local_max_uut = shared_max_uut[tid];
    __syncthreads();

#pragma unroll
    for(int i = warpSize / 2; i != 0; i >>= 1)
    {
        local_res     = local_res + __shfl_down(local_res, i);
        local_max_ref = fmaxf(local_max_ref, __shfl_down(local_max_ref, i));
        local_max_uut = fmaxf(local_max_uut, __shfl_down(local_max_uut, i));
    }

    if(tid == 0)
    {
        atomicAdd(rms, local_res);
        float block_max_total = fmaxf(local_max_ref, local_max_uut);
        atomicMax(max, block_max_total);
    }
}

template <int BlockSize>
__device__ __forceinline__ void CalculateRMS(const FLOAT_UUT* __restrict__ uut,
                                             const FLOAT_REF* __restrict__ ref,
                                             size_t sz,
                                             double* rms,
                                             double* max)
{
    __shared__ double shared_data[BlockSize];
    __shared__ double shared_max_ref[BlockSize];
    __shared__ double shared_max_uut[BlockSize];

    const int tid = threadIdx.x;
    const int gid = blockIdx.x * (blockDim.x * 2) + tid;

    double uut1 = gid < sz ? (CVT_FLOAT_UUT2ACCUM(uut[gid])) : 0;
    double uut2 = (gid + blockDim.x) < sz ? (CVT_FLOAT_UUT2ACCUM(uut[gid + blockDim.x])) : 0;
    double ref1 = gid < sz ? (CVT_FLOAT_REF2ACCUM(ref[gid])) : 0;
    double ref2 = (gid + blockDim.x) < sz ? (CVT_FLOAT_REF2ACCUM(ref[gid + blockDim.x])) : 0;

    shared_data[tid]    = pow(uut1 - ref1, 2) + pow(uut2 - ref2, 2);
    shared_max_ref[tid] = fmax(fabs(ref1), fabs(ref2));
    shared_max_uut[tid] = fmax(fabs(uut1), fabs(uut2));

    __syncthreads();

    for(int i = blockDim.x / 2; i >= warpSize; i >>= 1)
    {
        if(tid < i)
        {
            shared_data[tid]    = shared_data[tid] + shared_data[tid + i];
            shared_max_ref[tid] = fmax(shared_max_ref[tid], shared_max_ref[tid + i]);
            shared_max_uut[tid] = fmax(shared_max_uut[tid], shared_max_uut[tid + i]);
        }
        __syncthreads();
    }

    double local_res     = shared_data[tid];
    double local_max_ref = shared_max_ref[tid];
    double local_max_uut = shared_max_uut[tid];
    __syncthreads();

#pragma unroll
    for(int i = warpSize / 2; i != 0; i >>= 1)
    {
        local_res     = local_res + __shfl_down(local_res, i);
        local_max_ref = fmax(local_max_ref, __shfl_down(local_max_ref, i));
        local_max_uut = fmax(local_max_uut, __shfl_down(local_max_uut, i));
    }

    if(tid == 0)
    {
        atomicAdd(rms, local_res);
        double block_max_total = fmax(local_max_ref, local_max_uut);
        atomicMax(max, block_max_total);
    }
}

// template <int BlockSize, typename T, typename TVerify>
// __device__ void
// CalculateMAE(const T* __restrict__ uut, const T* __restrict__ ref, size_t sz, TVerify* mae)
// {
//     __shared__ TVerify shared_data[BlockSize];

//     const int tid = threadIdx.x;
//     const int gid = blockIdx.x * (blockDim.x * 2) + tid;

//     TVerify uut1 = gid < sz ? static_cast<TVerify>(uut[gid]) : static_cast<TVerify>(0);
//     TVerify uut2 = (gid + blockDim.x) < sz ? static_cast<TVerify>(uut[gid + blockDim.x])
//                                            : static_cast<TVerify>(0);
//     TVerify ref1 = gid < sz ? static_cast<TVerify>(ref[gid]) : static_cast<TVerify>(0);
//     TVerify ref2 = (gid + blockDim.x) < sz ? static_cast<TVerify>(ref[gid + blockDim.x])
//                                            : static_cast<TVerify>(0);

//     shared_data[tid] = fmaxf(fabsf(uut1 - ref1), fabsf(uut2 - ref2));

//     __syncthreads();

//     for(int i = blockDim.x / 2; i >= warpSize; i >>= 1)
//     {
//         if(tid < i)
//         {
//             shared_data[tid] = fmaxf(shared_data[tid], shared_data[tid + i]);
//         }
//         __syncthreads();
//     }

//     TVerify local_res = shared_data[tid];
//     __syncthreads();

// #pragma unroll
//     for(int i = warpSize / 2; i != 0; i >>= 1)
//     {
//         local_res = fmaxf(local_res, __shfl_down(local_res, i));
//     }

//     if(tid == 0)
//     {
//         atomicMax(mae, local_res);
//     }
// }

// template <typename T, typename TVerify>
// __device__ void
// FindMismatch(const T* __restrict__ uut, const T* __restrict__ ref, size_t sz, TVerify* mismatch)
// {
//     size_t gid  = blockIdx.x * blockDim.x + threadIdx.x;
//     size_t step = gridDim.x * blockDim.x;

//     if(gid >= sz)
//     {
//         return;
//     }

//     while(gid < sz && *mismatch == 0.0)
//     {
//         if(uut[gid] != ref[gid])
//         {
//             *mismatch = 1.0;
//         }
//         gid += step;
//     }
// }

struct ChecksResult
{
    bool all_zeros_ref;
    bool all_zeros_uut;
    bool all_finite_and_non_nan_ref;
    bool all_finite_and_non_nan_uut;
};

extern "C" __global__ void VerifyGPUKernel(const FLOAT_UUT* __restrict__ uut,
                                           const FLOAT_REF* __restrict__ ref,
                                           size_t sz,
                                           ChecksResult* res,
                                           FLOAT_ACCUM* error,
                                           [[maybe_unused]] FLOAT_ACCUM* max)
{
    if constexpr(DO_VALIDATE == 1)
    {
        size_t tid  = threadIdx.x;
        size_t gid  = blockIdx.x * blockDim.x + tid;
        size_t step = gridDim.x * blockDim.x;

        __shared__ bool all_zeros_uut_shared;
        __shared__ bool all_zeros_ref_shared;
        __shared__ bool all_valid_uut_shared;
        __shared__ bool all_valid_ref_shared;

        all_zeros_uut_shared = true;
        all_zeros_ref_shared = true;
        all_valid_uut_shared = true;
        all_valid_ref_shared = true;
        __syncthreads();

        // break earlier if all shared variables have changed their values?
        while(gid < sz)
        {
            if(uut[gid] != static_cast<FLOAT_UUT>(0))
            {
                all_zeros_uut_shared = false;
            }

            if(ref[gid] != static_cast<FLOAT_REF>(0))
            {
                all_zeros_ref_shared = false;
            }

            if(isinf(CVT_FLOAT_UUT2ACCUM(uut[gid])) || isnan(CVT_FLOAT_UUT2ACCUM(uut[gid])))
            {
                all_valid_uut_shared = false;
            }

            if(isinf(CVT_FLOAT_REF2ACCUM(ref[gid])) || isnan(CVT_FLOAT_REF2ACCUM(ref[gid])))
            {
                all_valid_ref_shared = false;
            }

            gid += step;
        }

        __syncthreads();
        // avoid multiple writings?
        if(tid == 0)
        {
            if(!all_zeros_uut_shared)
            {
                res->all_zeros_uut = false;
            }

            if(!all_zeros_ref_shared)
            {
                res->all_zeros_ref = false;
            }

            if(!all_valid_uut_shared)
            {
                res->all_finite_and_non_nan_uut = false;
            }

            if(!all_valid_ref_shared)
            {
                res->all_finite_and_non_nan_ref = false;
            }
        }
        __syncthreads();
    }

    if constexpr(CALCULATE_RMS == 1)
    {
        CalculateRMS<BLOCK_SZ>(uut, ref, sz, error, max);
    }

    // if constexpr(CALCULATE_MAE == 1)
    // {
    //     CalculateMAE<BLOCK_SZ, FP_TYPE, FP_TYPE_VERIFY>(uut, ref, sz, error);
    // }

    // if constexpr(FIND_MISMATCH == 1)
    // {
    //     FindMismatch<FP_TYPE, FP_TYPE_VERIFY>(uut, ref, sz, error);
    // }
}
