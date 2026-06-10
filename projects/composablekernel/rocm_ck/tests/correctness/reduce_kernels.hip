// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Reduction kernels for split-accumulation GEMM.
//
// Each kernel reads partial products from K-tiles and reduces them into
// a single output matrix. The three strategies differ only in how they
// accumulate the partial sums, allowing direct comparison of precision
// characteristics.

#include "split_gemm.hpp"

#include "ck_tile/host/hip_check_error.hpp"

#include <hip/hip_runtime.h>

namespace rocm_ck::test {

__global__ void reduce_naive_kernel(const float* __restrict__ partials,
                                    float* __restrict__ C,
                                    int MN,
                                    int num_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= MN)
        return;
    float acc = 0.0f;
    for(int t = 0; t < num_tiles; ++t)
        acc += partials[t * MN + idx];
    C[idx] = acc;
}

__global__ void reduce_compensated_kernel(const float* __restrict__ partials,
                                          float* __restrict__ C,
                                          int MN,
                                          int num_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= MN)
        return;
    float sum  = 0.0f;
    float comp = 0.0f;
    for(int t = 0; t < num_tiles; ++t)
    {
        float y     = partials[t * MN + idx] - comp;
        float t_val = sum + y;
        comp        = (t_val - sum) - y;
        sum         = t_val;
    }
    C[idx] = sum;
}

__global__ void reduce_twosum_kernel(const float* __restrict__ partials,
                                     float* __restrict__ C,
                                     int MN,
                                     int num_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= MN)
        return;
    float sum = 0.0f;
    float err = 0.0f;
    for(int t = 0; t < num_tiles; ++t)
    {
        float delta = partials[t * MN + idx];
        float s     = sum + delta;
        float v     = s - sum;
        err += (sum - (s - v)) + (delta - v);
        sum = s;
    }
    C[idx] = sum + err;
}

__global__ void reduce_pairwise_kernel(const float* __restrict__ partials,
                                       float* __restrict__ C,
                                       int MN,
                                       int num_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= MN)
        return;
    float vals[16];
    for(int t = 0; t < num_tiles; ++t)
        vals[t] = partials[t * MN + idx];
    for(int stride = 1; stride < num_tiles; stride *= 2)
        for(int i = 0; i + stride < num_tiles; i += stride * 2)
            vals[i] += vals[i + stride];
    C[idx] = vals[0];
}

__global__ void reduce_fp64_kernel(const float* __restrict__ partials,
                                   float* __restrict__ C,
                                   int MN,
                                   int num_tiles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= MN)
        return;
    double acc = 0.0;
    for(int t = 0; t < num_tiles; ++t)
        acc += static_cast<double>(partials[t * MN + idx]);
    C[idx] = static_cast<float>(acc);
}

void launch_reduce(ReduceStrategy strategy,
                   const float* partials,
                   float* C,
                   int MN,
                   int num_tiles,
                   hipStream_t stream)
{
    constexpr int kBlock = 256;
    int grid             = (MN + kBlock - 1) / kBlock;

    switch(strategy)
    {
    case ReduceStrategy::NaiveSum:
        reduce_naive_kernel<<<grid, kBlock, 0, stream>>>(partials, C, MN, num_tiles);
        break;
    case ReduceStrategy::CompensatedSum:
        reduce_compensated_kernel<<<grid, kBlock, 0, stream>>>(partials, C, MN, num_tiles);
        break;
    case ReduceStrategy::TwoSumReduce:
        reduce_twosum_kernel<<<grid, kBlock, 0, stream>>>(partials, C, MN, num_tiles);
        break;
    case ReduceStrategy::PairwiseSum:
        reduce_pairwise_kernel<<<grid, kBlock, 0, stream>>>(partials, C, MN, num_tiles);
        break;
    case ReduceStrategy::FP64Sum:
        reduce_fp64_kernel<<<grid, kBlock, 0, stream>>>(partials, C, MN, num_tiles);
        break;
    }
    HIP_CHECK_ERROR(hipGetLastError());
}

} // namespace rocm_ck::test
