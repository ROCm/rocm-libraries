// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "RMSNormCommon.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_RMSNORM_INNER_SIZE;
constexpr unsigned int OUTER_SIZE = HIP_PLUGIN_RMSNORM_OUTER_SIZE;
constexpr unsigned int STRIDE = HIP_PLUGIN_RMSNORM_STRIDE;

using XType = HIP_PLUGIN_RMSNORM_X_TYPE;
using DyType = HIP_PLUGIN_RMSNORM_DY_TYPE;
using DxType = HIP_PLUGIN_RMSNORM_DX_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

extern "C" __global__ void RMSnormBwdWeightBias(const DyType* __restrict__ dy,
                                                const XType* __restrict__ x,
                                                const ComputeType* __restrict__ rstd,
                                                ScaleType* __restrict__ dweight,
                                                ScaleType* __restrict__ dbias)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormBwdWeightBias kernel");

    const unsigned int tidx = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(tidx >= INNER_SIZE)
    {
        return;
    }

    float sum_dw = 0.0f;
    float sum_db = 0.0f;

    // backward weight calculation
    for(unsigned int o = 0; o < OUTER_SIZE; ++o)
    {
        for(unsigned int s = 0; s < STRIDE; ++s)
        {
            size_t idx = o * INNER_SIZE * STRIDE + tidx * STRIDE + s;

            float prstd = rstd[o * STRIDE + s];
            float pdy = to_float32<DyType>(dy[idx]);
            float px = to_float32<XType>(x[idx]);

            sum_dw += pdy * px * prstd;
            sum_db += pdy;
        }
    }

    dweight[tidx] = sum_dw;
    if(dbias)
    {
        dbias[tidx] = sum_db;
    }
}

extern "C" __global__ void RMSnormBwdData(const DyType* __restrict__ dy,
                                          const XType* __restrict__ x,
                                          const ScaleType* __restrict__ weight,
                                          const ComputeType* __restrict__ rstd,
                                          DxType* __restrict__ dx)
{
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormBwdData kernel");

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;
    const unsigned int o = gid / STRIDE;
    const unsigned int s = gid % STRIDE;

    __shared__ float ltmp[LOCAL_SIZE];
    float mean = 0.0f;

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = to_float32<DyType>(dy[idx]);
        float px = to_float32<XType>(x[idx]);
        float pw = to_float32<ScaleType>(weight[i]);

        mean += pdy * pw * px;
    }

    ltmp[lid] = mean;
    __syncthreads();

    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    mean = ltmp[0] / INNER_SIZE;
    float prstd = rstd[gid];

    // backward data calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = o * INNER_SIZE * STRIDE + i * STRIDE + s;

        float pdy = to_float32<DyType>(dy[idx]);
        float px = to_float32<XType>(x[idx]);
        float pw = to_float32<ScaleType>(weight[i]);

        float dx_val = (pdy * pw * prstd) - (mean * px * prstd * prstd * prstd);
        dx[idx] = from_float32<DxType>(dx_val);
    }
}
