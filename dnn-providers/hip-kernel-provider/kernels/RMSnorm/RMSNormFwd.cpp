// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "CstdInt.hpp"
#include "FloatTypes.h"

constexpr unsigned LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr uint64_t C_SIZE = HIP_PLUGIN_RMSNORM_C_SIZE;
constexpr uint64_t C_STRIDE = HIP_PLUGIN_RMSNORM_C_STRIDE;
constexpr uint64_t N_STRIDE = C_SIZE * C_STRIDE;

using IOType = HIP_PLUGIN_RMSNORM_IO_TYPE;

extern "C" __global__ void RMSnormFwd(const IOType* __restrict__ x,
                                      const FLOAT_ACCUM* __restrict__ weight,
                                      const FLOAT_ACCUM* __restrict__ bias,
                                      IOType* __restrict__ y,
                                      FLOAT_ACCUM* __restrict__ rstd,
                                      float eps)
{
    const uint64_t gid = blockIdx.x;
    const uint64_t lid = threadIdx.x;
    const uint64_t o = gid / C_STRIDE;
    const uint64_t s = gid % C_STRIDE;

    FLOAT_ACCUM pvar(0);
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    // reduce sum
    for(uint64_t i = lid; i < C_SIZE; i += LOCAL_SIZE)
    {
        size_t x_idx = (o * N_STRIDE) + (i * C_STRIDE) + s;
        FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[x_idx]);
        pvar += tmp * tmp;
    }

    ltmp[lid] = pvar;
    __syncthreads();
    for(uint32_t i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    pvar = ltmp[0] / C_SIZE;
    FLOAT_ACCUM prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // forward calculation
    for(uint64_t i = lid; i < C_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = (o * N_STRIDE) + (i * C_STRIDE) + s;
        FLOAT_ACCUM y_val = (CVT_FLOAT2ACCUM(x[idx])) * prstd * weight[i];
        if(bias != nullptr)
        {
            y_val += bias[i];
        }
        y[idx] = CVT_ACCUM2FLOAT(y_val);
    }
}
