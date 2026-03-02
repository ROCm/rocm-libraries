// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "FloatTypes.h"

constexpr int blockSize = HIP_PLUGIN_BN_GRP0 * HIP_PLUGIN_BN_GRP1 * HIP_PLUGIN_BN_GRP2;

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormBwdPerActivationSaved(const FLOAT* __restrict in,
                                   const FLOAT* __restrict dy_in,
                                   unsigned int N,
                                   unsigned int in_nstride,
                                   unsigned int in_cstride,
                                   FLOAT* __restrict dx_out,
                                   const FLOAT_ACCUM* __restrict scale,
                                   FLOAT_ACCUM* __restrict delta_scale,
                                   FLOAT_ACCUM* __restrict delta_bias,
                                   const FLOAT_ACCUM* __restrict savedMean,
                                   const FLOAT_ACCUM* __restrict savedInvVariance)
{
    unsigned int xgid = blockIdx.x * HIP_PLUGIN_BN_GRP0 + threadIdx.x;
    unsigned int ygid = blockIdx.y * HIP_PLUGIN_BN_GRP1 + threadIdx.y;

    if(xgid >= HIP_PLUGIN_BN_C || ygid >= HIP_PLUGIN_BN_HW)
    {
        return;
    }

    unsigned int yglb_sz = HIP_PLUGIN_BN_GRP1 * gridDim.y;
    int cidx = in_cstride * xgid;
    FLOAT_ACCUM N_float_accum = CVT_INTEGRAL2ACCUM(N);

    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        unsigned int adjIndex = cidx + idx;
        unsigned int index;
        FLOAT_ACCUM mean = savedMean[adjIndex];
        FLOAT_ACCUM invVar = savedInvVariance[adjIndex];
        FLOAT_ACCUM pvt_scale = scale[adjIndex];
        FLOAT_ACCUM pvt_dscale = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM pvt_dbias = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM dxhat = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM dxhathat = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM xhat, dyelem;
        FLOAT_ACCUM tmp1, tmp2, tmp3;

        for(unsigned int batchIdx = 0; batchIdx < N; batchIdx++)
        {
            index = in_nstride * batchIdx + adjIndex;
            xhat = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            dyelem = CVT_FLOAT2ACCUM(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = fma(xhat, dyelem, pvt_dscale);
            tmp1 = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = fma(tmp1, xhat, dxhathat);
        }

        for(unsigned int batchIdx = 0; batchIdx < N; batchIdx++)
        {
            index = in_nstride * batchIdx + adjIndex;
            xhat = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            tmp1 = fma(xhat, dxhathat, dxhat);
            tmp2 = fma(N_float_accum, CVT_FLOAT2ACCUM(dy_in[index]) * pvt_scale, -tmp1);
            tmp3 = invVar / N_float_accum;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * tmp2);
        }

        delta_bias[adjIndex] = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    }
}
