/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
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

#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_runtime.h>
#endif
#include "float_types.h"

// determine block size using parameters passed from the host
constexpr int blockSize = MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2;

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormBwdPerActivationSaved(const FLOAT* __restrict in,
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
    unsigned int xgid = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    unsigned int ygid = blockIdx.y * MIO_BN_GRP1 + threadIdx.y;

    // skip execution for out-of-bound threads
    if(xgid >= MIO_BN_C || ygid >= MIO_BN_HW)
    {
        return;
    }

    unsigned int yglb_sz      = MIO_BN_GRP1 * gridDim.y;
    int cidx                  = in_cstride * xgid;
    FLOAT_ACCUM N_float_accum = CVT_INTEGRAL2ACCUM(N);

    // move across the sections of an image in the mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        unsigned int adjIndex = cidx + idx;
        unsigned int index;
        FLOAT_ACCUM mean       = savedMean[adjIndex];
        FLOAT_ACCUM invVar     = savedInvVariance[adjIndex];
        FLOAT_ACCUM pvt_scale  = scale[adjIndex];
        FLOAT_ACCUM pvt_dscale = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM pvt_dbias  = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM dxhat      = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM dxhathat   = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM xhat, dyelem;
        FLOAT_ACCUM tmp1, tmp2, tmp3;

        for(int n = 0; n < N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            dyelem = CVT_FLOAT2ACCUM(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = fma(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = fma(tmp1, xhat, dxhathat);
        }

        for(int n = 0; n < N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            tmp1          = fma(xhat, dxhathat, dxhat);
            tmp2          = fma(N_float_accum, CVT_FLOAT2ACCUM(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / N_float_accum;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * tmp2);
        }

        // write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(img_offset) // image mini_batch is processed
}

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormBwdPerActivation(const FLOAT* __restrict in,
                                    const FLOAT* __restrict dy_in,
                                    unsigned int N,
                                    unsigned int in_nstride,
                                    unsigned int in_cstride,
                                    FLOAT* __restrict dx_out,
                                    const FLOAT_ACCUM* __restrict scale,
                                    FLOAT_ACCUM* __restrict delta_scale,
                                    FLOAT_ACCUM* __restrict delta_bias,
                                    double epsilon)
{
    unsigned int xgid = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    unsigned int ygid = blockIdx.y * MIO_BN_GRP1 + threadIdx.y;

    // skip execution for out-of-bound threads
    if(xgid >= MIO_BN_C || ygid >= MIO_BN_HW)
    {
        return;
    }

    unsigned int yglb_sz      = MIO_BN_GRP1 * gridDim.y;
    int cidx                  = in_cstride * xgid;
    FLOAT_ACCUM N_float_accum = CVT_INTEGRAL2ACCUM(N);

    // move across the sections of the image mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        FLOAT_ACCUM mean      = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        unsigned int adjIndex = cidx + idx; // gamma and beta tensor index
        unsigned int index;
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = in_nstride * n + adjIndex;
            mean += CVT_FLOAT2ACCUM(in[index]);
        }
        mean /= N_float_accum;

        FLOAT_ACCUM variance = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index             = in_nstride * n + adjIndex;
            FLOAT_ACCUM xdiff = CVT_FLOAT2ACCUM(in[index]) - mean;
            variance += (xdiff * xdiff);
        }
        variance /= N_float_accum;
        FLOAT_ACCUM invVar = rsqrt(variance + epsilon);

        FLOAT_ACCUM pvt_scale  = scale[adjIndex];
        FLOAT_ACCUM pvt_dscale = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM pvt_dbias  = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM dxhat      = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM dxhathat   = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        FLOAT_ACCUM xhat, dyelem;
        FLOAT_ACCUM tmp1, tmp2, tmp3;

        for(int n = 0; n < MIO_BN_N; n++)
        {
            // per (x-dims) channel load a block of data into LDS
            index  = in_nstride * n + adjIndex;
            xhat   = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            dyelem = CVT_FLOAT2ACCUM(dy_in[index]);
            pvt_dbias += dyelem;
            pvt_dscale = fma(xhat, dyelem, pvt_dscale);
            tmp1       = pvt_scale * dyelem;
            dxhat += tmp1;
            dxhathat = fma(tmp1, xhat, dxhathat);
        }

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index         = in_nstride * n + adjIndex;
            xhat          = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            tmp1          = fma(xhat, dxhathat, dxhat);
            tmp2          = fma(N_float_accum, CVT_FLOAT2ACCUM(dy_in[index]) * pvt_scale, -tmp1);
            tmp3          = invVar / N_float_accum;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * tmp2);
        }

        // write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(idx) // image mini_batch is processed
}
