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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
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
    unsigned int xgid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid    = blockIdx.y * blockDim.y + threadIdx.y;

    // skip execution for out-of-bound threads
    if(xgid >= MIO_BN_C || ygid >= MIO_BN_HW)
    {
        return;
    }

    unsigned int yglb_sz = blockDim.y * gridDim.y;
    int cidx             = in_cstride * xgid;

    unsigned int index, adjIndex;
    FLOAT_ACCUM mean, invVar;
    FLOAT_ACCUM xhat, dyelem;
    FLOAT_ACCUM pvt_scale, pvt_dscale;
    FLOAT_ACCUM pvt_dbias;
    FLOAT_ACCUM tmp1, tmp2, tmp3;
    FLOAT_ACCUM dxhat    = CVT_FP32_2ACCUM(static_cast<float>(0.0));
    FLOAT_ACCUM dxhathat = CVT_FP32_2ACCUM(static_cast<float>(0.0));

    // move across the sections of an image in the mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        adjIndex   = cidx + idx;
        mean       = savedMean[adjIndex];
        invVar     = savedInvVariance[adjIndex];
        pvt_scale  = scale[adjIndex];
        pvt_dscale = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        pvt_dbias  = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        dxhat      = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        dxhathat   = CVT_FP32_2ACCUM(static_cast<float>(0.0));

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
            index = in_nstride * n + adjIndex;
            xhat  = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            tmp1  = fma(xhat, dxhathat, dxhat);
            tmp2  = fma(CVT_INTEGRAL2ACCUM(N), CVT_FLOAT2ACCUM(dy_in[index]) * pvt_scale, -tmp1);
            tmp3  = invVar / (CVT_INTEGRAL2ACCUM(N));
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
    unsigned int xgid    = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid    = blockIdx.y * blockDim.y + threadIdx.y;

    // skip execution for out-of-bound threads
    if(xgid >= MIO_BN_C || ygid >= MIO_BN_HW)
    {
        return;
    }

    unsigned int yglb_sz = blockDim.y * gridDim.y;
    int cidx             = in_cstride * xgid;

    unsigned int index, adjIndex;
    FLOAT_ACCUM mean, invVar;
    FLOAT_ACCUM xhat, dyelem;
    FLOAT_ACCUM pvt_scale, pvt_dscale;
    FLOAT_ACCUM pvt_dbias;
    FLOAT_ACCUM tmp1, tmp2, tmp3;
    FLOAT_ACCUM dxhat    = CVT_FP32_2ACCUM(static_cast<float>(0.0));
    FLOAT_ACCUM dxhathat = CVT_FP32_2ACCUM(static_cast<float>(0.0));
    FLOAT_ACCUM variance = CVT_FP32_2ACCUM(static_cast<float>(0.0));

    // move across the sections of the image mini_batch stack
    for(int idx = ygid; idx < in_cstride; idx += yglb_sz)
    {
        mean     = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        adjIndex = cidx + idx; // gamma and beta tensor index
        for(int n = 0; n < MIO_BN_N; n++)
        {
            index = in_nstride * n + adjIndex;
            mean += CVT_FLOAT2ACCUM(in[index]);
        }
        mean /= CVT_INTEGRAL2ACCUM(N);
        variance = CVT_FP32_2ACCUM(static_cast<float>(0.0));

        for(int n = 0; n < MIO_BN_N; n++)
        {
            index             = in_nstride * n + adjIndex;
            FLOAT_ACCUM xdiff = CVT_FLOAT2ACCUM(in[index]) - mean;
            variance += (xdiff * xdiff);
        }
        variance /= CVT_INTEGRAL2ACCUM(N);
        invVar = rsqrt(variance + epsilon);

        pvt_scale  = scale[adjIndex];
        pvt_dscale = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        pvt_dbias  = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        dxhat      = CVT_FP32_2ACCUM(static_cast<float>(0.0));
        dxhathat   = CVT_FP32_2ACCUM(static_cast<float>(0.0));

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
            index = in_nstride * n + adjIndex;
            xhat  = (CVT_FLOAT2ACCUM(in[index]) - mean) * invVar;
            tmp1  = fma(xhat, dxhathat, dxhat);
            tmp2  = fma(CVT_INTEGRAL2ACCUM(N), CVT_FLOAT2ACCUM(dy_in[index]) * pvt_scale, -tmp1);
            tmp3  = invVar / (CVT_INTEGRAL2ACCUM(N));
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * tmp2);
        }

        // write out data
        delta_bias[adjIndex]  = pvt_dbias;
        delta_scale[adjIndex] = pvt_dscale;
    } // end for(idx) // image mini_batch is processed
}
