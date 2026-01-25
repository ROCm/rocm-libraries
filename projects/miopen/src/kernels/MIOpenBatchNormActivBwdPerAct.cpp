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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#include "activation_functions.hpp"

template <typename T>
__forceinline__ __device__ void activbwdperactivation(const T* __restrict__ x,
                                                      const T* __restrict__ y,
                                                      const T* __restrict__ dy,
                                                      T* __restrict__ dx,
                                                      const T diff_scale,
                                                      const T gamma,
                                                      const T beta,
                                                      const T alpha,
                                                      const float* __restrict__ bn_scale,
                                                      const float* __restrict__ bn_bias,
                                                      float* __restrict__ dscale,
                                                      float* __restrict__ dbias,
                                                      const float* __restrict__ saved_mean,
                                                      const float* __restrict__ saved_inv_variance)
{
    const unsigned int ygid      = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;
    const unsigned int adj_index = MIO_BN_HW * blockIdx.x + ygid;

    if(adj_index < MIO_BN_CHW)
    {
        const FLOAT_ACCUM mean      = CVT_FP32_2ACCUM(saved_mean[adj_index]);
        const FLOAT_ACCUM inv_var   = CVT_FP32_2ACCUM(saved_inv_variance[adj_index]);
        const FLOAT_ACCUM pvt_scale = CVT_FP32_2ACCUM(bn_scale[adj_index]);
        const FLOAT_ACCUM pvt_bias  = CVT_FP32_2ACCUM(bn_bias[adj_index]);
        FLOAT_ACCUM pvt_dscale{0};
        FLOAT_ACCUM pvt_dbias{0};
        FLOAT_ACCUM dxhat{0};
        FLOAT_ACCUM dxhathat{0};

        for(unsigned int n = 0; n < MIO_BN_N; ++n)
        {
            // per (x-dims) channel load a block of data into LDS
            unsigned int index          = n * MIO_BN_CHW + adj_index;
            const FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_var;
            const FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
            const FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
            const FLOAT_ACCUM bn_y[1]   = {xhat * pvt_scale + pvt_bias};
            FLOAT_ACCUM bn_dy[1];
            ActivationFunction_Diff(bn_dy,
                                    act_dy,
                                    bn_y,
                                    act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));
            pvt_dbias += bn_dy[0];
            pvt_dscale += xhat * bn_dy[0];
            FLOAT_ACCUM tmp = pvt_scale * bn_dy[0];
            dxhat += tmp;
            dxhathat += tmp * xhat;
        }

        for(unsigned int n = 0; n < MIO_BN_N; ++n)
        {
            unsigned int index          = n * MIO_BN_CHW + adj_index;
            const FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_var;
            FLOAT_ACCUM tmp             = xhat * dxhathat + dxhat;
            const FLOAT_ACCUM bn_y[1]   = {xhat * pvt_scale + pvt_bias};
            const FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
            const FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
            FLOAT_ACCUM bn_dy[1];
            ActivationFunction_Diff(bn_dy,
                                    act_dy,
                                    bn_y,
                                    act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));
            dx[index] = CVT_ACCUM2FLOAT(
                (bn_dy[0] * pvt_scale - tmp * static_cast<FLOAT_ACCUM>(1.0f / MIO_BN_N)) * inv_var);
        }

        // Write out data
        dbias[adj_index]  = CVT_ACCUM2FP32(pvt_dbias);
        dscale[adj_index] = CVT_ACCUM2FP32(pvt_dscale);
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) //
    void MIOpenBatchNormActivBwdPerActivation(const DATA_TYPE* __restrict__ x,
                                              const DATA_TYPE* __restrict__ y,
                                              const DATA_TYPE* __restrict__ dy,
                                              DATA_TYPE* __restrict__ dx,
                                              const DATA_TYPE diff_scale,
                                              const DATA_TYPE gamma,
                                              const DATA_TYPE beta,
                                              const DATA_TYPE alpha,
                                              const float* __restrict__ bn_scale,
                                              const float* __restrict__ bn_bias,
                                              float* __restrict__ dscale,
                                              float* __restrict__ dbias,
                                              const float* __restrict__ saved_mean,
                                              const float* __restrict__ saved_inv_variance)
{
    activbwdperactivation<DATA_TYPE>(x,
                                     y,
                                     dy,
                                     dx,
                                     diff_scale,
                                     gamma,
                                     beta,
                                     alpha,
                                     bn_scale,
                                     bn_bias,
                                     dscale,
                                     dbias,
                                     saved_mean,
                                     saved_inv_variance);
}
