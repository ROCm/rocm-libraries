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
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

#include "activation_functions.hpp"

template <typename TI, typename TO>
__device__ void activbwdperactivation(const TI* __restrict__ x,
                                      const TI* __restrict__ y,
                                      const TI* __restrict__ dy,
                                      TO* __restrict__ dx,
                                      const float diff_scale,
                                      const float gamma,
                                      const float beta,
                                      const float alpha,
                                      const TI* __restrict__ bn_scale,
                                      const TI* __restrict__ bn_bias,
                                      TO* __restrict__ dscale,
                                      TO* __restrict__ dbias,
                                      const TI* __restrict__ saved_mean,
                                      const TI* __restrict__ saved_inv_variance)
{
    auto xgid    = blockIdx.x * LOCAL_SIZE_X + threadIdx.x;
    auto ygid    = blockIdx.y * LOCAL_SIZE_Y + threadIdx.y;
    auto c_index = H * W * xgid;

    // move across the sections of an image in the mini_batch stack
    for(auto img_offset = 0; img_offset < H * W; img_offset += LOCAL_SIZE_Y)
    {
        auto img_index = img_offset + ygid;
        if(img_index < H * W)
        {
            auto adj_index = c_index + img_index; // gamma and beta tensor index
            auto mean      = CVT_FLOAT2ACCUM(saved_mean[adj_index]);
            auto inv_var   = CVT_FLOAT2ACCUM(saved_inv_variance[adj_index]);
            auto pvt_scale = CVT_FLOAT2ACCUM(bn_scale[adj_index]);
            auto pvt_bias  = CVT_FLOAT2ACCUM(bn_bias[adj_index]);
            FLOAT_ACCUM pvt_dscale{0};
            FLOAT_ACCUM pvt_dbias{0};
            FLOAT_ACCUM dxhat{0};
            FLOAT_ACCUM dxhathat{0};

            for(auto n = 0; n < BATCH_SIZE; ++n)
            {
                // per (x-dims) channel load a block of data into LDS
                auto index            = n * CHANNELS * H * W + adj_index;
                auto xhat             = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_var;
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                FLOAT_ACCUM bn_y[1]   = {xhat * pvt_scale + pvt_bias};
                FLOAT_ACCUM bn_dy[1];
                ActivationFunction_Diff(bn_dy, act_dy, bn_y, act_y, diff_scale, gamma, beta, alpha);
                pvt_dbias += bn_dy[0];
                pvt_dscale = xhat * bn_dy[0] + pvt_dscale;
                auto tmp   = pvt_scale * bn_dy[0];
                dxhat += tmp;
                dxhathat += tmp * xhat;
            }

            for(auto n = 0; n < BATCH_SIZE; ++n)
            {
                auto index            = n * CHANNELS * H * W + adj_index;
                auto xhat             = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_var;
                auto tmp              = xhat * dxhathat + dxhat;
                FLOAT_ACCUM bn_y[1]   = {xhat * pvt_scale + pvt_bias};
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                FLOAT_ACCUM bn_dy[1];
                ActivationFunction_Diff(bn_dy, act_dy, bn_y, act_y, diff_scale, gamma, beta, alpha);
                auto tmp2 = BATCH_SIZE * bn_dy[0] * pvt_scale - tmp;
                auto tmp3 = inv_var / BATCH_SIZE;
                dx[index] = CVT_ACCUM2FLOAT(tmp2 * tmp3);
            }

            // Write out data
            dbias[adj_index]  = CVT_ACCUM2FLOAT(pvt_dbias);
            dscale[adj_index] = CVT_ACCUM2FLOAT(pvt_dscale);
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) //
    void ActivBwdPerActivation(const INPUT_TYPE* __restrict__ x,
                               const INPUT_TYPE* __restrict__ y,
                               const INPUT_TYPE* __restrict__ dy,
                               OUTPUT_TYPE* __restrict__ dx,
                               const float diff_scale,
                               const float gamma,
                               const float beta,
                               const float alpha,
                               const INPUT_TYPE* __restrict__ bn_scale,
                               const INPUT_TYPE* __restrict__ bn_bias,
                               OUTPUT_TYPE* __restrict__ dscale,
                               OUTPUT_TYPE* __restrict__ dbias,
                               const INPUT_TYPE* __restrict__ saved_mean,
                               const INPUT_TYPE* __restrict__ saved_inv_variance)
{
    activbwdperactivation<INPUT_TYPE, OUTPUT_TYPE>(x,
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
