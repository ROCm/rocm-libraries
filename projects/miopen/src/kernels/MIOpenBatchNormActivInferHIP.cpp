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

#include "activation_functions.hpp"
#include "float_types.h"
#include "vector_types.hpp"

// determine block size using parameters passed from the host
constexpr int blockSize = MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2;

// define types for vectorized loads/stores
using FLOAT_VEC_TYPE = typename miopen::mapped_vector_type<FLOAT, MIOPEN_READ_UNIT>::type;
using FLOAT_ACCUM_VEC_TYPE =
    typename miopen::mapped_vector_type<FLOAT_ACCUM, MIOPEN_READ_UNIT>::type;

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormActivInferSpatialEst(const FLOAT alpha,
                                        const FLOAT beta,
                                        const FLOAT gamma,
                                        const double epsilon,
                                        const FLOAT* __restrict in,
                                        FLOAT* __restrict out,
                                        const FLOAT_ACCUM* __restrict bias,
                                        const FLOAT_ACCUM* __restrict scale,
                                        const FLOAT_ACCUM* __restrict estimatedMean,
                                        const FLOAT_ACCUM* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    // skip execution for out-of-bound threads
    if(tidx >= MIOPEN_SBN_BOUNDS)
    {
        return;
    }

    unsigned int tidy = blockIdx.y * MIO_BN_GRP1 + threadIdx.y;

    unsigned int c_i, hw_i, c_offset, hw_offset;
    if constexpr(MIO_LAYOUT_NHWC)
    {
        c_i       = tidx;
        hw_i      = tidy;
        c_offset  = c_i * MIOPEN_READ_UNIT;
        hw_offset = hw_i * MIO_BN_C;
    }
    else
    {
        c_i       = tidy;
        hw_i      = tidx;
        c_offset  = c_i * MIO_BN_HW;
        hw_offset = hw_i * MIOPEN_READ_UNIT;
    }

    // load the mean, variance, scale, and bias
    FLOAT_ACCUM pmean[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pvar[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pscale[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pbias[MIOPEN_READ_UNIT];
    FLOAT_ACCUM invVariance[MIOPEN_READ_UNIT];
    if constexpr(MIO_LAYOUT_NHWC)
    {
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pmean)) =
            *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + c_offset));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pvar)) =
            *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedVariance + c_offset));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale)) =
            *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + c_offset));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias)) =
            *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + c_offset));
    }
    else // NCHW layout
    {
        const auto mean_val  = estimatedMean[c_i];
        const auto var_val   = estimatedVariance[c_i];
        const auto scale_val = scale[c_i];
        const auto bias_val  = bias[c_i];
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            pmean[i]  = mean_val;
            pvar[i]   = var_val;
            pscale[i] = scale_val;
            pbias[i]  = bias_val;
        }
    }
#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        invVariance[i] = rsqrt(pvar[i] + static_cast<FLOAT_ACCUM>(epsilon));
    }

    FLOAT data[MIOPEN_READ_UNIT];
    FLOAT_ACCUM bnRes[MIOPEN_READ_UNIT];
    FLOAT_ACCUM actRes[MIOPEN_READ_UNIT];

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; ++n_i)
    {
        const unsigned int index = n_i * MIO_BN_CHW + c_offset + hw_offset;

        // load the input data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(data)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + index));

        // perform batch norm and activation
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] =
                pscale[i] * (CVT_FLOAT2ACCUM(data[i]) - pmean[i]) * invVariance[i] + pbias[i];
        }
        ActivationFunction(
            actRes, bnRes, CVT_FLOAT2ACCUM(gamma), CVT_FLOAT2ACCUM(beta), CVT_FLOAT2ACCUM(alpha));
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            data[i] = CVT_ACCUM2FLOAT(actRes[i]);
        }

        // write the output data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + index)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(data));
    }
}

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormActivInferPerActEst(const FLOAT alpha,
                                       const FLOAT beta,
                                       const FLOAT gamma,
                                       const double epsilon,
                                       const FLOAT* __restrict in,
                                       FLOAT* __restrict out,
                                       const FLOAT_ACCUM* __restrict bias,
                                       const FLOAT_ACCUM* __restrict scale,
                                       const FLOAT_ACCUM* __restrict estimatedMean,
                                       const FLOAT_ACCUM* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    // skip execution for out-of-bound threads
    if(tidx >= MIOPEN_SBN_BOUNDS)
    {
        return;
    }

    unsigned int chw_i = tidx * MIOPEN_READ_UNIT;

    // load the mean, variance, scale, and bias
    FLOAT_ACCUM pmean[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pvar[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pscale[MIOPEN_READ_UNIT];
    FLOAT_ACCUM pbias[MIOPEN_READ_UNIT];
    FLOAT_ACCUM invVariance[MIOPEN_READ_UNIT];
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pmean)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + chw_i));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pvar)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedVariance + chw_i));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + chw_i));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + chw_i));

#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        invVariance[i] = rsqrt(pvar[i] + static_cast<FLOAT_ACCUM>(epsilon));
    }

    FLOAT data[MIOPEN_READ_UNIT];
    FLOAT_ACCUM bnRes[MIOPEN_READ_UNIT];
    FLOAT_ACCUM actRes[MIOPEN_READ_UNIT];

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; ++n_i)
    {
        const unsigned int index = n_i * MIO_BN_CHW + chw_i;

        // load the input data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(data)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + index));

        // perform batch norm and activation
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] =
                pscale[i] * (CVT_FLOAT2ACCUM(data[i]) - pmean[i]) * invVariance[i] + pbias[i];
        }
        ActivationFunction(
            actRes, bnRes, CVT_FLOAT2ACCUM(gamma), CVT_FLOAT2ACCUM(beta), CVT_FLOAT2ACCUM(alpha));
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            data[i] = CVT_ACCUM2FLOAT(actRes[i]);
        }

        // write the output data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + index)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(data));
    }
}
