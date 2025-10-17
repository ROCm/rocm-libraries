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

#include "activation_functions.hpp"
#include "float_types.h"
#include "vector_types.hpp"

// determine block size using parameters passed from the host
constexpr int blockSize = MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2;

// define types for vectorized loads/stores
using FLOAT_VEC_TYPE = typename miopen::mapped_vector_type<FLOAT, MIOPEN_READ_UNIT>::type;

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormActivInferSpatialEst(const FLOAT_ACCUM alpha,
                                        const FLOAT_ACCUM beta,
                                        const FLOAT_ACCUM gamma,
                                        const double epsilon,
                                        const FLOAT* __restrict in,
                                        FLOAT* __restrict out,
                                        const FLOAT_ACCUM* __restrict bias,
                                        const FLOAT_ACCUM* __restrict scale,
                                        const FLOAT_ACCUM* __restrict estimatedMean,
                                        const FLOAT_ACCUM* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    // skip execution for out-of-bound threads
    if(tidx >= MIOPEN_SBN_BOUNDS)
    {
        return;
    }

    unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int c_i      = tidy;
    unsigned int hw_i     = tidx;
    unsigned int c_offset = c_i * MIO_BN_HW;

    // load the mean, variance, scale, and bias that is broadcast across the block
    const FLOAT_ACCUM pmean       = estimatedMean[c_i];
    const FLOAT_ACCUM pvar        = estimatedVariance[c_i];
    const FLOAT_ACCUM pscale      = scale[c_i];
    const FLOAT_ACCUM pbias       = bias[c_i];
    const FLOAT_ACCUM invVariance = rsqrt(pvar + static_cast<FLOAT_ACCUM>(epsilon));

    // load the input data (this is done in a vectorized manner)
    FLOAT data[MIOPEN_READ_UNIT];

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; ++n_i)
    {
        const unsigned int index = n_i * MIO_BN_CHW + c_offset + hw_i * MIOPEN_READ_UNIT;

        // load the input data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(data)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + index));

        // perform batch norm and activation
        FLOAT_ACCUM bnRes[MIOPEN_READ_UNIT];
        FLOAT_ACCUM actRes[MIOPEN_READ_UNIT];
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] =
                fma(pscale, (static_cast<FLOAT_ACCUM>(data[i]) - pmean) * invVariance, pbias);
        }
        ActivationFunction(actRes, bnRes, gamma, beta, alpha);

        // write the output data
        if constexpr(MIOPEN_USE_FP16)
        { // In this situation, FLOAT_ACCUM is FP32 whereas FLOAT is FP16
          // So, we cannot perform a vectorized store
#pragma unroll
            for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
            {
                out[index + i] = static_cast<FLOAT>(actRes[i]);
            }
        }
        else
        {
            // perform a vectorized store of the output data as FLOAT and FLOAT_ACCUM are same
            *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + index)) =
                *(reinterpret_cast<const FLOAT_VEC_TYPE*>(actRes));
        }
    }
}

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormActivInferPerActEst(const FLOAT_ACCUM alpha,
                                       const FLOAT_ACCUM beta,
                                       const FLOAT_ACCUM gamma,
                                       const double epsilon,
                                       const FLOAT* __restrict in,
                                       FLOAT* __restrict out,
                                       const FLOAT_ACCUM* __restrict bias,
                                       const FLOAT_ACCUM* __restrict scale,
                                       const FLOAT_ACCUM* __restrict estimatedMean,
                                       const FLOAT_ACCUM* __restrict estimatedVariance)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
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
#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        pmean[i]  = estimatedMean[chw_i + i];
        pvar[i]   = estimatedVariance[chw_i + i];
        pscale[i] = scale[chw_i + i];
        pbias[i]  = bias[chw_i + i];
    }

    FLOAT data[MIOPEN_READ_UNIT];
    FLOAT_ACCUM invVariance[MIOPEN_READ_UNIT];

#pragma unroll
    for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
    {
        invVariance[i] = rsqrt(pvar[i] + static_cast<FLOAT_ACCUM>(epsilon));
    }

#pragma unroll 2
    for(unsigned int n_i = 0; n_i < MIO_BN_N; ++n_i)
    {
        const unsigned int index = n_i * MIO_BN_CHW + chw_i;

        // load the input data
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(data)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + index));

        // perform batch norm and activation
        FLOAT_ACCUM bnRes[MIOPEN_READ_UNIT];
        FLOAT_ACCUM actRes[MIOPEN_READ_UNIT];
#pragma unroll
        for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
        {
            bnRes[i] = fma(pscale[i],
                           (static_cast<FLOAT_ACCUM>(data[i]) - pmean[i]) * invVariance[i],
                           pbias[i]);
        }
        ActivationFunction(actRes, bnRes, gamma, beta, alpha);

        // write the output data
        if constexpr(MIOPEN_USE_FP16)
        { // In this situation, FLOAT_ACCUM is FP32 whereas FLOAT is FP16
          // So, we cannot perform a vectorized store
#pragma unroll
            for(unsigned int i = 0; i < MIOPEN_READ_UNIT; ++i)
            {
                out[index + i] = static_cast<FLOAT>(actRes[i]);
            }
        }
        else
        {
            // perform a vectorized store of the output data as FLOAT and FLOAT_ACCUM are same
            *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + index)) =
                *(reinterpret_cast<const FLOAT_VEC_TYPE*>(actRes));
        }
    }
}
