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
    MIOpenBatchNormFwdInferPerActivationEst(const FLOAT* __restrict in,
                                            FLOAT* __restrict out,
                                            const FLOAT_ACCUM* __restrict estimatedMean,
                                            const FLOAT_ACCUM* __restrict estimatedVariance,
                                            const FLOAT_ACCUM* __restrict scale,
                                            const FLOAT_ACCUM* __restrict bias,
                                            const double epsilon,
                                            unsigned int c,
                                            unsigned int hw,
                                            unsigned int batchSize,
                                            unsigned int cStride,
                                            unsigned int hwStride,
                                            unsigned int batchStride)
{
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    // decide vector sizes based on problem layout
    unsigned int vecSizeX, vecSizeY;
    if constexpr(MIO_LAYOUT_NHWC)
    {
        vecSizeX = MIO_BN_VEC_SIZE;
        vecSizeY = 1;
    }
    else // NCHW layout
    {
        vecSizeX = 1;
        vecSizeY = MIO_BN_VEC_SIZE;
    }

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= c || tidy * vecSizeY >= hw)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = (tidx * cStride * vecSizeX) + (tidy * hwStride * vecSizeY);

    // batch parameters and values for current thread
    FLOAT_ACCUM mean[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM variance[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM pscale[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM pbias[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM invVariance[MIO_BN_VEC_SIZE];
#pragma unroll
    for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
    {
        mean[i]     = estimatedMean[adjIndex + i];
        variance[i] = estimatedVariance[adjIndex + i];
        pscale[i]   = scale[adjIndex + i];
        pbias[i]    = bias[adjIndex + i];
    }
#pragma unroll
    for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
    {
        invVariance[i] = rsqrt(fabs(variance[i] + static_cast<FLOAT_ACCUM>(epsilon)));
    }
    FLOAT_ACCUM inhat[MIO_BN_VEC_SIZE];
    FLOAT value[MIO_BN_VEC_SIZE];

    // loop over the batches
#pragma unroll 2
    for (unsigned int n = 0; n < MIO_BN_N; ++n)
    {
        // load input value
        const unsigned int batchIndex = (n * batchStride) + adjIndex;
#pragma unroll
        for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
        {
            value[i] = in[batchIndex + i];
        }

        // perform batchnorm operation
#pragma unroll
        for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
        {
            inhat[i] = (static_cast<FLOAT_ACCUM>(value[i]) - mean[i]) * invVariance[i];
            inhat[i] = fma(pscale[i], inhat[i], pbias[i]);
            value[i] = static_cast<FLOAT>(inhat[i]);
        }

        // write output value
#pragma unroll
        for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
        {
            out[batchIndex + i] = value[i];
        }
    }
}
