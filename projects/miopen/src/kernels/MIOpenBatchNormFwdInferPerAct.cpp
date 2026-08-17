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
#include "vector_types.hpp"

// determine block size using parameters passed from the host
constexpr int blockSize = MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2;

// define types for vectorized loads/stores
using FLOAT_VEC_TYPE = typename miopen::mapped_vector_type<FLOAT, MIO_BN_VEC_SIZE>::type;
using FLOAT_ACCUM_VEC_TYPE =
    typename miopen::mapped_vector_type<FLOAT_ACCUM, MIO_BN_VEC_SIZE>::type;

__device__ __forceinline__ void BNFwdInferPerActivationImpl(unsigned int adjIndex,
                                                            const FLOAT* in,
                                                            FLOAT* out,
                                                            const FLOAT_ACCUM* mean,
                                                            const FLOAT_ACCUM* invVariance,
                                                            const FLOAT_ACCUM* scale,
                                                            const FLOAT_ACCUM* bias,
                                                            unsigned int batchSize,
                                                            unsigned int batchStride)
{
    FLOAT_ACCUM inhat[MIO_BN_VEC_SIZE];
    FLOAT value[MIO_BN_VEC_SIZE];

    // loop over the batches
    // NOTE: We use zlocalsize = 1 and zgridsize = min(batchSize, maxGridSizeToFillTheGPU). So the
    // idea here is to use the blocks in z-dimension to cover the batch dimension first, and then
    // each block will loop over the remaining batches with stride of gridDim.z if necessary.
    for(unsigned int n = blockIdx.z; n < batchSize; n += gridDim.z)
    {
        // load input value
        const unsigned int batchIndex = (n * batchStride) + adjIndex;
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(value)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + batchIndex));

        // perform batchnorm operation
#pragma unroll
        for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
        {
            inhat[i] = (CVT_FLOAT2ACCUM(value[i]) - mean[i]) * invVariance[i];
            inhat[i] = scale[i] * inhat[i] + bias[i];
            value[i] = CVT_ACCUM2FLOAT(inhat[i]);
        }

        // write output value
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + batchIndex)) =
            *(reinterpret_cast<const FLOAT_VEC_TYPE*>(value));
    }
}

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
    unsigned int tidx = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * MIO_BN_GRP1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX = MIO_LAYOUT_NHWC ? MIO_BN_VEC_SIZE : 1;
    constexpr unsigned int vecSizeY = MIO_LAYOUT_NHWC ? 1 : MIO_BN_VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= c || tidy * vecSizeY >= hw || tidz >= batchSize)
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
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(mean)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(variance)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedVariance + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + adjIndex));
#pragma unroll
    for(unsigned int i = 0; i < MIO_BN_VEC_SIZE; ++i)
    {
        invVariance[i] = rsqrt(fabs(variance[i] + static_cast<FLOAT_ACCUM>(epsilon)));
    }
    BNFwdInferPerActivationImpl(
        adjIndex, in, out, mean, invVariance, pscale, pbias, batchSize, batchStride);
}

// Uses estimated inverse variance rather than inverse variance, which avoids need for an
// epsilon parameter and rsqrt() operations.
extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormFwdInferPerActivationEstInvVar(
        const FLOAT* __restrict in,
        FLOAT* __restrict out,
        const FLOAT_ACCUM* __restrict estimatedMean,
        const FLOAT_ACCUM* __restrict estimatedInvVariance,
        const FLOAT_ACCUM* __restrict scale,
        const FLOAT_ACCUM* __restrict bias,
        unsigned int c,
        unsigned int hw,
        unsigned int batchSize,
        unsigned int cStride,
        unsigned int hwStride,
        unsigned int batchStride)
{
    unsigned int tidx = blockIdx.x * MIO_BN_GRP0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * MIO_BN_GRP1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX = MIO_LAYOUT_NHWC ? MIO_BN_VEC_SIZE : 1;
    constexpr unsigned int vecSizeY = MIO_LAYOUT_NHWC ? 1 : MIO_BN_VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= c || tidy * vecSizeY >= hw || tidz >= batchSize)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = (tidx * cStride * vecSizeX) + (tidy * hwStride * vecSizeY);

    // batch parameters and values for current thread
    FLOAT_ACCUM mean[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM pscale[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM pbias[MIO_BN_VEC_SIZE];
    FLOAT_ACCUM invVariance[MIO_BN_VEC_SIZE];
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(mean)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(invVariance)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedInvVariance + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + adjIndex));
    *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias)) =
        *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + adjIndex));

    BNFwdInferPerActivationImpl(
        adjIndex, in, out, mean, invVariance, pscale, pbias, batchSize, batchStride);
}
