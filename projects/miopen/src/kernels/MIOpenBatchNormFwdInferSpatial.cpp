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

#include "batchnorm_functions.h"
#include "bnorm_spatial_activation_functions.h"

// determine block size using parameters passed from the host
constexpr int blockSize = MIO_BN_GRP0 * MIO_BN_GRP1 * MIO_BN_GRP2;

extern "C" __global__ void __launch_bounds__(blockSize)
    MIOpenBatchNormFwdInferSpatialEst(const __global _FLOAT* __restrict in,
                                      __global _FLOAT* __restrict out,
                                      const __global _FLOAT_PREC* __restrict estimatedMean,
                                      const __global _FLOAT_PREC* __restrict estimatedVariance,
                                      const __global _FLOAT_PREC* __restrict scale,
                                      const __global _FLOAT_PREC* __restrict bias,
                                      double epsilon,
                                      unsigned int c,
                                      unsigned int hw,
                                      unsigned int batchSize,
                                      unsigned int cStride,
                                      unsigned int hwStride,
                                      unsigned int batchStride,
                                      _FLOAT_PREC _alpha,
                                      _FLOAT_PREC _beta)
{
    ACTIVATION_SET();
    unsigned int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // skip execution for out-of-bound threads
    if(tidx * VEC_SIZE_X >= c || tidy * VEC_SIZE_Y >= hw)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = tidx * VEC_SIZE_X;
    unsigned int batchIndex = (tidx * cStride * VEC_SIZE_X) + (tidy * hwStride * VEC_SIZE_Y);

    // batch parameters and values for current thread
    _FLOAT_PREC_C mean = *(reinterpret_cast<const _FLOAT_PREC_C*>(estimatedMean + adjIndex));
    _FLOAT_PREC_C variance = *(reinterpret_cast<const _FLOAT_PREC_C*>(estimatedVariance + adjIndex));
    _FLOAT_PREC_C pscale = *(reinterpret_cast<const _FLOAT_PREC_C*>(scale + adjIndex));
    _FLOAT_PREC_C pbias = *(reinterpret_cast<const _FLOAT_PREC_C*>(bias + adjIndex));
    _FLOAT_PREC_C invVariance = rsqrt(fabs(variance + (_FLOAT_PREC_C)(epsilon)));
    _FLOAT_PREC_LS inhat;
    _FLOAT_LS value;
    
    // loop over the batches
    for (unsigned int n = 0; n < batchSize; ++n)
    {
        // load input value
        batchIndex += n * batchStride;
        value = *(reinterpret_cast<const _FLOAT_LS*>(in + batchIndex));
        
        // perform batchnorm operation
        inhat = FLOAT2FLOATPREC_VEC(value);
        inhat = (inhat - mean) * invVariance;
        inhat = fma(pscale, inhat, (_FLOAT_PREC_LS)pbias);
        ACTIVATION_OP(inhat, inhat, _FLOAT_PREC_LS);
        value = FLOATPREC2FLOAT_VEC(inhat);
        
        // write output value
        *(reinterpret_cast<_FLOAT_LS*>(out + batchIndex)) = value;
    }
}
