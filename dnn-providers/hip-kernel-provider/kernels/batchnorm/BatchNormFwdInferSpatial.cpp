// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "FloatTypes.h"
#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"

// determine block size using parameters passed from the host
constexpr int blockSize = HIP_PLUGIN_BN_GRP0 * HIP_PLUGIN_BN_GRP1 * HIP_PLUGIN_BN_GRP2;

// define types for vectorized loads/stores
using FLOAT_VEC_TYPE =
    typename hip_kernel_provider::mapped_vector_type<FLOAT, HIP_PLUGIN_BN_VEC_SIZE>::type;
using FLOAT_ACCUM_VEC_TYPE =
    typename hip_kernel_provider::mapped_vector_type<FLOAT_ACCUM, HIP_PLUGIN_BN_VEC_SIZE>::type;

template <unsigned int vecSizeX, unsigned int vecSizeY>
__device__ __forceinline__ void BNFwdInferSpatialImpl(unsigned int tidx,
                                                      unsigned int tidy,
                                                      const FLOAT* in,
                                                      FLOAT* out,
                                                      const FLOAT_ACCUM* mean,
                                                      const FLOAT_ACCUM* invVariance,
                                                      const FLOAT_ACCUM* scale,
                                                      const FLOAT_ACCUM* bias,
                                                      unsigned int batchSize,
                                                      unsigned int cStride,
                                                      unsigned int hwStride,
                                                      unsigned int batchStride,
                                                      FLOAT_ACCUM alpha,
                                                      FLOAT_ACCUM beta)
{
    FLOAT_ACCUM inhat[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT value[HIP_PLUGIN_BN_VEC_SIZE];

    // loop over the batches
    // NOTE: We use zlocalsize = 1 and zgridsize = min(batchSize, maxGridSizeToFillTheGPU). So the
    // idea here is to use the blocks in z-dimension to cover the batch dimension first, and then
    // each block will loop over the remaining batches with stride of gridDim.z if necessary.
    for(unsigned int n = blockIdx.z; n < batchSize; n += gridDim.z)
    {
        // load input value
        const unsigned int batchIndex
            = (n * batchStride) + (tidx * cStride * vecSizeX) + (tidy * hwStride * vecSizeY);
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(value))
            = *(reinterpret_cast<const FLOAT_VEC_TYPE*>(in + batchIndex));

        // perform batchnorm and activation
#pragma unroll
        for(unsigned int i = 0; i < HIP_PLUGIN_BN_VEC_SIZE; ++i)
        {
            inhat[i] = (CVT_FLOAT2ACCUM(value[i]) - mean[i]) * invVariance[i];
            inhat[i] = scale[i] * inhat[i] + bias[i];
            inhat[i] = hip_kernel_provider::applyActivation<
                FLOAT_ACCUM,
                static_cast<hip_kernel_provider::ActivationMode>(HIP_PLUGIN_NRN_OP_ID)>(
                inhat[i], alpha, beta);

            value[i] = CVT_ACCUM2FLOAT(inhat[i]);
        }

        // write output value
        *(reinterpret_cast<FLOAT_VEC_TYPE*>(out + batchIndex))
            = *(reinterpret_cast<const FLOAT_VEC_TYPE*>(value));
    }
}

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormFwdInferSpatialEst(const FLOAT* __restrict in,
                                FLOAT* __restrict out,
                                const FLOAT_ACCUM* __restrict estimatedMean,
                                const FLOAT_ACCUM* __restrict estimatedVariance,
                                const FLOAT_ACCUM* __restrict scale,
                                const FLOAT_ACCUM* __restrict bias,
                                double epsilon,
                                unsigned int c,
                                unsigned int hw,
                                unsigned int batchSize,
                                unsigned int cStride,
                                unsigned int hwStride,
                                unsigned int batchStride,
                                FLOAT_ACCUM alpha,
                                FLOAT_ACCUM beta)
{
    unsigned int tidx = blockIdx.x * HIP_PLUGIN_BN_GRP0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * HIP_PLUGIN_BN_GRP1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX = HIP_PLUGIN_LAYOUT_NHWC ? HIP_PLUGIN_BN_VEC_SIZE : 1;
    constexpr unsigned int vecSizeY = HIP_PLUGIN_LAYOUT_NHWC ? 1 : HIP_PLUGIN_BN_VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= c || tidy * vecSizeY >= hw || tidz >= batchSize)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = tidx * vecSizeX;

    // batch parameters and values for current thread
    FLOAT_ACCUM mean[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM variance[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM pscale[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM pbias[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(mean))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(variance))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedVariance + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + adjIndex));
    }
    else // NCHW layout
    {
        const auto mean_val = estimatedMean[adjIndex];
        const auto variance_val = estimatedVariance[adjIndex];
        const auto pscale_val = scale[adjIndex];
        const auto pbias_val = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < HIP_PLUGIN_BN_VEC_SIZE; ++i)
        {
            mean[i] = mean_val;
            variance[i] = variance_val;
            pscale[i] = pscale_val;
            pbias[i] = pbias_val;
        }
    }
#pragma unroll
    for(unsigned int i = 0; i < HIP_PLUGIN_BN_VEC_SIZE; ++i)
    {
        invVariance[i] = rsqrt(fabs(variance[i] + static_cast<FLOAT_ACCUM>(epsilon)));
    }

    BNFwdInferSpatialImpl<vecSizeX, vecSizeY>(tidx,
                                              tidy,
                                              in,
                                              out,
                                              mean,
                                              invVariance,
                                              pscale,
                                              pbias,
                                              batchSize,
                                              cStride,
                                              hwStride,
                                              batchStride,
                                              alpha,
                                              beta);
}

// Uses estimated inverse variance rather than inverse variance, which avoids need for an
// epsilon parameter and rsqrt() operations.
extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormFwdInferSpatialEstInvVar(const FLOAT* __restrict in,
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
                                      unsigned int batchStride,
                                      FLOAT_ACCUM alpha,
                                      FLOAT_ACCUM beta)
{
    unsigned int tidx = blockIdx.x * HIP_PLUGIN_BN_GRP0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * HIP_PLUGIN_BN_GRP1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX = HIP_PLUGIN_LAYOUT_NHWC ? HIP_PLUGIN_BN_VEC_SIZE : 1;
    constexpr unsigned int vecSizeY = HIP_PLUGIN_LAYOUT_NHWC ? 1 : HIP_PLUGIN_BN_VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= c || tidy * vecSizeY >= hw || tidz >= batchSize)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = tidx * vecSizeX;

    // batch parameters and values for current thread
    FLOAT_ACCUM mean[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM pscale[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM pbias[HIP_PLUGIN_BN_VEC_SIZE];
    FLOAT_ACCUM invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(mean))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedMean + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(invVariance))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(estimatedInvVariance + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pscale))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(scale + adjIndex));
        *(reinterpret_cast<FLOAT_ACCUM_VEC_TYPE*>(pbias))
            = *(reinterpret_cast<const FLOAT_ACCUM_VEC_TYPE*>(bias + adjIndex));
    }
    else // NCHW layout
    {
        const auto mean_val = estimatedMean[adjIndex];
        const auto invVariance_val = estimatedInvVariance[adjIndex];
        const auto pscale_val = scale[adjIndex];
        const auto pbias_val = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < HIP_PLUGIN_BN_VEC_SIZE; ++i)
        {
            mean[i] = mean_val;
            invVariance[i] = invVariance_val;
            pscale[i] = pscale_val;
            pbias[i] = pbias_val;
        }
    }

    BNFwdInferSpatialImpl<vecSizeX, vecSizeY>(tidx,
                                              tidy,
                                              in,
                                              out,
                                              mean,
                                              invVariance,
                                              pscale,
                                              pbias,
                                              batchSize,
                                              cStride,
                                              hwStride,
                                              batchStride,
                                              alpha,
                                              beta);
}
