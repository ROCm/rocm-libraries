// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelActivation.hpp"
#include "HipKernelCast.hpp"
#include "VectorTypes.hpp"
#include <type_traits>

using InputType = HIP_PLUGIN_BN_INPUT_TYPE;
using OutputType = HIP_PLUGIN_BN_OUTPUT_TYPE;
using ComputeType = HIP_PLUGIN_BN_COMPUTE_TYPE;

// determine block size using parameters passed from the host
constexpr int blockSize = HIP_PLUGIN_BN_GRP0 * HIP_PLUGIN_BN_GRP1 * HIP_PLUGIN_BN_GRP2;

// define types for vectorized loads/stores
using InputVecType =
    typename hip_kernel_provider::mapped_vector_type<InputType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using OutputVecType =
    typename hip_kernel_provider::mapped_vector_type<OutputType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using ComputeVecType =
    typename hip_kernel_provider::mapped_vector_type<ComputeType, HIP_PLUGIN_BN_VEC_SIZE>::type;

template <unsigned int vecSizeX, unsigned int vecSizeY>
__device__ __forceinline__ void BNFwdInferSpatialImpl(unsigned int tidx,
                                                      unsigned int tidy,
                                                      const InputType* in,
                                                      OutputType* out,
                                                      const ComputeType* mean,
                                                      const ComputeType* invVariance,
                                                      const ComputeType* scale,
                                                      const ComputeType* bias,
                                                      unsigned int batchSize,
                                                      unsigned int cStride,
                                                      unsigned int hwStride,
                                                      unsigned int batchStride,
                                                      ComputeType alpha,
                                                      ComputeType beta)
{
    // ComputeType must be float to prevent precision loss
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the BN fwd kernel");
    ComputeType inhat[HIP_PLUGIN_BN_VEC_SIZE];
    InputType value[HIP_PLUGIN_BN_VEC_SIZE];
    OutputType outValue[HIP_PLUGIN_BN_VEC_SIZE]; // Unused if InputType equals OutputType

    // loop over the batches
    // NOTE: We use zlocalsize = 1 and zgridsize = min(batchSize, maxGridSizeToFillTheGPU). So the
    // idea here is to use the blocks in z-dimension to cover the batch dimension first, and then
    // each block will loop over the remaining batches with stride of gridDim.z if necessary.
    for(unsigned int n = blockIdx.z; n < batchSize; n += gridDim.z)
    {
        // load input value
        const unsigned int batchIndex
            = (n * batchStride) + (tidx * cStride * vecSizeX) + (tidy * hwStride * vecSizeY);

        *(reinterpret_cast<InputVecType*>(value))
            = *(reinterpret_cast<const InputVecType*>(in + batchIndex));

        // perform batchnorm and activation
#pragma unroll
        for(unsigned int i = 0; i < HIP_PLUGIN_BN_VEC_SIZE; ++i)
        {
            inhat[i] = (hip_kernel_provider::to_float32(value[i]) - mean[i]) * invVariance[i];
            inhat[i] = scale[i] * inhat[i] + bias[i];
            inhat[i] = hip_kernel_provider::applyActivation<
                ComputeType,
                static_cast<hip_kernel_provider::ActivationMode>(HIP_PLUGIN_BN_NRN_OP_ID)>(
                inhat[i], alpha, beta);
            if constexpr(std::is_same_v<InputType, OutputType>)
            {
                value[i] = hip_kernel_provider::from_float32<OutputType>(inhat[i]);
            }
            else
            {
                outValue[i] = hip_kernel_provider::from_float32<OutputType>(inhat[i]);
            }
        }

        // write output value
        OutputVecType* outPtr = reinterpret_cast<OutputVecType*>(out + batchIndex);
        if constexpr(std::is_same_v<InputType, OutputType>)
        {
            *outPtr = *(reinterpret_cast<const OutputVecType*>(value));
        }
        else
        {
            *outPtr = *(reinterpret_cast<const OutputVecType*>(outValue));
        }
    }
}

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormFwdInferSpatialEst(const InputType* __restrict in,
                                OutputType* __restrict out,
                                const ComputeType* __restrict estimatedMean,
                                const ComputeType* __restrict estimatedVariance,
                                const ComputeType* __restrict scale,
                                const ComputeType* __restrict bias,
                                double epsilon,
                                unsigned int c,
                                unsigned int hw,
                                unsigned int batchSize,
                                unsigned int cStride,
                                unsigned int hwStride,
                                unsigned int batchStride,
                                ComputeType alpha,
                                ComputeType beta)
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
    ComputeType mean[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType variance[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType pscale[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType pbias[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<ComputeVecType*>(mean))
            = *(reinterpret_cast<const ComputeVecType*>(estimatedMean + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(variance))
            = *(reinterpret_cast<const ComputeVecType*>(estimatedVariance + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(pscale))
            = *(reinterpret_cast<const ComputeVecType*>(scale + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(pbias))
            = *(reinterpret_cast<const ComputeVecType*>(bias + adjIndex));
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
        invVariance[i] = rsqrt(fabs(variance[i] + static_cast<ComputeType>(epsilon)));
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
    BatchNormFwdInferSpatialEstInvVar(const InputType* __restrict in,
                                      OutputType* __restrict out,
                                      const ComputeType* __restrict estimatedMean,
                                      const ComputeType* __restrict estimatedInvVariance,
                                      const ComputeType* __restrict scale,
                                      const ComputeType* __restrict bias,
                                      unsigned int c,
                                      unsigned int hw,
                                      unsigned int batchSize,
                                      unsigned int cStride,
                                      unsigned int hwStride,
                                      unsigned int batchStride,
                                      ComputeType alpha,
                                      ComputeType beta)
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
    ComputeType mean[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType pscale[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType pbias[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<ComputeVecType*>(mean))
            = *(reinterpret_cast<const ComputeVecType*>(estimatedMean + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(invVariance))
            = *(reinterpret_cast<const ComputeVecType*>(estimatedInvVariance + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(pscale))
            = *(reinterpret_cast<const ComputeVecType*>(scale + adjIndex));
        *(reinterpret_cast<ComputeVecType*>(pbias))
            = *(reinterpret_cast<const ComputeVecType*>(bias + adjIndex));
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
