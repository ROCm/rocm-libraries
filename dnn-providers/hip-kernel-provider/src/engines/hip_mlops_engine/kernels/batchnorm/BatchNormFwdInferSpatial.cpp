// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"

using InputType = HIP_PLUGIN_BN_INPUT_TYPE;
using OutputType = HIP_PLUGIN_BN_OUTPUT_TYPE;
using ScaleType = HIP_PLUGIN_BN_SCALE_TYPE;
using MeanVarType = HIP_PLUGIN_BN_MEAN_VAR_TYPE;
using ComputeType = float;

// determine block size using parameters passed from the host
constexpr int blockSize = HIP_PLUGIN_BN_GRP0 * HIP_PLUGIN_BN_GRP1 * HIP_PLUGIN_BN_GRP2;

// define types for vectorized loads/stores
using InputVecType =
    typename hip_kernel_provider::mapped_vector_type<InputType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using OutputVecType =
    typename hip_kernel_provider::mapped_vector_type<OutputType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using MeanVarVecType =
    typename hip_kernel_provider::mapped_vector_type<MeanVarType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using ScaleVecType =
    typename hip_kernel_provider::mapped_vector_type<ScaleType, HIP_PLUGIN_BN_VEC_SIZE>::type;
using ComputeVecType =
    typename hip_kernel_provider::mapped_vector_type<ComputeType, HIP_PLUGIN_BN_VEC_SIZE>::type;

template <unsigned int vecSizeX, unsigned int vecSizeY>
__device__ __forceinline__ void BNFwdInferSpatialImpl(unsigned int tidx,
                                                      unsigned int tidy,
                                                      const InputType* in,
                                                      OutputType* out,
                                                      const MeanVarType* mean,
                                                      const MeanVarType* invVariance,
                                                      const ScaleType* scale,
                                                      const ScaleType* bias,
                                                      unsigned int batchSize,
                                                      unsigned int cStride,
                                                      unsigned int hwStride,
                                                      unsigned int batchStride,
                                                      float alpha,
                                                      float beta)
{
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
            inhat[i] = (hip_kernel_provider::cast<ComputeType>(value[i])
                        - hip_kernel_provider::cast<ComputeType>(mean[i]))
                       * hip_kernel_provider::cast<ComputeType>(invVariance[i]);

            inhat[i] = hip_kernel_provider::cast<ComputeType>(scale[i]) * inhat[i]
                       + hip_kernel_provider::cast<ComputeType>(bias[i]);

            inhat[i] = hip_kernel_provider::applyActivation<
                ComputeType,
                static_cast<hip_kernel_provider::ActivationMode>(HIP_PLUGIN_BN_NRN_OP_ID)>(
                inhat[i], alpha, beta);
            if constexpr(std::is_same_v<InputType, OutputType>)
            {
                value[i] = hip_kernel_provider::cast<OutputType>(inhat[i]);
            }
            else
            {
                outValue[i] = hip_kernel_provider::cast<OutputType>(inhat[i]);
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
                                const MeanVarType* __restrict estimatedMean,
                                const MeanVarType* __restrict estimatedVariance,
                                const ScaleType* __restrict scale,
                                const ScaleType* __restrict bias,
                                double epsilon,
                                unsigned int c,
                                unsigned int hw,
                                unsigned int batchSize,
                                unsigned int cStride,
                                unsigned int hwStride,
                                unsigned int batchStride,
                                float alpha,
                                float beta)
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
    MeanVarType mean[HIP_PLUGIN_BN_VEC_SIZE];
    MeanVarType variance[HIP_PLUGIN_BN_VEC_SIZE];
    ScaleType pscale[HIP_PLUGIN_BN_VEC_SIZE];
    ScaleType pbias[HIP_PLUGIN_BN_VEC_SIZE];
    ComputeType invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<MeanVarVecType*>(mean))
            = *(reinterpret_cast<const MeanVarVecType*>(estimatedMean + adjIndex));
        *(reinterpret_cast<MeanVarVecType*>(variance))
            = *(reinterpret_cast<const MeanVarVecType*>(estimatedVariance + adjIndex));
        *(reinterpret_cast<ScaleVecType*>(pscale))
            = *(reinterpret_cast<const ScaleVecType*>(scale + adjIndex));
        *(reinterpret_cast<ScaleVecType*>(pbias))
            = *(reinterpret_cast<const ScaleVecType*>(bias + adjIndex));
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
        invVariance[i] = rsqrt(fabs(hip_kernel_provider::cast<ComputeType>(variance[i])
                                    + static_cast<ComputeType>(epsilon)));
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
                                      const MeanVarType* __restrict estimatedMean,
                                      const MeanVarType* __restrict estimatedInvVariance,
                                      const ScaleType* __restrict scale,
                                      const ScaleType* __restrict bias,
                                      unsigned int c,
                                      unsigned int hw,
                                      unsigned int batchSize,
                                      unsigned int cStride,
                                      unsigned int hwStride,
                                      unsigned int batchStride,
                                      float alpha,
                                      float beta)
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
    MeanVarType mean[HIP_PLUGIN_BN_VEC_SIZE];
    ScaleType pscale[HIP_PLUGIN_BN_VEC_SIZE];
    ScaleType pbias[HIP_PLUGIN_BN_VEC_SIZE];
    MeanVarType invVariance[HIP_PLUGIN_BN_VEC_SIZE];
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        *(reinterpret_cast<MeanVarVecType*>(mean))
            = *(reinterpret_cast<const MeanVarVecType*>(estimatedMean + adjIndex));
        *(reinterpret_cast<MeanVarVecType*>(invVariance))
            = *(reinterpret_cast<const MeanVarVecType*>(estimatedInvVariance + adjIndex));
        *(reinterpret_cast<ScaleVecType*>(pscale))
            = *(reinterpret_cast<const ScaleVecType*>(scale + adjIndex));
        *(reinterpret_cast<ScaleVecType*>(pbias))
            = *(reinterpret_cast<const ScaleVecType*>(bias + adjIndex));
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
