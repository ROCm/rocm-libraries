// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "Configuration.hpp"
#include "HipKernelActivation.hpp"
#include "VectorTypes.hpp"

// Load the configs to this file
namespace /*anonymous*/
{
using hip_plugin_config = hip_kernel_provider::config;
using hip_plugin_bn_config = hip_kernel_provider::batchnorm::config;

using InputType = hip_plugin_bn_config::input_type;
using OutputType = hip_plugin_bn_config::output_type;
using ScaleBiasType = hip_plugin_bn_config::scale_bias_type;
using MeanVarType = hip_plugin_bn_config::mean_inv_variance_type;
using ComputeType = hip_plugin_bn_config::compute_type;

using InputVecType = hip_plugin_bn_config::input_vec_type;
using OutputVecType = hip_plugin_bn_config::output_vec_type;
using ScaleBiasVecType = hip_plugin_bn_config::scale_bias_vec_type;
using MeanVarVecType = hip_plugin_bn_config::mean_inv_variance_vec_type;
using ComputeVecType = hip_plugin_bn_config::compute_vec_type;
} // namespace

template <unsigned int vecSizeX, unsigned int vecSizeY>
__device__ __forceinline__ void
    bnFwdInferSpatialImpl(unsigned int tidx, // NOLINT(misc-use-internal-linkage)
                          unsigned int tidy,
                          const InputType* in,
                          OutputType* out,
                          const MeanVarType* mean,
                          const MeanVarType* invVariance,
                          const ScaleBiasType* scale,
                          const ScaleBiasType* bias,
                          unsigned int channelStride,
                          unsigned int spatialStride,
                          unsigned int batchStride,
                          float alpha,
                          float beta)
{
    ComputeType inhat[hip_plugin_bn_config::VEC_SIZE]; // NOLINT(modernize-avoid-c-arrays)
    InputType value[hip_plugin_bn_config::VEC_SIZE]; // NOLINT(modernize-avoid-c-arrays)
    OutputType outValue // NOLINT(modernize-avoid-c-arrays)
        [hip_plugin_bn_config::
             VEC_SIZE]; // NOLINT(modernize-avoid-c-arrays) Unused if InputType equals OutputType

    // loop over the batches
    // NOTE: We use zlocalsize = 1 and zgridsize = min(batches, maxGridSizeToFillTheGPU). So the
    // idea here is to use the blocks in z-dimension to cover the batch dimension first, and then
    // each block will loop over the remaining batches with stride of gridDim.z if necessary.
    for(unsigned int batch = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)
        batch
        < hip_plugin_bn_config::BATCHES; // NOLINT(readability-static-accessed-through-instance)
        batch += gridDim.z) // NOLINT(readability-static-accessed-through-instance)
    {
        // load input value
        const unsigned int batchIndex = batch * batchStride + tidx * channelStride * vecSizeX
                                        + tidy * spatialStride * vecSizeY;

        *reinterpret_cast<InputVecType*>(value)
            = *reinterpret_cast<const InputVecType*>(in + batchIndex);

        // perform batchnorm and activation
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::VEC_SIZE; ++i)
        {
            inhat[i] = (hip_kernel_provider::cast<ComputeType>(value[i])
                        - hip_kernel_provider::cast<ComputeType>(mean[i]))
                       * hip_kernel_provider::cast<ComputeType>(invVariance[i]);

            inhat[i] = hip_kernel_provider::cast<ComputeType>(scale[i]) * inhat[i]
                       + hip_kernel_provider::cast<ComputeType>(bias[i]);

            inhat[i]
                = hip_kernel_provider::applyActivation<ComputeType, hip_plugin_config::NEURON_OP>(
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
        auto* outPtr = reinterpret_cast<OutputVecType*>(out + batchIndex);
        if constexpr(std::is_same_v<InputType, OutputType>)
        {
            *outPtr = *reinterpret_cast<const OutputVecType*>(value);
        }
        else
        {
            *outPtr = *reinterpret_cast<const OutputVecType*>(outValue);
        }
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormFwdInferSpatialEst(const InputType* __restrict in,
                                OutputType* __restrict out,
                                const MeanVarType* __restrict estimatedMean,
                                const MeanVarType* __restrict estimatedVariance,
                                const ScaleBiasType* __restrict scale,
                                const ScaleBiasType* __restrict bias,
                                double epsilon,
                                unsigned int channelStride,
                                unsigned int spatialStride,
                                unsigned int batchStride,
                                float alpha,
                                float beta)
{
    const unsigned int tidx
        = blockIdx.x // NOLINT(readability-static-accessed-through-instance)
              * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                    .GRP0 // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int tidy
        = blockIdx.y // NOLINT(readability-static-accessed-through-instance)
              * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                    .GRP1 // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int tidz = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)

    // decide vector sizes based on problem layout
    constexpr unsigned int VEC_SIZE_X
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::VEC_SIZE : 1;
    constexpr unsigned int VEC_SIZE_Y
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS
       || tidy * VEC_SIZE_Y >= hip_plugin_bn_config::SPATIAL_SIZE
       || tidz >= hip_plugin_bn_config::BATCHES)
    {
        return;
    }

    // indices for current thread
    const unsigned int adjIndex = tidx * VEC_SIZE_X;

    // batch parameters and values for current thread
    MeanVarType mean[hip_plugin_bn_config::VEC_SIZE];
    MeanVarType variance[hip_plugin_bn_config::VEC_SIZE];
    ScaleBiasType pscale[hip_plugin_bn_config::VEC_SIZE];
    ScaleBiasType pbias[hip_plugin_bn_config::VEC_SIZE];
    ComputeType invVariance[hip_plugin_bn_config::VEC_SIZE];
    if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST)
    {
        *reinterpret_cast<MeanVarVecType*>(mean)
            = *reinterpret_cast<const MeanVarVecType*>(estimatedMean + adjIndex);
        *reinterpret_cast<MeanVarVecType*>(variance)
            = *reinterpret_cast<const MeanVarVecType*>(estimatedVariance + adjIndex);
        *reinterpret_cast<ScaleBiasVecType*>(pscale)
            = *reinterpret_cast<const ScaleBiasVecType*>(scale + adjIndex);
        *reinterpret_cast<ScaleBiasVecType*>(pbias)
            = *reinterpret_cast<const ScaleBiasVecType*>(bias + adjIndex);
    }
    else // NCL layout
    {
        const auto meanVal = estimatedMean[adjIndex];
        const auto varianceVal = estimatedVariance[adjIndex];
        const auto pscaleVal = scale[adjIndex];
        const auto pbiasVal = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::VEC_SIZE; ++i)
        {
            mean[i] = meanVal;
            variance[i] = varianceVal;
            pscale[i] = pscaleVal;
            pbias[i] = pbiasVal;
        }
    }
#pragma unroll
    for(unsigned int i = 0; i < hip_plugin_bn_config::VEC_SIZE; ++i)
    {
        invVariance[i] = rsqrt( // NOLINT(bugprone-narrowing-conversions)
            fabs(hip_kernel_provider::cast<ComputeType>( // NOLINT(bugprone-narrowing-conversions)
                     variance[i]) // NOLINT(bugprone-narrowing-conversions)
                 + static_cast<ComputeType>(epsilon)));
    }

    bnFwdInferSpatialImpl<VEC_SIZE_X, VEC_SIZE_Y>(tidx,
                                                  tidy,
                                                  in,
                                                  out,
                                                  mean,
                                                  invVariance,
                                                  pscale,
                                                  pbias,
                                                  channelStride,
                                                  spatialStride,
                                                  batchStride,
                                                  alpha,
                                                  beta);
}

// Uses estimated inverse variance rather than inverse variance, which avoids need for an
// epsilon parameter and rsqrt() operations.
extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormFwdInferSpatialEstInvVar(const InputType* __restrict in,
                                      OutputType* __restrict out,
                                      const MeanVarType* __restrict estimatedMean,
                                      const MeanVarType* __restrict estimatedInvVariance,
                                      const ScaleBiasType* __restrict scale,
                                      const ScaleBiasType* __restrict bias,
                                      unsigned int channelStride,
                                      unsigned int spatialStride,
                                      unsigned int batchStride,
                                      float alpha,
                                      float beta)
{
    const unsigned int tidx
        = blockIdx.x // NOLINT(readability-static-accessed-through-instance)
              * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                    .GRP0 // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int tidy
        = blockIdx.y // NOLINT(readability-static-accessed-through-instance)
              * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                    .GRP1 // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int tidz = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)

    // decide vector sizes based on problem layout
    constexpr unsigned int VEC_SIZE_X
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::VEC_SIZE : 1;
    constexpr unsigned int VEC_SIZE_Y
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::VEC_SIZE;

    // skip execution for out-of-bound threads
    if(tidx * VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS
       || tidy * VEC_SIZE_Y >= hip_plugin_bn_config::SPATIAL_SIZE
       || tidz >= hip_plugin_bn_config::BATCHES)
    {
        return;
    }

    // indices for current thread
    const unsigned int adjIndex = tidx * VEC_SIZE_X;

    // batch parameters and values for current thread
    MeanVarType mean[hip_plugin_bn_config::VEC_SIZE];
    ScaleBiasType pscale[hip_plugin_bn_config::VEC_SIZE];
    ScaleBiasType pbias[hip_plugin_bn_config::VEC_SIZE];
    MeanVarType invVariance[hip_plugin_bn_config::VEC_SIZE];
    if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST)
    {
        *reinterpret_cast<MeanVarVecType*>(mean)
            = *reinterpret_cast<const MeanVarVecType*>(estimatedMean + adjIndex);
        *reinterpret_cast<MeanVarVecType*>(invVariance)
            = *reinterpret_cast<const MeanVarVecType*>(estimatedInvVariance + adjIndex);
        *reinterpret_cast<ScaleBiasVecType*>(pscale)
            = *reinterpret_cast<const ScaleBiasVecType*>(scale + adjIndex);
        *reinterpret_cast<ScaleBiasVecType*>(pbias)
            = *reinterpret_cast<const ScaleBiasVecType*>(bias + adjIndex);
    }
    else // NCL layout
    {
        const auto meanVal = estimatedMean[adjIndex];
        const auto invVarianceVal = estimatedInvVariance[adjIndex];
        const auto pscaleVal = scale[adjIndex];
        const auto pbiasVal = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::VEC_SIZE; ++i)
        {
            mean[i] = meanVal;
            invVariance[i] = invVarianceVal;
            pscale[i] = pscaleVal;
            pbias[i] = pbiasVal;
        }
    }

    bnFwdInferSpatialImpl<VEC_SIZE_X, VEC_SIZE_Y>(tidx,
                                                  tidy,
                                                  in,
                                                  out,
                                                  mean,
                                                  invVariance,
                                                  pscale,
                                                  pbias,
                                                  channelStride,
                                                  spatialStride,
                                                  batchStride,
                                                  alpha,
                                                  beta);
}
