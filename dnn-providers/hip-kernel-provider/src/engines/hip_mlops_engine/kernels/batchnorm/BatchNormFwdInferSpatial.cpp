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
__device__ __forceinline__ void BNFwdInferSpatialImpl(unsigned int tidx,
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
    ComputeType inhat[hip_plugin_bn_config::vec_size];
    InputType value[hip_plugin_bn_config::vec_size];
    OutputType outValue[hip_plugin_bn_config::vec_size]; // Unused if InputType equals OutputType

    // loop over the batches
    // NOTE: We use zlocalsize = 1 and zgridsize = min(batches, maxGridSizeToFillTheGPU). So the
    // idea here is to use the blocks in z-dimension to cover the batch dimension first, and then
    // each block will loop over the remaining batches with stride of gridDim.z if necessary.
    for(unsigned int batch = blockIdx.z; batch < hip_plugin_bn_config::batches; batch += gridDim.z)
    {
        // load input value
        const unsigned int batchIndex = batch * batchStride + tidx * channelStride * vecSizeX
                                        + tidy * spatialStride * vecSizeY;

        *reinterpret_cast<InputVecType*>(value)
            = *reinterpret_cast<const InputVecType*>(in + batchIndex);

        // perform batchnorm and activation
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::vec_size; ++i)
        {
            inhat[i] = (hip_kernel_provider::cast<ComputeType>(value[i])
                        - hip_kernel_provider::cast<ComputeType>(mean[i]))
                       * hip_kernel_provider::cast<ComputeType>(invVariance[i]);

            inhat[i] = hip_kernel_provider::cast<ComputeType>(scale[i]) * inhat[i]
                       + hip_kernel_provider::cast<ComputeType>(bias[i]);

            inhat[i]
                = hip_kernel_provider::applyActivation<ComputeType, hip_plugin_config::neuron_op>(
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
            *outPtr = *reinterpret_cast<const OutputVecType*>(value);
        }
        else
        {
            *outPtr = *reinterpret_cast<const OutputVecType*>(outValue);
        }
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormFwdInferSpatialEst(const InputType* __restrict in,
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
    unsigned int tidx = blockIdx.x * hip_plugin_bn_config::launch_dim.grp0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * hip_plugin_bn_config::launch_dim.grp1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::vec_size : 1;
    constexpr unsigned int vecSizeY
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::vec_size;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= hip_plugin_bn_config::channels
       || tidy * vecSizeY >= hip_plugin_bn_config::spatial_size
       || tidz >= hip_plugin_bn_config::batches)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = tidx * vecSizeX;

    // batch parameters and values for current thread
    MeanVarType mean[hip_plugin_bn_config::vec_size];
    MeanVarType variance[hip_plugin_bn_config::vec_size];
    ScaleBiasType pscale[hip_plugin_bn_config::vec_size];
    ScaleBiasType pbias[hip_plugin_bn_config::vec_size];
    ComputeType invVariance[hip_plugin_bn_config::vec_size];
    if constexpr(hip_plugin_config::layout_channel_last)
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
        const auto mean_val = estimatedMean[adjIndex];
        const auto variance_val = estimatedVariance[adjIndex];
        const auto pscale_val = scale[adjIndex];
        const auto pbias_val = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::vec_size; ++i)
        {
            mean[i] = mean_val;
            variance[i] = variance_val;
            pscale[i] = pscale_val;
            pbias[i] = pbias_val;
        }
    }
#pragma unroll
    for(unsigned int i = 0; i < hip_plugin_bn_config::vec_size; ++i)
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
                                              channelStride,
                                              spatialStride,
                                              batchStride,
                                              alpha,
                                              beta);
}

// Uses estimated inverse variance rather than inverse variance, which avoids need for an
// epsilon parameter and rsqrt() operations.
extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormFwdInferSpatialEstInvVar(const InputType* __restrict in,
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
    unsigned int tidx = blockIdx.x * hip_plugin_bn_config::launch_dim.grp0 + threadIdx.x;
    unsigned int tidy = blockIdx.y * hip_plugin_bn_config::launch_dim.grp1 + threadIdx.y;
    unsigned int tidz = blockIdx.z;

    // decide vector sizes based on problem layout
    constexpr unsigned int vecSizeX
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::vec_size : 1;
    constexpr unsigned int vecSizeY
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::vec_size;

    // skip execution for out-of-bound threads
    if(tidx * vecSizeX >= hip_plugin_bn_config::channels
       || tidy * vecSizeY >= hip_plugin_bn_config::spatial_size
       || tidz >= hip_plugin_bn_config::batches)
    {
        return;
    }

    // indices for current thread
    unsigned int adjIndex = tidx * vecSizeX;

    // batch parameters and values for current thread
    MeanVarType mean[hip_plugin_bn_config::vec_size];
    ScaleBiasType pscale[hip_plugin_bn_config::vec_size];
    ScaleBiasType pbias[hip_plugin_bn_config::vec_size];
    MeanVarType invVariance[hip_plugin_bn_config::vec_size];
    if constexpr(hip_plugin_config::layout_channel_last)
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
        const auto mean_val = estimatedMean[adjIndex];
        const auto invVariance_val = estimatedInvVariance[adjIndex];
        const auto pscale_val = scale[adjIndex];
        const auto pbias_val = bias[adjIndex];
#pragma unroll
        for(unsigned int i = 0; i < hip_plugin_bn_config::vec_size; ++i)
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
                                              channelStride,
                                              spatialStride,
                                              batchStride,
                                              alpha,
                                              beta);
}
