// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// NOTE: These included headers should be standalone, they shouldn't rely on each other,
// otherwise the dependencies will be pretty messed up.
// TODO: actually these headers are not that independent due to the Macros that being used
// currently. we should remove as more macros as we can.
#include "BatchnormFunctions.hpp"
#include "Configuration.hpp"
#include "HipKernelActivation.hpp"
#include "HipKernelMath.hpp"
#include "ReductionFunctions.hpp"
#include "StaticUnroll.hpp"
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

using InputChannelVecType = hip_plugin_bn_config::input_channel_vec_type;
using OutputChannelVecType = hip_plugin_bn_config::output_channel_vec_type;
using ScaleBiasChannelVecType = hip_plugin_bn_config::scale_bias_channel_vec_type;
using MeanVarChannelVecType = hip_plugin_bn_config::mean_inv_variance_channel_vec_type;
using ComputeChannelVecType = hip_plugin_bn_config::compute_channel_vec_type;
} // namespace

namespace hip_kernel_provider::batchnorm
{

template <int BnVariant>
struct BatchNormFwdTrainSpatialImpl
{
    static_assert(false, "this variant is not supported.");
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 0
template <>
struct BatchNormFwdTrainSpatialImpl<0>
{
    // These are the configs for this variant
    static constexpr unsigned int segtmp_1
        = hip_plugin_bn_config::launch_dim.grp0 / hip_plugin_bn_config::spatial_size;
    static constexpr unsigned int segtmp_2 = (segtmp_1 == 0) ? 1 : segtmp_1;
    static constexpr unsigned int segtmp = hip_plugin_bn_config::spatial_size * segtmp_2;
    static constexpr unsigned int segment = (segtmp > hip_plugin_bn_config::channel_slice_size)
                                                ? hip_plugin_bn_config::channel_slice_size
                                                : segtmp;
    static constexpr unsigned int batchLoop
        = (hip_plugin_bn_config::channel_slice_size + segment - 1) / segment;
    static constexpr unsigned int segiSpatial = segment / hip_plugin_bn_config::spatial_size;
    static constexpr unsigned int batchLoopm = batchLoop - 1;
    static constexpr unsigned int snSpatial = batchLoopm * segiSpatial;

    __forceinline__ __device__ void operator()(const InputType* __restrict in,
                                               OutputType* __restrict out,
                                               const ScaleBiasType* __restrict scale,
                                               const ScaleBiasType* __restrict bias,
                                               double epsilon,
                                               MeanVarType& mean,
                                               MeanVarType& variance,
                                               MeanVarType& invVariance,
                                               ComputeType _alpha,
                                               ComputeType _beta)
    {
        // SPATIAL
        mean = hip_kernel_provider::cast<MeanVarType>(0.0);
        variance = hip_kernel_provider::cast<MeanVarType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarType>(0.0);
        InputType batchvalues[batchLoop];
        ComputeType temp;

        __shared__ ComputeType lcl_scale;
        __shared__ ComputeType lcl_bias;

        unsigned int index = 0;
        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int channelSpatialId = grpid * hip_plugin_bn_config::spatial_size
                                        + (lid % hip_plugin_bn_config::spatial_size);
        unsigned int lidiSpatial = lid / hip_plugin_bn_config::spatial_size;
        unsigned int batchId = 0;

        if(lid == 0)
        {
            lcl_scale = scale[grpid];
            lcl_bias = bias[grpid];
        }

        __syncthreads();

        if(lid < segment)
        {
            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int unrollHint
                = hip_plugin_config::input_type_strategy == type_strategy::fpmix ? 2 : 1;
            static_unroll_count<unsigned int, 0, batchLoopm, 1, unrollHint>{
                [&](unsigned int batch) {
                    batchId = batch * segiSpatial + lidiSpatial;
                    index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
                    batchvalues[batch] = *(in + index);
                    temp = hip_kernel_provider::cast<ComputeType>(in[index]);
                    mean += temp;
                    variance = fma(temp, temp, variance);
                }};
            batchId = snSpatial + lidiSpatial;
            index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
            batchvalues[batchLoopm] = (index < hip_plugin_bn_config::tensor_size)
                                          ? in[index]
                                          : hip_kernel_provider::cast<InputType>(0.0);
            temp = hip_kernel_provider::cast<ComputeType>(batchvalues[batchLoopm]);
            mean += temp;
            variance = fma(temp, temp, variance);
        }
        __syncthreads();

        constexpr auto lcl_data_size = hip_plugin_bn_config::use_amdgcn
                                           ? hip_plugin_bn_config::lds_gcn_size
                                           : hip_plugin_bn_config::lds_size;
        __shared__ ComputeType lcl_data_x[lcl_data_size];
        __shared__ ComputeType lcl_data_y[lcl_data_size];
        if constexpr(hip_plugin_bn_config::use_amdgcn)
        {
            hip_kernel_provider::batchnorm::reduction::gcn_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::lds_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }

        // Reduction complete

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = hip_kernel_provider::rsqrt(variance
                                                 + hip_kernel_provider::cast<ComputeType>(epsilon));

        ComputeType pvscale = hip_kernel_provider::cast<ComputeType>(lcl_scale);
        ComputeType pvbias = hip_kernel_provider::cast<ComputeType>(lcl_bias);

        if(lid < segment)
        {
            // Calculate norm

            auto inhat = hip_kernel_provider::cast<ComputeType>(0.0);

            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int unrollHint
                = hip_plugin_config::input_type_strategy == type_strategy::fpmix ? 2 : 1;

            static_unroll_count<unsigned int, 0, batchLoopm, 1, unrollHint>{
                [&](unsigned int batch) {
                    // Apply normalization
                    inhat = (hip_kernel_provider::cast<ComputeType>(batchvalues[batch]) - mean)
                            * invVariance;
                    batchId = batch * segiSpatial + lidiSpatial;
                    index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
                    auto value
                        = hip_kernel_provider::cast<ComputeType>(fma(pvscale, inhat, pvbias));
                    value = hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::neuron_op>(
                        value, _alpha, _beta);
                    out[index] = hip_kernel_provider::cast<OutputType>(value);
                }};

            // Tail of loop
            inhat = (hip_kernel_provider::cast<ComputeType>(batchvalues[batchLoopm]) - mean)
                    * invVariance;
            batchId = snSpatial + lidiSpatial;
            index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
            if(index < hip_plugin_bn_config::tensor_size)
            {
                auto value = hip_kernel_provider::cast<ComputeType>(fma(pvscale, inhat, pvbias));
                value = hip_kernel_provider::applyActivation<ComputeType,
                                                             hip_plugin_config::neuron_op>(
                    value, _alpha, _beta);
                out[index] = hip_kernel_provider::cast<OutputType>(value);
            }
        }
    }
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 1
template <>
struct BatchNormFwdTrainSpatialImpl<1>
{
    // These are the configs for this variant
    static constexpr unsigned int max_read
        = hip_plugin_config::layout_channel_last
              ? 1
              : (hip_plugin_bn_config::spatial_size >= 4096 ? 3 : 2);
    static constexpr unsigned int read_block = 1;
    static constexpr unsigned int grpread
        = hip_plugin_config::layout_channel_last
              ? (hip_plugin_bn_config::launch_dim.grp0 * read_block)
              : (hip_plugin_bn_config::launch_dim.grp0 * read_block * 4);
    static constexpr unsigned int rem4
        = hip_plugin_bn_config::channel_slice_size
          - ((hip_plugin_bn_config::channel_slice_size / grpread) * grpread);
    static constexpr unsigned int less4 = hip_plugin_bn_config::channel_slice_size - rem4;
    static constexpr unsigned int chunk4 = max_read * grpread;
    static constexpr unsigned int remout4
        = hip_plugin_bn_config::channel_slice_size
          - ((hip_plugin_bn_config::channel_slice_size / chunk4) * chunk4);
    static constexpr unsigned int lessout4 = hip_plugin_bn_config::channel_slice_size - remout4;
    static constexpr unsigned int rem
        = hip_plugin_bn_config::channel_slice_size
          - ((hip_plugin_bn_config::channel_slice_size / hip_plugin_bn_config::launch_dim.grp0)
             * hip_plugin_bn_config::launch_dim.grp0);
    static constexpr unsigned int less = hip_plugin_bn_config::channel_slice_size - rem;
    static constexpr unsigned int chunk = max_read * hip_plugin_bn_config::launch_dim.grp0;
    static constexpr unsigned int remout
        = hip_plugin_bn_config::channel_slice_size
          - ((hip_plugin_bn_config::channel_slice_size / chunk) * chunk);
    static constexpr unsigned int lessout = hip_plugin_bn_config::channel_slice_size - remout;

    // Kernel
    __forceinline__ __device__ void operator()(const InputType* __restrict in,
                                               OutputType* __restrict out,
                                               const ScaleBiasType* __restrict scale,
                                               const ScaleBiasType* __restrict bias,
                                               double epsilon,
                                               MeanVarType& mean,
                                               MeanVarType& variance,
                                               MeanVarType& invVariance,
                                               ComputeType alpha,
                                               ComputeType beta)
    {
        ComputeType pvscale, pvbias;

        mean = hip_kernel_provider::cast<MeanVarType>(0.0);
        variance = hip_kernel_provider::cast<MeanVarType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarType>(0.0);

        __shared__ ComputeType lcl_scale;
        __shared__ ComputeType lcl_bias;

        unsigned int index = 0;
        const unsigned int lid = threadIdx.x;
        const unsigned int grpid = blockIdx.x;

        // Note: this variable is only used when hip_plugin_config::layout_channel_last is false.
        unsigned int channelSpatialId;
        if constexpr(!hip_plugin_config::layout_channel_last)
        {
            channelSpatialId = grpid * hip_plugin_bn_config::spatial_size;
        }

        unsigned int batchIdx = 0;
        unsigned int lidx = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!hip_plugin_config::layout_channel_last
                     && hip_plugin_bn_config::spatial_size >= 4096)
        {
            using InputVecType4 = typename mapped_vector_type<InputType, 4>::type;
            InputVecType4 read4;

            static_unroll_count<unsigned int, 0, less4, grpread, 2>{[&](unsigned int k) {
                if((k + (lid << 2)) < less4)
                {
                    batchIdx = (k + (lid << 2)) / hip_plugin_bn_config::spatial_size;
                    lidx = (k + (lid << 2)) - (batchIdx * hip_plugin_bn_config::spatial_size);
                    index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;
                    read4 = *reinterpret_cast<const InputVecType4*>(in + index);
                    hip_kernel_provider::batchnorm::_accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::_accumulate_mad(variance, read4, read4);
                }
            }};

            if constexpr(rem4 > 0u)
            {
                const unsigned int remkey = (lid << 2) + less4;
                batchIdx = remkey / hip_plugin_bn_config::spatial_size;
                lidx = remkey - (batchIdx * hip_plugin_bn_config::spatial_size);
                index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;

                // index is unsigned int, so if the result would normally end up negative,
                // the value wraps around and the check fails. Improves on the
                // previous way of handling which was: if(index < (hip_plugin_bn_config::tensor_size - 3))
                if(index + 3 < (hip_plugin_bn_config::tensor_size))
                {
                    read4 = *reinterpret_cast<const InputVecType4*>(in + index);
                    hip_kernel_provider::batchnorm::_accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::_accumulate_mad(variance, read4, read4);
                }
            }
        }
        else
        {
            static_unroll_count<unsigned int, 0, less, hip_plugin_bn_config::launch_dim.grp0, 4>{
                [&](unsigned int k) {
                    if(k + lid < less)
                    {
                        batchIdx = (k + lid) / hip_plugin_bn_config::spatial_size;
                        lidx = (k + lid) - (batchIdx * hip_plugin_bn_config::spatial_size);
                        if constexpr(hip_plugin_config::layout_channel_last)
                        {
                            index = batchIdx * hip_plugin_bn_config::batch_size
                                    + lidx * hip_plugin_bn_config::channels + grpid;
                        }
                        else
                        {
                            index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId
                                    + lidx;
                        }
                        const auto xin = hip_kernel_provider::cast<ComputeType>(in[index]);
                        mean += xin;
                        variance = fma(xin, xin, variance);
                    }
                }};

            if constexpr(rem > 0u)
            {
                // Note: hip compiler has a bug, it throws compiler warning for comparing unsigned
                // int with 0 value, when rem is 0. but when rem is 0, this code block should not be
                // compiled due to the if constexpr used above.
                if(lid < rem)
                {
                    unsigned int remkey = lid + less;
                    batchIdx = remkey / hip_plugin_bn_config::spatial_size;
                    lidx = remkey - (batchIdx * hip_plugin_bn_config::spatial_size);
                    if constexpr(hip_plugin_config::layout_channel_last)
                    {
                        index = batchIdx * hip_plugin_bn_config::batch_size
                                + lidx * hip_plugin_bn_config::channels + grpid;
                    }
                    else
                    {
                        index
                            = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;
                    }

                    const auto xin = index < hip_plugin_bn_config::tensor_size
                                         ? hip_kernel_provider::cast<ComputeType>(in[index])
                                         : hip_kernel_provider::cast<ComputeType>(0.0);
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto lcl_data_size = hip_plugin_bn_config::use_amdgcn
                                           ? hip_plugin_bn_config::lds_gcn_size
                                           : hip_plugin_bn_config::lds_size;
        __shared__ ComputeType lcl_data_x[lcl_data_size];
        __shared__ ComputeType lcl_data_y[lcl_data_size];
        if constexpr(hip_plugin_bn_config::use_amdgcn)
        {
            hip_kernel_provider::batchnorm::reduction::gcn_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::lds_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }

        // REDUCTION COMPLETE ---------------------------
        variance = fmax(fma(-mean, mean, variance), hip_kernel_provider::cast<MeanVarType>(0.0));

        // unsafe: casting double to ComputeType
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<ComputeType>(epsilon));
        pvscale = lcl_scale;
        pvbias = lcl_bias;
        if constexpr(hip_plugin_config::layout_channel_last || rem == 0)
        {
            constexpr unsigned int k_limit = hip_plugin_config::layout_channel_last
                                                 ? hip_plugin_bn_config::channel_slice_size
                                                 : less;

            static_unroll_count<unsigned int, 0, k_limit, hip_plugin_bn_config::launch_dim.grp0, 2>{
                [&](unsigned int k) {
                    if(k + lid < k_limit)
                    {
                        batchIdx = (k + lid) / hip_plugin_bn_config::spatial_size;
                        lidx = (k + lid) - (batchIdx * hip_plugin_bn_config::spatial_size);
                        if constexpr(hip_plugin_config::layout_channel_last)
                        {
                            index = batchIdx * hip_plugin_bn_config::batch_size
                                    + lidx * hip_plugin_bn_config::channels + grpid;
                        }
                        else
                        {
                            index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId
                                    + lidx;
                        }

                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::neuron_op>(
                                fma(pvscale,
                                    (hip_kernel_provider::cast<ComputeType>(in[index]) - mean)
                                        * invVariance,
                                    pvbias),
                                alpha,
                                beta));
                    }
                }};
        }
        else
        {
            ComputeType xhat[max_read];

            static_unroll_count<unsigned int, 0, lessout, chunk, 2>{[&](unsigned int k) {
                if(k + (max_read * lid) < lessout)
                {
                    for(unsigned int j = 0; j < max_read; ++j)
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        batchIdx = l / hip_plugin_bn_config::spatial_size;
                        lidx = l - (batchIdx * hip_plugin_bn_config::spatial_size);
                        index
                            = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;
                        xhat[j] = (hip_kernel_provider::cast<ComputeType>(in[index]) - mean)
                                  * invVariance;
                    }

                    __syncthreads();

                    for(unsigned int j = 0; j < max_read; ++j) // This part takes 0.405
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        batchIdx = l / hip_plugin_bn_config::spatial_size;
                        lidx = l - (batchIdx * hip_plugin_bn_config::spatial_size);
                        index
                            = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;
                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::neuron_op>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }};

            if constexpr(remout > 0u)
            {
                const unsigned int remkeyout = (max_read * lid) + lessout;
                for(unsigned int j = 0; j < max_read; ++j)
                {
                    unsigned int l = remkeyout + j;
                    batchIdx = l / hip_plugin_bn_config::spatial_size;
                    lidx = l - (batchIdx * hip_plugin_bn_config::spatial_size);
                    index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;
                    // TODO: comparing different types
                    const auto xin = index < hip_plugin_bn_config::tensor_size
                                         ? hip_kernel_provider::cast<ComputeType>(in[index])
                                         : hip_kernel_provider::cast<ComputeType>(0.0);
                    xhat[j] = (xin - hip_kernel_provider::cast<ComputeType>(mean))
                              * hip_kernel_provider::cast<ComputeType>(invVariance);
                }

                __syncthreads();
                for(unsigned int j = 0; j < max_read; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    batchIdx = l / hip_plugin_bn_config::spatial_size;
                    lidx = l - (batchIdx * hip_plugin_bn_config::spatial_size);
                    index = batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + lidx;

                    if(index < hip_plugin_bn_config::tensor_size)
                    {
                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::neuron_op>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }
        }
        return;
    }
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 3
template <>
struct BatchNormFwdTrainSpatialImpl<3>
{
    __forceinline__ __device__ void operator()(const InputType* __restrict in,
                                               OutputType* __restrict out,
                                               const ScaleBiasType* __restrict scale,
                                               const ScaleBiasType* __restrict bias,
                                               double epsilon,
                                               MeanVarType& mean,
                                               MeanVarType& variance,
                                               MeanVarType& invVariance,
                                               ComputeType alpha,
                                               ComputeType beta)
    {
        // SPATIAL
        mean = hip_kernel_provider::cast<MeanVarType>(0.0);
        variance = hip_kernel_provider::cast<MeanVarType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarType>(0.0);
        ComputeType inhat = hip_kernel_provider::cast<ComputeType>(0.0);
        ComputeType pvscale = hip_kernel_provider::cast<ComputeType>(0.0);
        ComputeType pvbias = hip_kernel_provider::cast<ComputeType>(0.0);
        ComputeType xin = hip_kernel_provider::cast<ComputeType>(0.0);

        __shared__ ComputeType lcl_scale;
        __shared__ ComputeType lcl_bias;

        unsigned int index = 0;
        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int channelIdx = grpid * hip_plugin_bn_config::spatial_size;

        // Unused if hip_plugin_bn_config::batches >= hip_plugin_bn_config::max_batches
        InputType minibatch[hip_plugin_bn_config::batches];

        if(lid == 0)
        {
            lcl_scale = scale[grpid];
            lcl_bias = bias[grpid];
        }

        if(lid < hip_plugin_bn_config::spatial_size)
        {
            static_unroll_count<unsigned int, 0, hip_plugin_bn_config::batches, 1, 2>{
                [&](unsigned int batch) {
                    index = batch * hip_plugin_bn_config::batch_size + channelIdx + lid;
                    xin = hip_kernel_provider::cast<ComputeType>(in[index]);
                    mean += xin;
                    variance = fma(xin, xin, variance);

                    if constexpr(hip_plugin_bn_config::batches < hip_plugin_bn_config::max_batches)
                    {
                        minibatch[batch] = in[index];
                    }
                }};
        }
        __syncthreads();

        constexpr auto lcl_data_size = hip_plugin_bn_config::use_amdgcn
                                           ? hip_plugin_bn_config::lds_gcn_size
                                           : hip_plugin_bn_config::lds_size;
        __shared__ ComputeType lcl_data_x[lcl_data_size];
        __shared__ ComputeType lcl_data_y[lcl_data_size];
        if constexpr(hip_plugin_bn_config::use_amdgcn)
        {
            hip_kernel_provider::batchnorm::reduction::gcn_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::lds_reduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lcl_data_x,
                lcl_data_y,
                lid);
        }

        variance = fmax(fma(-mean, mean, variance), hip_kernel_provider::cast<MeanVarType>(0.0));
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<ComputeType>(epsilon));

        if(lid < hip_plugin_bn_config::spatial_size)
        {
            pvscale = lcl_scale;
            pvbias = lcl_bias;
            static_unroll_count<unsigned int, 0, hip_plugin_bn_config::batches, 1, 2>{
                [&](unsigned int batch) { // apply normalization
                    index = batch * hip_plugin_bn_config::batch_size + channelIdx + lid;
                    if constexpr(hip_plugin_bn_config::batches < hip_plugin_bn_config::max_batches)
                    {
                        inhat = (hip_kernel_provider::cast<ComputeType>(minibatch[batch]) - mean)
                                * invVariance; // (in[index] - mean) * invVariance;
                    }
                    else
                    {
                        inhat = (hip_kernel_provider::cast<ComputeType>(in[index]) - mean)
                                * invVariance;
                    }
                    out[index] = hip_kernel_provider::cast<OutputType>(
                        hip_kernel_provider::applyActivation<ComputeType,
                                                             hip_plugin_config::neuron_op>(
                            fma(pvscale, inhat, pvbias), alpha, beta));
                }}; // end for

        } // end if
    }
};

// these are the kernels for HIP_PLUGIN_BN_VARIANT == 2
#if(HIP_PLUGIN_BN_VARIANT == 2)

struct BatchNormFwdTrainSpatialImplVar2
{
    static constexpr unsigned int ngrps = hip_plugin_bn_config::launch_groups.grp1;
    static constexpr unsigned int ngrps2 = hip_plugin_bn_config::launch_groups.grp2;

    static __forceinline__ __device__ void Norm(const InputType* __restrict__ in,
                                                OutputType* __restrict__ out,
                                                const ScaleBiasType* scale,
                                                const ScaleBiasType* bias,
                                                ComputeType alpha,
                                                ComputeType beta)
    {
        unsigned int xstride
            = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
        unsigned int ystride
            = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_size = hip_plugin_bn_config::launch_dim.grp0;
        unsigned int ygrp_size = hip_plugin_bn_config::launch_dim.grp1;
        unsigned int zgrp_size = hip_plugin_bn_config::launch_dim.grp2;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xgid = xgrp_id * xgrp_size + xlid;
        unsigned int ygid = ygrp_id * ygrp_size + ylid;
        unsigned int zgid = zgrp_id * zgrp_size + zlid;

        ComputeChannelVecType mean;
        ComputeChannelVecType invVariance;
        ComputeChannelVecType pvt_scale;
        ComputeChannelVecType pvt_bias;

        __shared__ ComputeChannelVecType lcl_bias[hip_plugin_bn_config::launch_dim.grp0];
        __shared__ ComputeChannelVecType lcl_scale[hip_plugin_bn_config::launch_dim.grp0];
        __shared__ ComputeChannelVecType lcl_mean[hip_plugin_bn_config::launch_dim.grp0];
        __shared__ ComputeChannelVecType lcl_ivar[hip_plugin_bn_config::launch_dim.grp0];

        if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
        {
            return;
        }

        // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        if(ylid == 0 && zlid == 0)
        {
            lcl_scale[xlid] = *reinterpret_cast<const ComputeChannelVecType*>(
                scale + xgid * hip_plugin_bn_config::vec_size_x);
            lcl_bias[xlid] = *reinterpret_cast<const ComputeChannelVecType*>(
                bias + xgid * hip_plugin_bn_config::vec_size_x);
            lcl_mean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType,
                                                                           OutputChannelVecType>(
                reinterpret_cast<const OutputChannelVecType*>(out),
                0,
                zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
                ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
            lcl_ivar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType,
                                                                           OutputChannelVecType>(
                reinterpret_cast<const OutputChannelVecType*>(out),
                1,
                zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
                ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
        }
        __syncthreads();

        if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::spatial_size
           && zgid < hip_plugin_bn_config::batches)
        {
            mean = lcl_mean[xlid];
            invVariance = lcl_ivar[xlid];
            pvt_scale = lcl_scale[xlid];
            pvt_bias = lcl_bias[xlid];
            unsigned int index_base
                = zgid * hip_plugin_bn_config::batch_elements * hip_plugin_bn_config::batch_size
                  + ygid * ystride * hip_plugin_bn_config::vec_size_y
                  + xgid * xstride * hip_plugin_bn_config::vec_size_x;

            // The original OpenCL kernel unrolled the loop only when this condition was met.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr unsigned int unrollHint
                = hip_plugin_bn_config::spatial_size > hip_plugin_bn_config::loop_unroll_max_spatial
                      ? 1
                      : 2;

            // This method of unrolling is used as opposed to
            // the struct static_unroll_count due to a bug where
            // the kernel writes a tensor for the output twice bigger
            // than it should be, and it repeats values equal to the number of channels
            // in the examined test cases that were failing.

#if(HIP_PLUGIN_BN_SPATIAL_SIZE > HIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL)
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
#else
#pragma unroll(2)
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
#endif
            {
                unsigned int index = index_base + batch * hip_plugin_bn_config::batch_size;
                auto value = hip_kernel_provider::cast<ComputeVecType>(
                    *reinterpret_cast<const InputVecType*>(in + index));
                // This is a float4 when not vectorizing; for FPMIX ComputeVecType is float, FpType is __half
                ComputeVecType inhat
                    = (hip_kernel_provider::cast<ComputeVecType>(value) - mean) * invVariance;
                inhat
                    = hip_kernel_provider::fma(hip_kernel_provider::cast<ComputeVecType>(pvt_scale),
                                               inhat,
                                               hip_kernel_provider::cast<ComputeVecType>(pvt_bias));

                value = hip_kernel_provider::applyActivation<ComputeVecType,
                                                             hip_plugin_config::neuron_op>(
                    inhat,
                    hip_kernel_provider::cast<ComputeVecType>(alpha),
                    hip_kernel_provider::cast<ComputeVecType>(beta));

                *reinterpret_cast<OutputVecType*>(out + index)
                    = hip_kernel_provider::cast<OutputVecType>(value);
            } // end for(batch)
        } // end if(inImgIndex)
    } // end spatial norm

    static __forceinline__ __device__ void FinalMeanVariance(
        OutputType* __restrict__ meanvarbuff, // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
        double epsilon,
        const unsigned int& xgid,
        const unsigned int& ygid,
        const unsigned int& zgid,
        unsigned int& commitID,
        MeanVarChannelVecType& mean,
        MeanVarChannelVecType& variance,
        MeanVarChannelVecType& invVariance)
    {
        variance = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);
        mean = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);

        unsigned int xgrp_id = blockIdx.x;

        // These values (?grp_size) cannot be substituted with hip_plugin_bn_config::launch_dim.grp? because
        // the dimensions of the blocks for this kernel may be different from the other
        // kernels that take part in the operation. Given that the launch dimensions are
        // rounded up, blockDim would return a consistent value.

        unsigned int xgrp_size = blockDim.x;
        unsigned int ygrp_size = blockDim.y;
        unsigned int zgrp_size = blockDim.z;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xstride
            = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
        unsigned int ystride
            = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

        commitID = 0;

        for(unsigned int zoffset = zlid; zoffset < hip_plugin_bn_config::launch_groups.grp2;
            zoffset += zgrp_size)
        {
            for(unsigned int yoffset = ylid; yoffset < hip_plugin_bn_config::launch_groups.grp1;
                yoffset += ygrp_size)
            {
                mean += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    0,
                    hip_plugin_bn_config::launch_dim.grp2 * zoffset
                        * hip_plugin_bn_config::batch_elements,
                    hip_plugin_bn_config::launch_dim.grp1 * yoffset
                        * hip_plugin_bn_config::vec_size_y,
                    ystride / hip_plugin_bn_config::vec_size_x,
                    xgrp_size,
                    xgrp_id,
                    xlid,
                    xstride);
                variance += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    1,
                    hip_plugin_bn_config::launch_dim.grp2 * zoffset
                        * hip_plugin_bn_config::batch_elements,
                    hip_plugin_bn_config::launch_dim.grp1 * yoffset
                        * hip_plugin_bn_config::vec_size_y,
                    ystride / hip_plugin_bn_config::vec_size_x,
                    xgrp_size,
                    xgrp_id,
                    xlid,
                    xstride);
            }
        }

        if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                     || (hip_plugin_bn_config::lds_gcn_size == 1)
                     || hip_plugin_bn_config::vec_size_x > 1
                     || (hip_plugin_bn_config::launch_dim_final.bounds < 64))
        {
            __shared__ ComputeChannelVecType
                lcl_data[2 * hip_plugin_bn_config::launch_dim_final.bounds];

            hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(
                static_cast<ComputeChannelVecType&>(mean),
                static_cast<ComputeChannelVecType&>(variance),
                hip_plugin_bn_config::i_channel_slice_size,
                lcl_data,
                xgrp_size,
                xlid,
                ylid + zlid * ygrp_size,
                ygrp_size * zgrp_size);
        }
        else
        {
            // C++17 idiomatic: ensure array size is never zero using constexpr ternary
            constexpr auto lds_gcn_array_size
                = hip_plugin_bn_config::launch_dim_final.bounds >= 64
                      ? hip_plugin_bn_config::launch_dim_final.bounds / 64
                      : 1;

            commitID = 64;
            __shared__ ComputeChannelVecType lcl_data_x[lds_gcn_array_size];
            __shared__ ComputeChannelVecType lcl_data_y[lds_gcn_array_size];
            hip_kernel_provider::batchnorm::reduction::gcn_reduce2(
                static_cast<ComputeChannelVecType&>(mean),
                static_cast<ComputeChannelVecType&>(variance),
                hip_plugin_bn_config::i_channel_slice_size,
                lcl_data_x,
                lcl_data_y,
                ylid + zlid * ygrp_size);
        }

        variance = hip_kernel_provider::fma(-mean, mean, variance);
        variance
            = hip_kernel_provider::max(hip_kernel_provider::cast<ComputeChannelVecType>(variance),
                                       hip_kernel_provider::cast<ComputeChannelVecType>(0.0));
        invVariance = hip_kernel_provider::rsqrt(
            hip_kernel_provider::cast<ComputeChannelVecType>(variance)
            + hip_kernel_provider::cast<ComputeChannelVecType>(epsilon));

        for(unsigned int zoffset = zlid; zoffset < hip_plugin_bn_config::launch_groups.grp2;
            zoffset += zgrp_size)
        {
            for(unsigned int yoffset = ylid; yoffset < hip_plugin_bn_config::launch_groups.grp1;
                yoffset += ygrp_size)
            {
                storeToStash(mean,
                             reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                             0,
                             hip_plugin_bn_config::launch_dim.grp2 * zoffset
                                 * hip_plugin_bn_config::batch_elements,
                             hip_plugin_bn_config::launch_dim.grp1 * yoffset
                                 * hip_plugin_bn_config::vec_size_y,
                             ystride / hip_plugin_bn_config::vec_size_x,
                             xgrp_size,
                             xgrp_id,
                             xlid,
                             xstride);
                storeToStash(invVariance,
                             reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                             1,
                             hip_plugin_bn_config::launch_dim.grp2 * zoffset
                                 * hip_plugin_bn_config::batch_elements,
                             hip_plugin_bn_config::launch_dim.grp1 * yoffset
                                 * hip_plugin_bn_config::vec_size_y,
                             ystride / hip_plugin_bn_config::vec_size_x,
                             xgrp_size,
                             xgrp_id,
                             xlid,
                             xstride);
            }
        }
    }

    static __forceinline__ __device__ void MeanVariance(
        const InputType* __restrict__ in,
        OutputType* __restrict__ meanvarbuff) // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
    {
        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_size = hip_plugin_bn_config::launch_dim.grp0;
        unsigned int ygrp_size = hip_plugin_bn_config::launch_dim.grp1;
        unsigned int zgrp_size = hip_plugin_bn_config::launch_dim.grp2;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xgid = xgrp_id * xgrp_size + xlid;
        unsigned int ygid = ygrp_id * ygrp_size + ylid;
        unsigned int zgid = zgrp_id * zgrp_size + zlid;

        unsigned int xstride
            = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
        unsigned int ystride
            = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

        unsigned int index;

        auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
        auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

        if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
        {
            return;
        }

        if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::spatial_size
           && zgid < hip_plugin_bn_config::batches)
        {
            unsigned int index_base
                = zgid * hip_plugin_bn_config::batch_elements * hip_plugin_bn_config::batch_size
                  + ygid * ystride * hip_plugin_bn_config::vec_size_y
                  + xgid * xstride * hip_plugin_bn_config::vec_size_x;
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
            {
                index = index_base + batch * hip_plugin_bn_config::batch_size;
                auto read4 = hip_kernel_provider::cast<ComputeVecType>(
                    *reinterpret_cast<const InputVecType*>(in + index));

                hip_kernel_provider::batchnorm::_accumulate(mean, read4);
                hip_kernel_provider::batchnorm::_accumulate_mad(variance, read4, read4);
            }
        }

        if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                     || (hip_plugin_bn_config::lds_gcn_size == 1)
                     || hip_plugin_bn_config::vec_size_x > 1)
        {
            __shared__ ComputeChannelVecType lcl_data[2 * hip_plugin_bn_config::lds_size];
            hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(
                mean,
                variance,
                hip_kernel_provider::cast<ComputeType>(1.0),
                lcl_data,
                xgrp_size,
                xlid,
                ylid + zlid * ygrp_size,
                ygrp_size * zgrp_size);
        }
        else
        {
            __shared__ ComputeChannelVecType lcl_data_x[hip_plugin_bn_config::lds_gcn_size];
            __shared__ ComputeChannelVecType lcl_data_y[hip_plugin_bn_config::lds_gcn_size];
            hip_kernel_provider::batchnorm::reduction::gcn_reduce2(
                mean,
                variance,
                hip_kernel_provider::cast<ComputeType>(1.0),
                lcl_data_x,
                lcl_data_y,
                ylid + zlid * ygrp_size);
        }

        if(ylid == 0 && zlid == 0)
        {
            storeToStash(mean,
                         reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                         0,
                         zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
                         ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
                         ystride / hip_plugin_bn_config::vec_size_x,
                         xgrp_size,
                         xgrp_id,
                         xlid,
                         xstride);
            storeToStash(variance,
                         reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                         1,
                         zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
                         ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
                         ystride / hip_plugin_bn_config::vec_size_x,
                         xgrp_size,
                         xgrp_id,
                         xlid,
                         xstride);
        }
    }
};

#endif // HIP_PLUGIN_BN_VARIANT == 2

} // namespace hip_kernel_provider::batchnorm

/// C interfaces

// TODO: This can be removed after every variant has been implemnted
// [[deprecated]]
#if(HIP_PLUGIN_BN_VARIANT != 2)
extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormFwdTrainSpatial(const InputType* __restrict in,
                             OutputType* __restrict out,
                             const ScaleBiasType* __restrict scale,
                             const ScaleBiasType* __restrict bias,
// TODO: should find a better way of doing this
// but it's hard because C does not support function
// overloads.
// [[deprecated]]
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
                             double expAvgFactor,
                             const MeanVarType* __restrict prevResultRunningMean,
                             const MeanVarType* __restrict prevResultRunningVariance,
                             MeanVarType* __restrict nextResultRunningMean,
                             MeanVarType* __restrict nextResultRunningVariance,
#endif
                             double epsilon,
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
                             ComputeType* __restrict resultSaveMean,
                             ComputeType* __restrict resultSaveInvVariance,
#endif
                             ComputeType alpha,
                             ComputeType beta)
{
    constexpr auto variant = hip_plugin_bn_config::variant;

    using forward_train_spatial_impl
        = hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImpl<variant>;

    MeanVarType mean, variance, invVariance;
    const unsigned int lid = threadIdx.x;
    const unsigned int grpid = blockIdx.x;

    forward_train_spatial_impl{}(
        in, out, scale, bias, epsilon, mean, variance, invVariance, alpha, beta);

    if(lid == 0)
    {
// TODO: this should also be removed, but using constexpr can lead to compile errors
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<MeanVarChannelVecType>;
        StashUpdater updater(static_cast<MeanVarChannelVecType>(mean),
                             static_cast<MeanVarChannelVecType>(variance),
                             static_cast<MeanVarChannelVecType>(expAvgFactor));

        hip_kernel_provider::batchnorm::running_stash<MeanVarChannelVecType,
                                                      ComputeChannelVecType,
                                                      StashUpdater>(prevResultRunningMean,
                                                                    prevResultRunningVariance,
                                                                    nextResultRunningMean,
                                                                    nextResultRunningVariance,
                                                                    updater,
                                                                    grpid);
#endif
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::saved_stash<MeanVarChannelVecType, ComputeChannelVecType>(
            resultSaveMean,
            resultSaveInvVariance,
            static_cast<MeanVarChannelVecType>(mean),
            static_cast<MeanVarChannelVecType>(invVariance),
            grpid);
#endif
    }
}

#else

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormFwdTrainSpatialNorm(const InputType* __restrict__ in,
                                 OutputType* __restrict__ out,
                                 const ScaleBiasType* scale,
                                 const ScaleBiasType* bias,
                                 ComputeType alpha,
                                 ComputeType beta)
{
    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2{}.Norm(
        in, out, scale, bias, alpha, beta);
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim_final.bounds)
    BatchNormFwdTrainSpatialFinalMeanVariance(
        OutputType* __restrict__ meanvarbuff, // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        double expAvgFactor, /* input momentum */
        const MeanVarType* __restrict__ prevResultRunningMean,
        const MeanVarType* __restrict__ prevResultRunningVariance,
        MeanVarType* __restrict__ nextResultRunningMean,
        MeanVarType* __restrict__ nextResultRunningVariance,
#endif
        double epsilon
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        ,
        ComputeType* __restrict__ resultSaveMean, /*output only*/
        ComputeType* __restrict__ resultSaveInvVariance
#endif
    )
{
    MeanVarChannelVecType mean;
    MeanVarChannelVecType variance;
    MeanVarChannelVecType invVariance;

    unsigned int xgid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int zgid = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int commitID;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
        return;

    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2{}.FinalMeanVariance(
        meanvarbuff, epsilon, xgid, ygid, zgid, commitID, mean, variance, invVariance);
    // Save mean and calculate and save running mean
    if(ygid == commitID && zgid == 0)
    {
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<MeanVarChannelVecType>;
        StashUpdater updater(hip_kernel_provider::cast<MeanVarChannelVecType>(mean),
                             hip_kernel_provider::cast<MeanVarChannelVecType>(variance),
                             hip_kernel_provider::cast<MeanVarChannelVecType>(expAvgFactor));

        hip_kernel_provider::batchnorm::
            running_stash<ComputeChannelVecType, MeanVarChannelVecType, StashUpdater>(
                reinterpret_cast<const MeanVarChannelVecType*>(prevResultRunningMean),
                reinterpret_cast<const MeanVarChannelVecType*>(prevResultRunningVariance),
                reinterpret_cast<MeanVarChannelVecType*>(nextResultRunningMean),
                reinterpret_cast<MeanVarChannelVecType*>(nextResultRunningVariance),
                updater,
                xgid);
#endif

#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::saved_stash<MeanVarChannelVecType, ComputeChannelVecType>(
            reinterpret_cast<ComputeChannelVecType*>(resultSaveMean),
            reinterpret_cast<ComputeChannelVecType*>(resultSaveInvVariance),
            mean,
            invVariance,
            xgid);
#endif
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormFwdTrainSpatialMeanVariance(
        const InputType* __restrict__ in,
        OutputType* __restrict__ meanvarbuff) // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
{
    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2{}.MeanVariance(in,
                                                                                    meanvarbuff);
}

#endif
