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
    static constexpr unsigned int SEGTMP_1
        = hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
              .GRP0 // NOLINT(readability-static-accessed-through-instance)
          / hip_plugin_bn_config::
              SPATIAL_SIZE; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int SEGTMP_2 = (SEGTMP_1 == 0) ? 1 : SEGTMP_1;
    static constexpr unsigned int SEGTMP = hip_plugin_bn_config::SPATIAL_SIZE * SEGTMP_2;
    static constexpr unsigned int SEGMENT = (SEGTMP > hip_plugin_bn_config::CHANNEL_SLICE_SIZE)
                                                ? hip_plugin_bn_config::CHANNEL_SLICE_SIZE
                                                : SEGTMP;
    static constexpr unsigned int BATCH_LOOP
        = (hip_plugin_bn_config::CHANNEL_SLICE_SIZE + SEGMENT - 1) / SEGMENT;
    static constexpr unsigned int SEGI_SPATIAL = SEGMENT / hip_plugin_bn_config::SPATIAL_SIZE;
    static constexpr unsigned int BATCH_LOOPM = BATCH_LOOP - 1;
    static constexpr unsigned int SN_SPATIAL = BATCH_LOOPM * SEGI_SPATIAL;

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
        InputType batchvalues[BATCH_LOOP]; // NOLINT(modernize-avoid-c-arrays)
        ComputeType temp;

        __shared__ ComputeType s_lcl_scale;
        __shared__ ComputeType s_lcl_bias;

        unsigned int index = 0;
        const unsigned int lid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        unsigned int channelSpatialId = grpid * hip_plugin_bn_config::SPATIAL_SIZE
                                        + (lid % hip_plugin_bn_config::SPATIAL_SIZE);
        unsigned int lidiSpatial = lid / hip_plugin_bn_config::SPATIAL_SIZE;
        unsigned int batchId = 0;

        if(lid == 0)
        {
            s_lcl_scale = scale[grpid];
            s_lcl_bias = bias[grpid];
        }

        __syncthreads();

        if(lid < SEGMENT)
        {
            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this UNROLL_HINT and the StaticUnrollCount struct replicates this.
            constexpr int UNROLL_HINT
                = hip_plugin_config::INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX ? 2 : 1;
            StaticUnrollCount<unsigned int, 0, BATCH_LOOPM, 1, UNROLL_HINT>{
                [&](unsigned int batch) {
                    batchId = batch * SEGI_SPATIAL + lidiSpatial;
                    index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
                    batchvalues[batch] = *(in + index);
                    temp = hip_kernel_provider::cast<ComputeType>(in[index]);
                    mean += temp;
                    variance = fma(temp, temp, variance);
                }};
            batchId = SN_SPATIAL + lidiSpatial;
            index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
            batchvalues[BATCH_LOOPM] = (index < hip_plugin_bn_config::TENSOR_SIZE)
                                           ? in[index]
                                           : hip_kernel_provider::cast<InputType>(0.0);
            temp = hip_kernel_provider::cast<ComputeType>(batchvalues[BATCH_LOOPM]);
            mean += temp;
            variance = fma(temp, temp, variance);
        }
        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ ComputeType s_lcl_data_x[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        __shared__ ComputeType s_lcl_data_y[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        // Reduction complete

        variance = fma(-mean, mean, variance);
        // NOLINTNEXTLINE(readability-use-std-min-max): <algorithm> is unavailable in hipRTC
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = hip_kernel_provider::rsqrt(variance
                                                 + hip_kernel_provider::cast<ComputeType>(epsilon));

        auto pvscale = hip_kernel_provider::cast<ComputeType>(s_lcl_scale);
        auto pvbias = hip_kernel_provider::cast<ComputeType>(s_lcl_bias);

        if(lid < SEGMENT)
        {
            // Calculate norm

            auto inhat = hip_kernel_provider::cast<ComputeType>(0.0);

            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this UNROLL_HINT and the StaticUnrollCount struct replicates this.
            constexpr int UNROLL_HINT
                = hip_plugin_config::INPUT_TYPE_STRATEGY == TypeStrategy::FPMIX ? 2 : 1;

            StaticUnrollCount<unsigned int, 0, BATCH_LOOPM, 1, UNROLL_HINT>{
                [&](unsigned int batch) {
                    // Apply normalization
                    inhat = (hip_kernel_provider::cast<ComputeType>(batchvalues[batch]) - mean)
                            * invVariance;
                    batchId = batch * SEGI_SPATIAL + lidiSpatial;
                    index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
                    auto value
                        = hip_kernel_provider::cast<ComputeType>(fma(pvscale, inhat, pvbias));
                    value = hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::NEURON_OP>(
                        value, alpha, beta);
                    out[index] = hip_kernel_provider::cast<OutputType>(value);
                }};

            // Tail of loop
            inhat = (hip_kernel_provider::cast<ComputeType>(batchvalues[BATCH_LOOPM]) - mean)
                    * invVariance;
            batchId = SN_SPATIAL + lidiSpatial;
            index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
            if(index < hip_plugin_bn_config::TENSOR_SIZE)
            {
                auto value = hip_kernel_provider::cast<ComputeType>(fma(pvscale, inhat, pvbias));
                value = hip_kernel_provider::applyActivation<ComputeType,
                                                             hip_plugin_config::NEURON_OP>(
                    value, alpha, beta);
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
    static constexpr unsigned int MAX_READ
        = hip_plugin_config::LAYOUT_CHANNEL_LAST
              ? 1
              : (hip_plugin_bn_config:: // NOLINT(readability-avoid-nested-conditional-operator)
                         SPATIAL_SIZE // NOLINT(readability-avoid-nested-conditional-operator)
                         >= 4096 // NOLINT(readability-avoid-nested-conditional-operator)
                     ? 3
                     : 2); // NOLINT(readability-avoid-nested-conditional-operator)
    static constexpr unsigned int READ_BLOCK = 1;
    static constexpr unsigned int GRPREAD
        = hip_plugin_config::LAYOUT_CHANNEL_LAST
              ? (hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                 LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                     .GRP0 // NOLINT(readability-static-accessed-through-instance)
                 * READ_BLOCK) // NOLINT(readability-static-accessed-through-instance)
              : (hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                 LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                     .GRP0 // NOLINT(readability-static-accessed-through-instance)
                 * READ_BLOCK // NOLINT(readability-static-accessed-through-instance)
                 * 4); // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int REM4
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - ((hip_plugin_bn_config::CHANNEL_SLICE_SIZE / GRPREAD) * GRPREAD);
    static constexpr unsigned int LESS4 = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REM4;
    static constexpr unsigned int CHUNK4 = MAX_READ * GRPREAD;
    static constexpr unsigned int REMOUT4
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - ((hip_plugin_bn_config::CHANNEL_SLICE_SIZE / CHUNK4) * CHUNK4);
    static constexpr unsigned int LESSOUT4 = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REMOUT4;
    static constexpr unsigned int REM
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - ((hip_plugin_bn_config::CHANNEL_SLICE_SIZE
              / hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                    .GRP0) // NOLINT(readability-static-accessed-through-instance)
             * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
               LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                   .GRP0); // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int LESS = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REM;
    static constexpr unsigned int CHUNK
        = MAX_READ
          * hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                .GRP0; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int REMOUT
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - ((hip_plugin_bn_config::CHANNEL_SLICE_SIZE / CHUNK) * CHUNK);
    static constexpr unsigned int LESSOUT = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REMOUT;

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
        ComputeType pvscale;
        ComputeType pvbias;

        mean = hip_kernel_provider::cast<MeanVarType>(0.0);
        variance = hip_kernel_provider::cast<MeanVarType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarType>(0.0);

        __shared__ ComputeType s_lcl_scale;
        __shared__ ComputeType s_lcl_bias;

        unsigned int index = 0;
        const unsigned int lid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)

        // Note: this variable is only used when hip_plugin_config::LAYOUT_CHANNEL_LAST is false.
        unsigned int channelSpatialId;
        if constexpr(!hip_plugin_config::LAYOUT_CHANNEL_LAST)
        {
            channelSpatialId = grpid * hip_plugin_bn_config::SPATIAL_SIZE;
        }

        unsigned int batchIdx = 0;
        unsigned int lidx = 0;

        if(lid == 0)
        {
            s_lcl_scale = *(scale + grpid);
            s_lcl_bias = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!hip_plugin_config::LAYOUT_CHANNEL_LAST
                     && hip_plugin_bn_config::SPATIAL_SIZE >= 4096)
        {
            using InputVecType4 = typename MappedVectorType<InputType, 4>::type;
            InputVecType4 read4;

            StaticUnrollCount<unsigned int, 0, LESS4, GRPREAD, 2>{[&](unsigned int k) {
                if((k + (lid << 2)) < LESS4)
                {
                    batchIdx = (k + (lid << 2)) / hip_plugin_bn_config::SPATIAL_SIZE;
                    lidx = (k + (lid << 2)) - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                    index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;
                    read4 = *reinterpret_cast<const InputVecType4*>(in + index);
                    hip_kernel_provider::batchnorm::accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                }
            }};

            if constexpr(REM4 > 0u)
            {
                const unsigned int remkey = (lid << 2) + LESS4;
                batchIdx = remkey / hip_plugin_bn_config::SPATIAL_SIZE;
                lidx = remkey - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;

                // index is unsigned int, so if the result would normally end up negative,
                // the value wraps around and the check fails. Improves on the
                // previous way of handling which was: if(index < (hip_plugin_bn_config::TENSOR_SIZE - 3))
                if(index + 3 < (hip_plugin_bn_config::TENSOR_SIZE))
                {
                    read4 = *reinterpret_cast<const InputVecType4*>(in + index);
                    hip_kernel_provider::batchnorm::accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                }
            }
        }
        else
        {
            // NOLINTNEXTLINE(readability-static-accessed-through-instance)
            StaticUnrollCount<unsigned int, 0, LESS, hip_plugin_bn_config::LAUNCH_DIM.GRP0, 4>{
                [&](unsigned int k) {
                    if(k + lid < LESS)
                    {
                        batchIdx = (k + lid) / hip_plugin_bn_config::SPATIAL_SIZE;
                        lidx = (k + lid) - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                        if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST)
                        {
                            index = batchIdx * hip_plugin_bn_config::BATCH_SIZE
                                    + lidx * hip_plugin_bn_config::CHANNELS + grpid;
                        }
                        else
                        {
                            index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId
                                    + lidx;
                        }
                        const auto xin = hip_kernel_provider::cast<ComputeType>(in[index]);
                        mean += xin;
                        variance = fma(xin, xin, variance);
                    }
                }};

            if constexpr(REM > 0u)
            {
                // Note: hip compiler has a bug, it throws compiler warning for comparing unsigned
                // int with 0 value, when REM is 0. but when REM is 0, this code block should not be
                // compiled due to the if constexpr used above.
                if(lid < REM)
                {
                    const unsigned int remkey = lid + LESS;
                    batchIdx = remkey / hip_plugin_bn_config::SPATIAL_SIZE;
                    lidx = remkey - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                    if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST)
                    {
                        index = batchIdx * hip_plugin_bn_config::BATCH_SIZE
                                + lidx * hip_plugin_bn_config::CHANNELS + grpid;
                    }
                    else
                    {
                        index
                            = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;
                    }

                    const auto xin = index < hip_plugin_bn_config::TENSOR_SIZE
                                         ? hip_kernel_provider::cast<ComputeType>(in[index])
                                         : hip_kernel_provider::cast<ComputeType>(0.0);
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ ComputeType s_lcl_data_x[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        __shared__ ComputeType s_lcl_data_y[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        // REDUCTION COMPLETE ---------------------------
        variance = fmax(fma(-mean, mean, variance), hip_kernel_provider::cast<MeanVarType>(0.0));

        // unsafe: casting double to ComputeType
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<ComputeType>(epsilon));
        pvscale = s_lcl_scale;
        pvbias = s_lcl_bias;
        if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST || REM == 0)
        {
            constexpr unsigned int K_LIMIT = hip_plugin_config::LAYOUT_CHANNEL_LAST
                                                 ? hip_plugin_bn_config::CHANNEL_SLICE_SIZE
                                                 : LESS;

            // NOLINTNEXTLINE(readability-static-accessed-through-instance)
            StaticUnrollCount<unsigned int, 0, K_LIMIT, hip_plugin_bn_config::LAUNCH_DIM.GRP0, 2>{
                [&](unsigned int k) {
                    if(k + lid < K_LIMIT)
                    {
                        batchIdx = (k + lid) / hip_plugin_bn_config::SPATIAL_SIZE;
                        lidx = (k + lid) - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                        if constexpr(hip_plugin_config::LAYOUT_CHANNEL_LAST)
                        {
                            index = batchIdx * hip_plugin_bn_config::BATCH_SIZE
                                    + lidx * hip_plugin_bn_config::CHANNELS + grpid;
                        }
                        else
                        {
                            index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId
                                    + lidx;
                        }

                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::NEURON_OP>(
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
            ComputeType xhat[MAX_READ]; // NOLINT(modernize-avoid-c-arrays)

            StaticUnrollCount<unsigned int, 0, LESSOUT, CHUNK, 2>{[&](unsigned int k) {
                if(k + (MAX_READ * lid) < LESSOUT)
                {
                    for(unsigned int j = 0; j < MAX_READ; ++j)
                    {
                        const unsigned int l = k + (MAX_READ * lid) + j;
                        batchIdx = l / hip_plugin_bn_config::SPATIAL_SIZE;
                        lidx = l - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                        index
                            = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;
                        xhat[j] = (hip_kernel_provider::cast<ComputeType>(in[index]) - mean)
                                  * invVariance;
                    }

                    __syncthreads();

                    for(unsigned int j = 0; j < MAX_READ; ++j) // This part takes 0.405
                    {
                        const unsigned int l = k + (MAX_READ * lid) + j;
                        batchIdx = l / hip_plugin_bn_config::SPATIAL_SIZE;
                        lidx = l - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                        index
                            = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;
                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::NEURON_OP>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }};

            if constexpr(REMOUT > 0u)
            {
                const unsigned int remkeyout = (MAX_READ * lid) + LESSOUT;
                for(unsigned int j = 0; j < MAX_READ; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    batchIdx = l / hip_plugin_bn_config::SPATIAL_SIZE;
                    lidx = l - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                    index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;
                    // TODO: comparing different types
                    const auto xin = index < hip_plugin_bn_config::TENSOR_SIZE
                                         ? hip_kernel_provider::cast<ComputeType>(in[index])
                                         : hip_kernel_provider::cast<ComputeType>(0.0);
                    xhat[j] = (xin - hip_kernel_provider::cast<ComputeType>(mean))
                              * hip_kernel_provider::cast<ComputeType>(invVariance);
                }

                __syncthreads();
                for(unsigned int j = 0; j < MAX_READ; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    batchIdx = l / hip_plugin_bn_config::SPATIAL_SIZE;
                    lidx = l - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
                    index = batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + lidx;

                    if(index < hip_plugin_bn_config::TENSOR_SIZE)
                    {
                        out[index] = hip_kernel_provider::cast<OutputType>(
                            hip_kernel_provider::applyActivation<ComputeType,
                                                                 hip_plugin_config::NEURON_OP>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }
        }
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
        auto inhat = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pvscale = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pvbias = hip_kernel_provider::cast<ComputeType>(0.0);
        auto xin = hip_kernel_provider::cast<ComputeType>(0.0);

        __shared__ ComputeType s_lcl_scale;
        __shared__ ComputeType s_lcl_bias;

        unsigned int index = 0;
        unsigned int lid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        unsigned int channelIdx = grpid * hip_plugin_bn_config::SPATIAL_SIZE;

        // Unused if hip_plugin_bn_config::BATCHES >= hip_plugin_bn_config::MAX_BATCHES
        InputType minibatch[hip_plugin_bn_config::BATCHES]; // NOLINT(modernize-avoid-c-arrays)

        if(lid == 0)
        {
            s_lcl_scale = scale[grpid];
            s_lcl_bias = bias[grpid];
        }

        if(lid < hip_plugin_bn_config::SPATIAL_SIZE)
        {
            StaticUnrollCount<unsigned int, 0, hip_plugin_bn_config::BATCHES, 1, 2>{
                [&](unsigned int batch) {
                    index = batch * hip_plugin_bn_config::BATCH_SIZE + channelIdx + lid;
                    xin = hip_kernel_provider::cast<ComputeType>(in[index]);
                    mean += xin;
                    variance = fma(xin, xin, variance);

                    if constexpr(hip_plugin_bn_config::BATCHES < hip_plugin_bn_config::MAX_BATCHES)
                    {
                        minibatch[batch] = in[index];
                    }
                }};
        }
        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ ComputeType s_lcl_data_x[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        __shared__ ComputeType s_lcl_data_y[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2(
                static_cast<ComputeType&>(mean),
                static_cast<ComputeType&>(variance),
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        variance = fmax(fma(-mean, mean, variance), hip_kernel_provider::cast<MeanVarType>(0.0));
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<ComputeType>(epsilon));

        if(lid < hip_plugin_bn_config::SPATIAL_SIZE)
        {
            pvscale = s_lcl_scale;
            pvbias = s_lcl_bias;
            StaticUnrollCount<unsigned int, 0, hip_plugin_bn_config::BATCHES, 1, 2>{
                [&](unsigned int batch) { // apply normalization
                    index = batch * hip_plugin_bn_config::BATCH_SIZE + channelIdx + lid;
                    if constexpr(hip_plugin_bn_config::BATCHES < hip_plugin_bn_config::MAX_BATCHES)
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
                                                             hip_plugin_config::NEURON_OP>(
                            fma(pvscale, inhat, pvbias), alpha, beta));
                }}; // end for

        } // end if
    }
};

// these are the kernels for HIP_PLUGIN_BN_VARIANT == 2
#if(HIP_PLUGIN_BN_VARIANT == 2)

struct BatchNormFwdTrainSpatialImplVar2
{
    static constexpr unsigned int NGRPS
        = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
          LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
              .GRP1; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int NGRPS2
        = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
          LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
              .GRP2; // NOLINT(readability-static-accessed-through-instance)

    static __forceinline__ __device__ void norm(const InputType* __restrict__ in,
                                                OutputType* __restrict__ out,
                                                const ScaleBiasType* scale,
                                                const ScaleBiasType* bias,
                                                ComputeType alpha,
                                                ComputeType beta)
    {
        const unsigned int xstride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
        const unsigned int ystride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

        const unsigned int xgrpId
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpId
            = blockIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpId
            = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xgrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP0; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP1; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP2; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xlid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ylid
            = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zlid
            = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xgid = xgrpId * xgrpSize + xlid;
        const unsigned int ygid = ygrpId * ygrpSize + ylid;
        const unsigned int zgid = zgrpId * zgrpSize + zlid;

        ComputeChannelVecType mean;
        ComputeChannelVecType invVariance;
        ComputeChannelVecType pvtScale;
        ComputeChannelVecType pvtBias;

        __shared__ ComputeChannelVecType s_lcl_bias // NOLINT(modernize-avoid-c-arrays)
            [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                 .GRP0]; // NOLINT(modernize-avoid-c-arrays, readability-static-accessed-through-instance)
        __shared__ ComputeChannelVecType s_lcl_scale // NOLINT(modernize-avoid-c-arrays)
            [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                 .GRP0]; // NOLINT(modernize-avoid-c-arrays, readability-static-accessed-through-instance)
        __shared__ ComputeChannelVecType s_lcl_mean // NOLINT(modernize-avoid-c-arrays)
            [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                 .GRP0]; // NOLINT(modernize-avoid-c-arrays, readability-static-accessed-through-instance)
        __shared__ ComputeChannelVecType s_lcl_ivar // NOLINT(modernize-avoid-c-arrays)
            [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                 .GRP0]; // NOLINT(modernize-avoid-c-arrays, readability-static-accessed-through-instance)

        if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
        {
            return;
        }

        // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        if(ylid == 0 && zlid == 0)
        {
            s_lcl_scale[xlid] = *reinterpret_cast<const ComputeChannelVecType*>(
                scale // NOLINT(bugprone-implicit-widening-of-multiplication-result)
                + xgid
                      * hip_plugin_bn_config::
                          VEC_SIZE_X); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
            s_lcl_bias[xlid] = *reinterpret_cast<const ComputeChannelVecType*>(
                bias // NOLINT(bugprone-implicit-widening-of-multiplication-result)
                + xgid
                      * hip_plugin_bn_config::
                          VEC_SIZE_X); // NOLINT(bugprone-implicit-widening-of-multiplication-result)
            s_lcl_mean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType,
                                                                             OutputChannelVecType>(
                reinterpret_cast<const OutputChannelVecType*>(out),
                0,
                zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
                ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
            s_lcl_ivar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType,
                                                                             OutputChannelVecType>(
                reinterpret_cast<const OutputChannelVecType*>(out),
                1,
                zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
                ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
        }
        __syncthreads();

        if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::SPATIAL_SIZE
           && zgid < hip_plugin_bn_config::BATCHES)
        {
            mean = s_lcl_mean[xlid];
            invVariance = s_lcl_ivar[xlid];
            pvtScale = s_lcl_scale[xlid];
            pvtBias = s_lcl_bias[xlid];
            const unsigned int indexBase
                = zgid * hip_plugin_bn_config::BATCH_ELEMENTS * hip_plugin_bn_config::BATCH_SIZE
                  + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
                  + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;

            // The original OpenCL kernel unrolled the loop only when this condition was met.
            // Using this UNROLL_HINT and the StaticUnrollCount struct replicates this.
            constexpr unsigned int UNROLL_HINT
                = hip_plugin_bn_config::SPATIAL_SIZE > hip_plugin_bn_config::LOOP_UNROLL_MAX_SPATIAL
                      ? 1
                      : 2;

            // This method of unrolling is used as opposed to
            // the struct StaticUnrollCount due to a bug where
            // the kernel writes a tensor for the output twice bigger
            // than it should be, and it repeats values equal to the number of channels
            // in the examined test cases that were failing.

#if(HIP_PLUGIN_BN_SPATIAL_SIZE > HIP_PLUGIN_BN_LOOP_UNROLL_MAX_SPATIAL)
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
#else
#pragma unroll(2)
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
#endif
            {
                const unsigned int index = indexBase + batch * hip_plugin_bn_config::BATCH_SIZE;
                auto value = hip_kernel_provider::cast<ComputeVecType>(
                    *reinterpret_cast<const InputVecType*>(in + index));
                // This is a float4 when not vectorizing; for FPMIX ComputeVecType is float, FpType is __half
                ComputeVecType inhat
                    = (hip_kernel_provider::cast<ComputeVecType>(value) - mean) * invVariance;
                inhat
                    = hip_kernel_provider::fma(hip_kernel_provider::cast<ComputeVecType>(pvtScale),
                                               inhat,
                                               hip_kernel_provider::cast<ComputeVecType>(pvtBias));

                value = hip_kernel_provider::applyActivation<ComputeVecType,
                                                             hip_plugin_config::NEURON_OP>(
                    inhat,
                    hip_kernel_provider::cast<ComputeVecType>(alpha),
                    hip_kernel_provider::cast<ComputeVecType>(beta));

                *reinterpret_cast<OutputVecType*>(out + index)
                    = hip_kernel_provider::cast<OutputVecType>(value);
            } // end for(batch)
        } // end if(inImgIndex)
    } // end spatial norm

    static __forceinline__ __device__ void finalMeanVariance(
        OutputType* __restrict__ meanvarbuff, // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
        double epsilon,
        const unsigned int& /* xgid */,
        const unsigned int& /* ygid */,
        const unsigned int& /* zgid */,
        unsigned int& commitID,
        MeanVarChannelVecType& mean,
        MeanVarChannelVecType& variance,
        MeanVarChannelVecType& invVariance)
    {
        variance = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);
        invVariance = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);
        mean = hip_kernel_provider::cast<MeanVarChannelVecType>(0.0);

        const unsigned int xgrpId
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)

        // These values (?grp_size) cannot be substituted with hip_plugin_bn_config::LAUNCH_DIM.grp? because
        // the dimensions of the blocks for this kernel may be different from the other
        // kernels that take part in the operation. Given that the launch dimensions are
        // rounded up, blockDim would return a consistent value.

        const unsigned int xgrpSize
            = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpSize
            = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpSize
            = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xlid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ylid
            = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zlid
            = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xstride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
        const unsigned int ystride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

        commitID = 0;

        for(unsigned int zoffset = zlid;
            zoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                      LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                          .GRP2; // NOLINT(readability-static-accessed-through-instance)
            zoffset += zgrpSize)
        {
            for(unsigned int yoffset = ylid;
                yoffset
                < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                      .GRP1; // NOLINT(readability-static-accessed-through-instance)
                yoffset += ygrpSize)
            {
                mean += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    0,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP2 // NOLINT(readability-static-accessed-through-instance)
                        * zoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::BATCH_ELEMENTS,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP1 // NOLINT(readability-static-accessed-through-instance)
                        * yoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrpSize,
                    xgrpId,
                    xlid,
                    xstride);
                variance += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    1,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP2 // NOLINT(readability-static-accessed-through-instance)
                        * zoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::BATCH_ELEMENTS,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP1 // NOLINT(readability-static-accessed-through-instance)
                        * yoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrpSize,
                    xgrpId,
                    xlid,
                    xstride);
            }
        }

        if constexpr(
            !hip_plugin_bn_config::USE_AMDGCN
            || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                   LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                       .GRP0 // NOLINT(readability-static-accessed-through-instance)
                   > 1 // NOLINT(readability-static-accessed-through-instance)
            || (hip_plugin_bn_config::LDS_GCN_SIZE == 1) || hip_plugin_bn_config::VEC_SIZE_X > 1
            || (hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                    .BOUNDS // NOLINT(readability-static-accessed-through-instance)
                < 64)) // NOLINT(readability-static-accessed-through-instance)
        {
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclData
                    [2
                     * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                       LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                           .BOUNDS]; // NOLINT(readability-static-accessed-through-instance)

            hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
                static_cast<ComputeChannelVecType&>(mean),
                static_cast<ComputeChannelVecType&>(variance),
                hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE,
                s_lclData,
                xgrpSize,
                xlid,
                ylid + zlid * ygrpSize,
                ygrpSize * zgrpSize);
        }
        else
        {
            // C++17 idiomatic: ensure array size is never zero using constexpr ternary
            constexpr auto LDS_GCN_ARRAY_SIZE
                = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                          LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                              .BOUNDS // NOLINT(readability-static-accessed-through-instance)
                          >= 64 // NOLINT(readability-static-accessed-through-instance)
                      ? hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                            LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                                .BOUNDS // NOLINT(readability-static-accessed-through-instance)
                            / 64 // NOLINT(readability-static-accessed-through-instance)
                      : 1;

            commitID = 64;
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclDataX[LDS_GCN_ARRAY_SIZE]; // NOLINT(modernize-avoid-c-arrays)
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclDataY[LDS_GCN_ARRAY_SIZE]; // NOLINT(modernize-avoid-c-arrays)
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                static_cast<ComputeChannelVecType&>(mean),
                static_cast<ComputeChannelVecType&>(variance),
                hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE,
                s_lclDataX,
                s_lclDataY,
                ylid + zlid * ygrpSize);
        }

        variance = hip_kernel_provider::fma(-mean, mean, variance);
        variance
            = hip_kernel_provider::max(hip_kernel_provider::cast<ComputeChannelVecType>(variance),
                                       hip_kernel_provider::cast<ComputeChannelVecType>(0.0));
        invVariance = hip_kernel_provider::rsqrt(
            hip_kernel_provider::cast<ComputeChannelVecType>(variance)
            + hip_kernel_provider::cast<ComputeChannelVecType>(epsilon));

        for(unsigned int zoffset = zlid;
            zoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                      LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                          .GRP2; // NOLINT(readability-static-accessed-through-instance)
            zoffset += zgrpSize)
        {
            for(unsigned int yoffset = ylid;
                yoffset
                < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                      .GRP1; // NOLINT(readability-static-accessed-through-instance)
                yoffset += ygrpSize)
            {
                storeToStash(
                    mean,
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    0,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP2 // NOLINT(readability-static-accessed-through-instance)
                        * zoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::BATCH_ELEMENTS,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP1 // NOLINT(readability-static-accessed-through-instance)
                        * yoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrpSize,
                    xgrpId,
                    xlid,
                    xstride);
                storeToStash(
                    invVariance,
                    reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                    1,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP2 // NOLINT(readability-static-accessed-through-instance)
                        * zoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::BATCH_ELEMENTS,
                    hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP1 // NOLINT(readability-static-accessed-through-instance)
                        * yoffset // NOLINT(readability-static-accessed-through-instance)
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrpSize,
                    xgrpId,
                    xlid,
                    xstride);
            }
        }
    }

    static __forceinline__ __device__ void meanVariance(
        const InputType* __restrict__ in,
        OutputType* __restrict__ meanvarbuff) // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
    {
        const unsigned int xgrpId
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpId
            = blockIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpId
            = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xgrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP0; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP1; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpSize
            = hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
              LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                  .GRP2; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xlid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ylid
            = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zlid
            = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xgid = xgrpId * xgrpSize + xlid;
        const unsigned int ygid = ygrpId * ygrpSize + ylid;
        const unsigned int zgid = zgrpId * zgrpSize + zlid;

        const unsigned int xstride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
        const unsigned int ystride
            = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

        unsigned int index;

        auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
        auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

        if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
        {
            return;
        }

        if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::SPATIAL_SIZE
           && zgid < hip_plugin_bn_config::BATCHES)
        {
            const unsigned int indexBase
                = zgid * hip_plugin_bn_config::BATCH_ELEMENTS * hip_plugin_bn_config::BATCH_SIZE
                  + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
                  + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
            {
                index = indexBase + batch * hip_plugin_bn_config::BATCH_SIZE;
                auto read4 = hip_kernel_provider::cast<ComputeVecType>(
                    *reinterpret_cast<const InputVecType*>(in + index));

                hip_kernel_provider::batchnorm::accumulate(mean, read4);
                hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
            }
        }

        if constexpr(
            !hip_plugin_bn_config::USE_AMDGCN
            || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                   LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                       .GRP0 // NOLINT(readability-static-accessed-through-instance)
                   > 1 // NOLINT(readability-static-accessed-through-instance)
            || (hip_plugin_bn_config::LDS_GCN_SIZE == 1) || hip_plugin_bn_config::VEC_SIZE_X > 1)
        {
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclData[2 * hip_plugin_bn_config::LDS_SIZE]; // NOLINT(modernize-avoid-c-arrays)
            hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
                mean,
                variance,
                hip_kernel_provider::cast<ComputeType>(1.0),
                s_lclData,
                xgrpSize,
                xlid,
                ylid + zlid * ygrpSize,
                ygrpSize * zgrpSize);
        }
        else
        {
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclDataX[hip_plugin_bn_config::LDS_GCN_SIZE]; // NOLINT(modernize-avoid-c-arrays)
            __shared__ ComputeChannelVecType // NOLINT(modernize-avoid-c-arrays)
                s_lclDataY[hip_plugin_bn_config::LDS_GCN_SIZE]; // NOLINT(modernize-avoid-c-arrays)
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                mean,
                variance,
                hip_kernel_provider::cast<ComputeType>(1.0),
                s_lclDataX,
                s_lclDataY,
                ylid + zlid * ygrpSize);
        }

        if(ylid == 0 && zlid == 0)
        {
            storeToStash(mean,
                         reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                         0,
                         zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
                         ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
                         ystride / hip_plugin_bn_config::VEC_SIZE_X,
                         xgrpSize,
                         xgrpId,
                         xlid,
                         xstride);
            storeToStash(variance,
                         reinterpret_cast<OutputChannelVecType*>(meanvarbuff),
                         1,
                         zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
                         ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
                         ystride / hip_plugin_bn_config::VEC_SIZE_X,
                         xgrpSize,
                         xgrpId,
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
extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormFwdTrainSpatial(const InputType* __restrict in,
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
    constexpr auto VARIANT = hip_plugin_bn_config::VARIANT;

    using ForwardTrainSpatialImpl
        = hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImpl<VARIANT>;

    MeanVarType mean;
    MeanVarType variance;
    MeanVarType invVariance;
    const unsigned int lid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int grpid = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)

    ForwardTrainSpatialImpl{}(
        in, out, scale, bias, epsilon, mean, variance, invVariance, alpha, beta);

    if(lid == 0)
    {
// TODO: this should also be removed, but using constexpr can lead to compile errors
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<MeanVarChannelVecType>;
        const StashUpdater updater(static_cast<MeanVarChannelVecType>(mean),
                                   static_cast<MeanVarChannelVecType>(variance),
                                   static_cast<MeanVarChannelVecType>(expAvgFactor));

        hip_kernel_provider::batchnorm::runningStash<MeanVarChannelVecType,
                                                     ComputeChannelVecType,
                                                     StashUpdater>(prevResultRunningMean,
                                                                   prevResultRunningVariance,
                                                                   nextResultRunningMean,
                                                                   nextResultRunningVariance,
                                                                   updater,
                                                                   grpid);
#endif
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::savedStash<MeanVarChannelVecType, ComputeChannelVecType>(
            resultSaveMean,
            resultSaveInvVariance,
            static_cast<MeanVarChannelVecType>(mean),
            static_cast<MeanVarChannelVecType>(invVariance),
            grpid);
#endif
    }
}

#else

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormFwdTrainSpatialNorm(const InputType* __restrict__ in,
                                 OutputType* __restrict__ out,
                                 const ScaleBiasType* scale,
                                 const ScaleBiasType* bias,
                                 ComputeType alpha,
                                 ComputeType beta)
{
    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2::norm(
        in, out, scale, bias, alpha, beta);
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM_FINAL.BOUNDS)
    batchNormFwdTrainSpatialFinalMeanVariance(
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

    const unsigned int xgid
        = blockIdx.x * blockDim.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygid
        = blockIdx.y * blockDim.y // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgid
        = blockIdx.z * blockDim.z // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    unsigned int commitID;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2::finalMeanVariance(
        meanvarbuff, epsilon, xgid, ygid, zgid, commitID, mean, variance, invVariance);
    // Save mean and calculate and save running mean
    if(ygid == commitID && zgid == 0)
    {
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<MeanVarChannelVecType>;
        StashUpdater const updater(hip_kernel_provider::cast<MeanVarChannelVecType>(mean),
                                   hip_kernel_provider::cast<MeanVarChannelVecType>(variance),
                                   hip_kernel_provider::cast<MeanVarChannelVecType>(expAvgFactor));

        hip_kernel_provider::batchnorm::
            runningStash<ComputeChannelVecType, MeanVarChannelVecType, StashUpdater>(
                reinterpret_cast<const MeanVarChannelVecType*>(prevResultRunningMean),
                reinterpret_cast<const MeanVarChannelVecType*>(prevResultRunningVariance),
                reinterpret_cast<MeanVarChannelVecType*>(nextResultRunningMean),
                reinterpret_cast<MeanVarChannelVecType*>(nextResultRunningVariance),
                updater,
                xgid);
#endif

#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::savedStash<MeanVarChannelVecType, ComputeChannelVecType>(
            reinterpret_cast<ComputeChannelVecType*>(resultSaveMean),
            reinterpret_cast<ComputeChannelVecType*>(resultSaveInvVariance),
            mean,
            invVariance,
            xgid);
#endif
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormFwdTrainSpatialMeanVariance(
        const InputType* __restrict__ in,
        OutputType* __restrict__ meanvarbuff) // meanvarbuff uses the y buffer as a workspace, so OutputType instead of MeanVarType
{
    hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2::meanVariance(
        in, // NOLINT(readability-static-accessed-through-instance)
        meanvarbuff);
}

#endif
