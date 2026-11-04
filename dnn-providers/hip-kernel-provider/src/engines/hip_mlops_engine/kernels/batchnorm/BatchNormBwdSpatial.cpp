// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormFunctions.hpp"
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

template <typename ComputeWriteVecType>
__forceinline__ __device__ __host__ auto batchBwdNormalization(const ComputeWriteVecType value,
                                                               const ComputeWriteVecType xhat,
                                                               const ComputeType dbias,
                                                               const ComputeType dscale,
                                                               const ComputeType pscale,
                                                               const ComputeType invVariance)
{
    ComputeWriteVecType tmp1 = hip_kernel_provider::fma(
        hip_kernel_provider::cast<ComputeWriteVecType>(hip_plugin_bn_config::CHANNEL_SLICE_SIZE),
        value,
        hip_kernel_provider::cast<ComputeWriteVecType>(-dbias));
    ComputeWriteVecType tmp2 = -xhat * hip_kernel_provider::cast<ComputeWriteVecType>(dscale);
    const ComputeType tmp3 = pscale * invVariance * hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE;
    return hip_kernel_provider::cast<ComputeWriteVecType>(tmp3) * (tmp2 + tmp1);
}

// Specialized version for HIP_PLUGIN_BN_VARIANT 2
template <typename ComputeWriteVecType>
__forceinline__ __device__ __host__ auto
    batchBwdNormalizationVar2(const ComputeWriteVecType value,
                              const ComputeWriteVecType xhat,
                              const ComputeWriteVecType dbias,
                              const ComputeWriteVecType dscale,
                              const ComputeWriteVecType pscale,
                              const ComputeWriteVecType invVariance)
{
    ComputeWriteVecType tmp1 = hip_kernel_provider::fma(
        hip_kernel_provider::cast<ComputeWriteVecType>(hip_plugin_bn_config::CHANNEL_SLICE_SIZE),
        value,
        -dbias);
    ComputeWriteVecType tmp2 = -xhat * dscale;
    ComputeWriteVecType tmp3 = pscale * invVariance * hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE;
    return tmp3 * (tmp2 + tmp1);
}

template <typename ComputeReadVecType>
__forceinline__ __host__ __device__ ComputeReadVecType
    vectorizedBwdActivationOp(ComputeReadVecType const& dy,
                              ComputeReadVecType const& xnorm,
                              ComputeType const& scale,
                              ComputeType const& bias,
                              ComputeType const& alpha,
                              ComputeType const& beta)
{
    auto constexpr SIZE = hip_kernel_provider::MappedVectorInfo<ComputeReadVecType>::SIZE;
    if constexpr(SIZE == 4)
    {
        ComputeReadVecType out;
        out.x = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        out.z = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.z, xnorm.z, scale, bias, alpha, beta);
        out.w = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.w, xnorm.w, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(SIZE == 2)
    {
        ComputeReadVecType out;
        out.x = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                                hip_plugin_config::NEURON_OP>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(SIZE == 1)
    {
        return hip_kernel_provider::batchnorm::bwdActivationOp<ComputeType,
                                                               hip_plugin_config::NEURON_OP>(
            dy, xnorm, scale, bias, alpha, beta);
    }
    else
    {
        static_assert(false, "Unsupported miopen vector operation.");
    }
}

} // namespace

namespace hip_kernel_provider::batchnorm
{

template <int BnVariant>
struct BatchNormBwdSpatialImpl
{
    static_assert(false, "This variant is not supported.");
};

template <>
struct BatchNormBwdSpatialImpl<0>
{
    static constexpr unsigned int SEGTMP1
        = hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
              .GRP0 // NOLINT(readability-static-accessed-through-instance)
          / hip_plugin_bn_config::
              SPATIAL_SIZE; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int SEGTMP2 = SEGTMP1 == 0 ? 1 : SEGTMP1;
    static constexpr unsigned int SEGTMP = hip_plugin_bn_config::SPATIAL_SIZE * SEGTMP2;
    static constexpr unsigned int SEGMENT = SEGTMP > hip_plugin_bn_config::CHANNEL_SLICE_SIZE
                                                ? hip_plugin_bn_config::CHANNEL_SLICE_SIZE
                                                : SEGTMP;
    static constexpr unsigned int BATCH_LOOP
        = (hip_plugin_bn_config::CHANNEL_SLICE_SIZE + SEGMENT - 1) / SEGMENT;
    static constexpr unsigned int SEGI_SPATIAL = SEGMENT / hip_plugin_bn_config::SPATIAL_SIZE;
    static_assert(BATCH_LOOP > 0);
    static constexpr unsigned int BATCH_LOOPM = BATCH_LOOP - 1;
    static constexpr unsigned int SN_SPATIAL = BATCH_LOOPM * SEGI_SPATIAL;

    __forceinline__ __device__ void operator()(const InputType* __restrict xIn,
                                               const OutputType* __restrict dyIn,
                                               InputType* __restrict dxOut,
                                               const ScaleBiasType* __restrict bnScale,
                                               const ScaleBiasType* __restrict bnBias,
                                               ScaleBiasType* __restrict dscale,
                                               ScaleBiasType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                               double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                                               const MeanVarType* savedMean,
                                               const MeanVarType* savedInvVariance,
#endif
                                               ComputeType alpha,
                                               ComputeType beta)
    {
#if(HIP_PLUGIN_BN_USESAVED == 0)
        auto variance = hip_kernel_provider::cast<MeanVarType>(0.0);
#endif
        auto mean = hip_kernel_provider::cast<ComputeType>(0.0);
        auto invVariance = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pscale = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pbias = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pdscale = hip_kernel_provider::cast<ComputeType>(0.0);
        auto pdbias = hip_kernel_provider::cast<ComputeType>(0.0);

        ComputeType batchValues[BATCH_LOOP]; // NOLINT(modernize-avoid-c-arrays)
        ComputeType dyValues[BATCH_LOOP]; // NOLINT(modernize-avoid-c-arrays)

        __shared__ ComputeType s_lbnScale;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        __shared__ ComputeType s_lbnBias;
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        __shared__ ComputeType s_lmean;
        __shared__ ComputeType s_lvar;
#endif
        unsigned int index = 0;
        const unsigned int lid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int channelSpatialId = grpid * hip_plugin_bn_config::SPATIAL_SIZE
                                              + (lid % hip_plugin_bn_config::SPATIAL_SIZE);
        const unsigned int lidiSpatial = lid / hip_plugin_bn_config::SPATIAL_SIZE;
        unsigned int batchId = 0;

        if(lid == 0)
        {
            s_lbnScale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
            s_lbnBias = bnBias[grpid];
#endif
        }

#if(HIP_PLUGIN_BN_USESAVED == 1)
        if(lid == 0)
        {
            s_lmean = savedMean[grpid];
            s_lvar = savedInvVariance[grpid];
        }
        __syncthreads();
        mean = s_lmean;
        invVariance = s_lvar;
#else // recalc mean and variance below \
    // == RECALC MEAN AND VARIANCE ===========
        if(lid < SEGMENT)
        {
            for(unsigned int batch = 0; batch < BATCH_LOOPM; ++batch)
            {
                batchId = batch * SEGI_SPATIAL + lidiSpatial;
                index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
                batchValues[batch] = hip_kernel_provider::cast<ComputeType>(xIn[index]);
                mean += batchValues[batch];
                variance = fma(batchValues[batch], batchValues[batch], variance);
            }
            batchId = SN_SPATIAL + lidiSpatial;
            index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
            batchValues[BATCH_LOOPM] = index < hip_plugin_bn_config::TENSOR_SIZE
                                           ? hip_kernel_provider::cast<ComputeType>(xIn[index])
                                           : 0;
            mean += batchValues[BATCH_LOOPM];
            variance = fma(batchValues[BATCH_LOOPM], batchValues[BATCH_LOOPM], variance);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                lid);

        variance = fma(-mean, mean, variance);
        // NOLINTNEXTLINE(readability-use-std-min-max): <algorithm> is unavailable in hipRTC
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon); // NOLINT(bugprone-narrowing-conversions)

#endif // end -- Recalc mean and variance \
    //-------------------------------------------
        pscale = s_lbnScale;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = s_lbnBias;
#endif

        //==== CALC DB and DS =========================================
        if(lid < SEGMENT)
        {
            for(unsigned int batch = 0; batch < BATCH_LOOPM; ++batch)
            {
                batchId = batch * SEGI_SPATIAL + lidiSpatial;
                index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
                dyValues[batch] = hip_kernel_provider::cast<ComputeType>(dyIn[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                batchValues[batch]
                    = (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance;
#else
                batchValues[batch] = (batchValues[batch] - mean) * invVariance;
#endif
                dyValues[batch] = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                    dyValues[batch], batchValues[batch], pscale, pbias, alpha, beta);
                // batchValues is now xhat
                pdbias += dyValues[batch];
                pdscale = fma(batchValues[batch], dyValues[batch], pdscale);
            }
            batchId = SN_SPATIAL + lidiSpatial;
            index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
            dyValues[BATCH_LOOPM] = index < hip_plugin_bn_config::TENSOR_SIZE
                                        ? hip_kernel_provider::cast<ComputeType>(dyIn[index])
                                        : hip_kernel_provider::cast<ComputeType>(0.0);

#if(HIP_PLUGIN_BN_USESAVED == 1)
            batchValues[BATCH_LOOPM]
                = index < hip_plugin_bn_config::TENSOR_SIZE
                      ? (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance
                      : 0;
#else
            batchValues[BATCH_LOOPM] = (batchValues[BATCH_LOOPM] - mean) * invVariance;
#endif
            dyValues[BATCH_LOOPM] = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                dyValues[BATCH_LOOPM], batchValues[BATCH_LOOPM], pscale, pbias, alpha, beta);
            // batchValues is now xhat
            pdbias += dyValues[BATCH_LOOPM];
            pdscale = fma(batchValues[BATCH_LOOPM], dyValues[BATCH_LOOPM], pdscale);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        if(lid < SEGMENT)
        {
            //==== CALC NORM =======================
            for(unsigned int batch = 0; batch < BATCH_LOOPM; batch++)
            {
                batchId = batch * SEGI_SPATIAL + lidiSpatial;
                index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
                dxOut[index] = hip_kernel_provider::cast<OutputType>(batchBwdNormalization(
                    dyValues[batch], batchValues[batch], pdbias, pdscale, pscale, invVariance));
            } // end for
            batchId = SN_SPATIAL + lidiSpatial;
            index = batchId * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId;
            if(index < hip_plugin_bn_config::TENSOR_SIZE)
            {
                dxOut[index] = hip_kernel_provider::cast<OutputType>(
                    batchBwdNormalization(dyValues[BATCH_LOOPM],
                                          batchValues[BATCH_LOOPM],
                                          pdbias,
                                          pdscale,
                                          pscale,
                                          invVariance));
            }
        }
        if(lid == 0)
        {
            dbias[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdbias);
            dscale[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdscale);
        }
    }
};

template <>
struct BatchNormBwdSpatialImpl<1>
{
    static constexpr unsigned int READ_SIZE = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : 4;
    static constexpr unsigned int WRITE_SIZE = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : 2;

    using InputReadVecType = typename MappedVectorType<InputType, READ_SIZE>::type;
    using OutputReadVecType = typename MappedVectorType<OutputType, READ_SIZE>::type;
    using ComputeReadVecType = typename MappedVectorType<ComputeType, READ_SIZE>::type;
    using InputWriteVecType = typename MappedVectorType<InputType, WRITE_SIZE>::type;
    using OutputWriteVecType = typename MappedVectorType<OutputType, WRITE_SIZE>::type;
    using ComputeWriteVecType = typename MappedVectorType<ComputeType, WRITE_SIZE>::type;

    static constexpr unsigned int READ_BLOCK = 1;
    static constexpr unsigned int GRPRD
        = hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
              .GRP0 // NOLINT(readability-static-accessed-through-instance)
          * READ_BLOCK // NOLINT(readability-static-accessed-through-instance)
          * READ_SIZE; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int REM4
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - (hip_plugin_bn_config::CHANNEL_SLICE_SIZE / GRPRD) * GRPRD;
    static constexpr unsigned int LESS4 = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REM4;
    static constexpr unsigned int REM
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - (hip_plugin_bn_config::CHANNEL_SLICE_SIZE
             / hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
               LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                   .GRP0) // NOLINT(readability-static-accessed-through-instance)
                * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                      .GRP0; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int LESS = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REM;
    static constexpr unsigned int CHUNK
        = WRITE_SIZE
          * hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                .GRP0; // NOLINT(readability-static-accessed-through-instance)
    static constexpr unsigned int REMOUT
        = hip_plugin_bn_config::CHANNEL_SLICE_SIZE
          - ((hip_plugin_bn_config::CHANNEL_SLICE_SIZE / CHUNK) * CHUNK);
    static constexpr unsigned int LESSOUT = hip_plugin_bn_config::CHANNEL_SLICE_SIZE - REMOUT;

    __forceinline__ __device__ unsigned int
        getTensorIndex( // NOLINT(readability-convert-member-functions-to-static)
            unsigned int loopIndex) // NOLINT(readability-convert-member-functions-to-static)
    {
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int channelSpatialId = grpid * hip_plugin_bn_config::SPATIAL_SIZE;
        const unsigned int batchIdx = loopIndex / hip_plugin_bn_config::SPATIAL_SIZE;
        const unsigned int spatialIdx = loopIndex - (batchIdx * hip_plugin_bn_config::SPATIAL_SIZE);
        return hip_plugin_config::LAYOUT_CHANNEL_LAST
                   ? batchIdx * hip_plugin_bn_config::BATCH_SIZE
                         + spatialIdx * hip_plugin_bn_config::CHANNELS + grpid
                   : batchIdx * hip_plugin_bn_config::BATCH_SIZE + channelSpatialId + spatialIdx;
    }

    __forceinline__ __device__ void operator()(const InputType* __restrict xIn,
                                               const OutputType* __restrict dyIn,
                                               InputType* __restrict dxOut,
                                               const ScaleBiasType* __restrict bnScale,
                                               const ScaleBiasType* __restrict bnBias,
                                               ScaleBiasType* __restrict dscale,
                                               ScaleBiasType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                               double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                                               const MeanVarType* savedMean,
                                               const MeanVarType* savedInvVariance,
#endif
                                               ComputeType alpha,
                                               ComputeType beta)
    {
        ComputeType mean = 0;
        ComputeType invVariance = 0;
        ComputeType pscale = 0;
        ComputeType pbias = 0;
        ComputeType pdbias = 0;
        ComputeType pdscale = 0;
        const ComputeType xhat = 0;

        unsigned int lid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int channelSpatialId = grpid * hip_plugin_bn_config::SPATIAL_SIZE;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 0)
        //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
        ComputeType variance = 0;
        if constexpr(!hip_plugin_config::LAYOUT_CHANNEL_LAST
                     && hip_plugin_bn_config::SPATIAL_SIZE >= 4096)
        {
            ComputeReadVecType read4;
            for(unsigned int k = lid << 2; k < LESS4; k += GRPRD)
            {
                read4 = hip_kernel_provider::cast<ComputeReadVecType>(
                    *reinterpret_cast<const InputReadVecType*>(xIn + getTensorIndex(k)));
                hip_kernel_provider::batchnorm::accumulate(mean, read4);
                hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
            }

            if constexpr(REM4 > 0)
            {
                if(lid < REM4)
                {
                    const unsigned int index = getTensorIndex((lid << 2) + LESS4);
                    if(index + READ_SIZE - 1 < hip_plugin_bn_config::TENSOR_SIZE)
                    {
                        read4 = hip_kernel_provider::cast<ComputeReadVecType>(
                            *reinterpret_cast<const InputReadVecType*>(xIn + index));
                        hip_kernel_provider::batchnorm::accumulate(mean, read4);
                        hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                    }
                }
            }
        }
        else
        {
            for(unsigned int k = lid; k < LESS;
                k += hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                     LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                         .GRP0) // NOLINT(readability-static-accessed-through-instance)
            {
                auto in = hip_kernel_provider::cast<ComputeType>(xIn[getTensorIndex(k)]);
                mean += in;
                variance = fma(in, in, variance);
            }
            if constexpr(REM > 0)
            {
                if(lid < REM)
                {
                    const unsigned int index = getTensorIndex(lid + LESS);
                    auto in = (index < hip_plugin_bn_config::TENSOR_SIZE)
                                  ? hip_kernel_provider::cast<ComputeType>(xIn[index])
                                  : hip_kernel_provider::cast<ComputeType>(0.0);
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                lid);

        // REDUCTION COMPLETE ---------------------------
        variance = fma(-mean, mean, variance);
        // NOLINTNEXTLINE(readability-use-std-min-max): <algorithm> is unavailable in hipRTC
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon); // NOLINT(bugprone-narrowing-conversions)

#else // HIP_PLUGIN_BN_USESAVED == 1
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#endif

        constexpr unsigned int READ_UNROLL_HINT
            = hip_plugin_bn_config::BATCHES > hip_plugin_bn_config::LOOP_UNROLL_MAX_BATCH ? 4 : 2;
        StaticUnrollCount<unsigned int, 0, LESS4, GRPRD, READ_UNROLL_HINT>{[&](unsigned int k) {
            const unsigned int l = k + (lid << 2 * (1 - hip_plugin_config::LAYOUT_CHANNEL_LAST));
            if(l < LESS4)
            {
                const unsigned int index = getTensorIndex(l);
                auto xread = *reinterpret_cast<const InputReadVecType*>(xIn + index);
                auto dyRead = *reinterpret_cast<const OutputReadVecType*>(dyIn + index);
                auto dyValue = hip_kernel_provider::cast<ComputeReadVecType>(dyRead);
                const ComputeReadVecType xhat
                    = (hip_kernel_provider::cast<ComputeReadVecType>(xread) - mean) * invVariance;

                dyValue = vectorizedBwdActivationOp<ComputeReadVecType>(
                    dyValue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::accumulate(pdbias, dyValue);
                hip_kernel_provider::batchnorm::accumulateMad(pdscale, xhat, dyValue);
            }
        }};

        if constexpr(REM4 > 0)
        {
            const unsigned int index = getTensorIndex(
                (lid << 2 * (1 - static_cast<int>(hip_plugin_config::LAYOUT_CHANNEL_LAST)))
                + LESS4);
            if(index + READ_SIZE - 1 < hip_plugin_bn_config::TENSOR_SIZE)
            {
                auto xread = *reinterpret_cast<const InputReadVecType*>(xIn + index);
                auto dyRead = *reinterpret_cast<const OutputReadVecType*>(dyIn + index);
                auto dyValue = hip_kernel_provider::cast<ComputeReadVecType>(dyRead);
                auto xhat
                    = (hip_kernel_provider::cast<ComputeReadVecType>(xread) - mean) * invVariance;

                dyValue = vectorizedBwdActivationOp<ComputeReadVecType>(
                    dyValue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::accumulate(pdbias, dyValue);
                hip_kernel_provider::batchnorm::accumulateMad(pdscale, xhat, dyValue);
            }
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        __syncthreads();

        if(lid == 0)
        {
            dbias[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdbias);
            dscale[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdscale);
        }

        constexpr unsigned int WRITE_UNROLL_HINT
            = hip_plugin_bn_config::BATCHES > hip_plugin_bn_config::LOOP_UNROLL_MAX_BATCH ? 2 : 1;
        StaticUnrollCount<unsigned int, 0, LESSOUT, CHUNK, WRITE_UNROLL_HINT>{[&](unsigned int k) {
            // Unrolling the loop requires forcing explicit data vectorization otherwise the
            // compiler will start splitting global loads into smaller chunks resulting in
            // significant slowdown.
            ComputeWriteVecType vals;
            const unsigned int l = k + (WRITE_SIZE * lid);
            const unsigned int index = getTensorIndex(l);
            if(l < LESSOUT)
            {
                auto xread = *reinterpret_cast<const InputWriteVecType*>(xIn + index);
                auto dyRead = *reinterpret_cast<const OutputWriteVecType*>(dyIn + index);
                auto value1 = hip_kernel_provider::cast<ComputeWriteVecType>(dyRead);
                const ComputeWriteVecType xhat1
                    = (hip_kernel_provider::cast<ComputeWriteVecType>(xread) - mean) * invVariance;

                value1 = vectorizedBwdActivationOp<ComputeWriteVecType>(
                    value1, xhat1, pscale, pbias, alpha, beta);

                vals = batchBwdNormalization(value1, xhat1, pdbias, pdscale, pscale, invVariance);
            }

            // Synchronization is not required for correctness but enhances performance.
            //
            // Loop is memory bound as it iterates across all the batches in the tensor,
            // and has memory access strides of CL size once all the elements in a single
            // sample have been processed, which may be large.
            //
            // `__syncthreads()` helps to coalesce memory accesses as each work-item accesses
            // adjacent elements to its neighbours on the same loop iteration, leading to contiguous
            // memory access across all the waves in a workgroup. By keeping all the waves on the
            // same loop iteration it prevents waves on different loop iterations from stalling
            // as they wait for memory.
            //
            // This can be seen by profiling the kernel with rocprofv3 and comparing the
            // `TCP_PENDING_STALL_CYCLES_sum` counter and also looking at a thread trace in
            // compute viewer and seeing the impact on occupancy.
            __syncthreads();

            if(l < LESSOUT)
            {
                *reinterpret_cast<InputWriteVecType*>(dxOut + index)
                    = hip_kernel_provider::cast<InputWriteVecType>(vals);
            }
        }};

        if constexpr(REMOUT > 0)
        {
            const unsigned int remkeyout = (WRITE_SIZE * lid) + LESSOUT;
            for(unsigned int j = 0; j < WRITE_SIZE; j++)
            {
                const unsigned int index = getTensorIndex(remkeyout + j);
                if(index < hip_plugin_bn_config::TENSOR_SIZE)
                {
                    auto value1 = hip_kernel_provider::cast<ComputeType>(dyIn[index]);
                    const ComputeType xhat
                        = (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance;

                    value1 = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                        value1, xhat, pscale, pbias, alpha, beta);

                    dxOut[index] = hip_kernel_provider::cast<InputType>(
                        batchBwdNormalization(value1, xhat, pdbias, pdscale, pscale, invVariance));
                }
            }
        }
    }
};

template <>
struct BatchNormBwdSpatialImpl<3>
{
    __forceinline__ __device__ void operator()(const InputType* __restrict xIn,
                                               const OutputType* __restrict dyIn,
                                               InputType* __restrict dxOut,
                                               const ScaleBiasType* __restrict bnScale,
                                               const ScaleBiasType* __restrict bnBias,
                                               ScaleBiasType* __restrict dscale,
                                               ScaleBiasType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                               double epsilon,
#else
                                               const MeanVarType* savedMean,
                                               const MeanVarType* savedInvVariance,
#endif
                                               ComputeType alpha,
                                               ComputeType beta)
    {
        ComputeType mean = 0;
#if(HIP_PLUGIN_BN_USESAVED == 0)
        ComputeType variance = 0;
#endif
        ComputeType invVariance = 0;
        ComputeType pscale = 0;
        ComputeType pbias = 0;
        ComputeType pdscale = 0;
        ComputeType pdbias = 0;

        // Unused if hip_plugin_bn_config::BATCHES >= hip_plugin_bn_config::MAX_BATCHES
        ComputeType batchValues[hip_plugin_bn_config::BATCHES]; // NOLINT(modernize-avoid-c-arrays)
        ComputeType dyValues[hip_plugin_bn_config::BATCHES]; // NOLINT(modernize-avoid-c-arrays)

        const unsigned int lid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        unsigned int index;
        const unsigned int channelIdx = grpid * hip_plugin_bn_config::SPATIAL_SIZE;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#else // recalc mean and variance

        if(lid < hip_plugin_bn_config::SPATIAL_SIZE)
        {
            for(int batch = 0; batch < hip_plugin_bn_config::BATCHES; batch++)
            {
                index = batch * hip_plugin_bn_config::BATCH_SIZE + channelIdx + lid;
                if constexpr(hip_plugin_bn_config::BATCHES < hip_plugin_bn_config::MAX_BATCHES)
                {
                    batchValues[batch] = hip_kernel_provider::cast<ComputeType>(xIn[index]);
                    mean += batchValues[batch];
                    variance = fma(batchValues[batch], batchValues[batch], variance);
                }
                else
                {
                    auto in = hip_kernel_provider::cast<ComputeType>(xIn[index]);
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }
        else
        {
            mean = hip_kernel_provider::cast<ComputeType>(0.0);
            variance = hip_kernel_provider::cast<ComputeType>(0.0);
        }

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
                lid);

        // REDUCTION COMPLETE -----------------------
        variance = fma(-mean, mean, variance);
        // NOLINTNEXTLINE(readability-use-std-min-max): <algorithm> is unavailable in hipRTC
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon); // NOLINT(bugprone-narrowing-conversions)

// RECALC of MEAN and VARIANCE complete
//===========================================
#endif

        if(lid < hip_plugin_bn_config::SPATIAL_SIZE)
        {
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCHES; batch++)
            {
                index = batch * hip_plugin_bn_config::BATCH_SIZE + channelIdx + lid;
                if constexpr(hip_plugin_bn_config::BATCHES < hip_plugin_bn_config::MAX_BATCHES)
                {
                    dyValues[batch] = hip_kernel_provider::cast<ComputeType>(dyIn[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                    batchValues[batch]
                        = (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance;
#else
                    batchValues[batch] = (batchValues[batch] - mean) * invVariance;
#endif // batchValues is now xhat

                    dyValues[batch] = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                        dyValues[batch], batchValues[batch], pscale, pbias, alpha, beta);

                    pdbias += dyValues[batch];
                    pdscale = fma(batchValues[batch], dyValues[batch], pdscale);
                }
                else
                {
                    auto dyValue = hip_kernel_provider::cast<ComputeType>(dyIn[index]);
                    const ComputeType xhat
                        = (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance;

                    dyValue = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                        dyValue, xhat, pscale, pbias, alpha, beta);

                    pdbias += dyValue;
                    pdscale = fma(xhat, dyValue, pdscale);
                }
            }
        }
        else
        {
            pdbias = 0;
            pdscale = 0;
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxL/NxHxW/NxDxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < hip_plugin_bn_config::SPATIAL_SIZE)
        {
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCHES; batch++)
            {
                index = batch * hip_plugin_bn_config::BATCH_SIZE + channelIdx + lid;
                ComputeType dyValue;
                ComputeType xhat;
                if constexpr(hip_plugin_bn_config::BATCHES < hip_plugin_bn_config::MAX_BATCHES)
                {
                    dyValue = dyValues[batch];
                    xhat = batchValues[batch];
                }
                else
                {
                    dyValue = hip_kernel_provider::cast<ComputeType>(dyIn[index]);
                    xhat
                        = (hip_kernel_provider::cast<ComputeType>(xIn[index]) - mean) * invVariance;

                    dyValue = bwdActivationOp<ComputeType, hip_plugin_config::NEURON_OP>(
                        dyValue, xhat, pscale, pbias, alpha, beta);
                }

                dxOut[index] = hip_kernel_provider::cast<InputType>(
                    batchBwdNormalization(dyValue, xhat, pdbias, pdscale, pscale, invVariance));
            }
        }
        if(lid == 0)
        {
            dbias[grpid] = pdbias;
            dscale[grpid] = pdscale;
        }
    }
};

} // namespace hip_kernel_provider::batchnorm

/// C interfaces

#if(HIP_PLUGIN_BN_VARIANT != 2)

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormBwdSpatial(const InputType* __restrict xIn,
                        const OutputType* __restrict dyIn,
                        InputType* __restrict dxOut,
                        const ScaleBiasType* __restrict bnScale,
                        const ScaleBiasType* __restrict bnBias,
                        ScaleBiasType* __restrict dscale,
                        ScaleBiasType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                        double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                        const MeanVarType* savedMean,
                        const MeanVarType* savedInvVariance,
#endif
                        ComputeType alpha,
                        ComputeType beta)
{
    using BwdSpatialHIPImpl
        = hip_kernel_provider::batchnorm::BatchNormBwdSpatialImpl<hip_plugin_bn_config::VARIANT>;

#if(HIP_PLUGIN_BN_USESAVED == 0)
    BwdSpatialHIPImpl{}(xIn, dyIn, dxOut, bnScale, bnBias, dscale, dbias, epsilon, alpha, beta);
#elif(HIP_PLUGIN_BN_USESAVED == 1)
    BwdSpatialHIPImpl{}(
        xIn, dyIn, dxOut, bnScale, bnBias, dscale, dbias, savedMean, savedInvVariance, alpha, beta);
#endif
}

#else

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM_FINAL.BOUNDS)
    batchNormBwdSpatialFinalMeanVariance(
        InputType* __restrict meanvarbuff, // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
        double epsilon)
{
    const unsigned int xlid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ylid = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zlid = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpId = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgid
        = blockDim.x * blockIdx.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpSize
        = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpSize
        = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpSize
        = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

    const unsigned int xstride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
    const unsigned int ystride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    ComputeChannelVecType invVariance;

    for(unsigned int zoffset = zlid;
        zoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                      .GRP2; // NOLINT(readability-static-accessed-through-instance)
        zoffset += zgrpSize)
    {
        for(unsigned int yoffset = ylid;
            yoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                      LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                          .GRP1; // NOLINT(readability-static-accessed-through-instance)
            yoffset += ygrpSize)
        {
            mean += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
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
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
            variance += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
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
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
        }
    }

    if constexpr(!hip_plugin_bn_config::USE_AMDGCN
                 || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP0 // NOLINT(readability-static-accessed-through-instance)
                        > 1 // NOLINT(readability-static-accessed-through-instance)
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ ComputeChannelVecType s_lcl_data
            [2
             * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
               LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                   .BOUNDS]; // NOLINT(readability-static-accessed-through-instance)
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
            mean,
            variance,
            hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
            s_lcl_data,
            xgrpSize,
            xlid,
            ylid + zlid * ygrpSize,
            ygrpSize * zgrpSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<
            ComputeChannelVecType,
            ComputeType,
            hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
            LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                .BOUNDS>( // NOLINT(readability-static-accessed-through-instance)
            mean,
            variance,
            hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::I_CHANNEL_SLICE_SIZE),
            ylid + zlid * ygrpSize);
    }

    variance = hip_kernel_provider::fma(-mean, mean, variance);
    variance
        = hip_kernel_provider::max(variance, hip_kernel_provider::cast<ComputeChannelVecType>(0.0));
    invVariance = hip_kernel_provider::rsqrt(
        variance + hip_kernel_provider::cast<ComputeChannelVecType>(epsilon));

    for(unsigned int zoffset = zlid;
        zoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                      .GRP2; // NOLINT(readability-static-accessed-through-instance)
        zoffset += zgrpSize)
    {
        for(unsigned int yoffset = ylid;
            yoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                      LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                          .GRP1; // NOLINT(readability-static-accessed-through-instance)
            yoffset += ygrpSize)
        {
            // Replicate mean and variance for all y groups because stash == dxOut and
            // BatchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            hip_kernel_provider::batchnorm::storeToStash(
                mean,
                reinterpret_cast<InputChannelVecType*>(meanvarbuff),
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
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
            hip_kernel_provider::batchnorm::storeToStash(
                invVariance,
                reinterpret_cast<InputChannelVecType*>(meanvarbuff),
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
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                xstride);
        }
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormBwdSpatialMeanVariance(
        const InputType* __restrict in,
        InputType* __restrict meanvarbuff) // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
{
    const unsigned int xlid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ylid = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zlid = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpId = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpId = blockIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpId = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgid
        = blockDim.x * blockIdx.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygid
        = blockDim.y * blockIdx.y // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgid
        = blockDim.z * blockIdx.z // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpSize
        = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpSize
        = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpSize
        = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

    const unsigned int xstride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
    const unsigned int ystride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::SPATIAL_SIZE
       && zgid < hip_plugin_bn_config::BATCHES)
    {
        const unsigned int indexBase
            = zgid * hip_plugin_bn_config::BATCH_ELEMENTS * hip_plugin_bn_config::BATCH_SIZE
              + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
              + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
        {
            const unsigned int index = indexBase + batch * hip_plugin_bn_config::BATCH_SIZE;
            auto value = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(in + index));

            hip_kernel_provider::batchnorm::accumulate(mean, value);
            hip_kernel_provider::batchnorm::accumulateMad(variance, value, value);
        }
    }

    if constexpr(!hip_plugin_bn_config::USE_AMDGCN
                 || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP0 // NOLINT(readability-static-accessed-through-instance)
                        > 1 // NOLINT(readability-static-accessed-through-instance)
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ ComputeChannelVecType s_lcl_data[2 * hip_plugin_bn_config::LDS_SIZE];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
            mean,
            variance,
            hip_kernel_provider::cast<ComputeChannelVecType>(1.0),
            s_lcl_data,
            xgrpSize,
            xlid,
            ylid + zlid * ygrpSize,
            ygrpSize * zgrpSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<ComputeChannelVecType,
                                                           ComputeType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            mean, variance, hip_kernel_provider::cast<ComputeType>(1.0), ylid + zlid * ygrpSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        hip_kernel_provider::batchnorm::storeToStash(
            mean,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            0,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
        hip_kernel_provider::batchnorm::storeToStash(
            variance,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            1,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
    }
} // end spatial mean kernel

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormBwdSpatialDScaleDBias(
        const InputType* __restrict xIn,
        const OutputType* __restrict dyIn,
        InputType* __restrict meanvarbuff, // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
        const ScaleBiasType* __restrict bnScale,
        const ScaleBiasType* __restrict bnBias,
#if HIP_PLUGIN_BN_USESAVED == 1
        const MeanVarType* __restrict savedMean,
        const MeanVarType* __restrict savedInvVariance,
#endif
        ComputeType alpha,
        ComputeType beta)
{
    const unsigned int xlid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ylid = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zlid = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpId = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpId = blockIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpId = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgid
        = blockDim.x * blockIdx.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygid
        = blockDim.y * blockIdx.y // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgid
        = blockDim.z * blockIdx.z // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpSize
        = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpSize
        = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpSize
        = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

    const unsigned int xstride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
    const unsigned int ystride
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    ComputeChannelVecType mean;
    ComputeChannelVecType invVar;
    auto dscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto dbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    __shared__ ComputeChannelVecType s_lmean
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_livar
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ ComputeChannelVecType s_lcl_scale
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_lcl_bias
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        s_lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
            0,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
        s_livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
            1,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
#else
        s_lmean[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedMean)[xgid]);
        s_livar[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedInvVariance)[xgid]);
#endif
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        s_lcl_scale[xlid] = reinterpret_cast<const ComputeChannelVecType*>(bnScale)[xgid];
        s_lcl_bias[xlid] = reinterpret_cast<const ComputeChannelVecType*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::SPATIAL_SIZE
       && zgid < hip_plugin_bn_config::BATCHES)
    {
        mean = s_lmean[xlid];
        invVar = s_livar[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pscale = s_lcl_scale[xlid];
        pbias = s_lcl_bias[xlid];
#endif

        const unsigned int indexBase
            = (zgid * hip_plugin_bn_config::BATCH_ELEMENTS) * hip_plugin_bn_config::BATCH_SIZE
              + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
              + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
        {
            const unsigned int index = indexBase + batch * hip_plugin_bn_config::BATCH_SIZE;
            auto value1 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const OutputVecType*>(dyIn + index));
            auto value2 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(xIn + index));
            const ComputeVecType xhat = (value2 - mean) * invVar;
            // apply activation function on dy
            value1 = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeVecType,
                                                                     hip_plugin_config::NEURON_OP>(
                value1,
                xhat,
                hip_kernel_provider::cast<ComputeVecType>(pscale),
                hip_kernel_provider::cast<ComputeVecType>(pbias),
                hip_kernel_provider::cast<ComputeVecType>(alpha),
                hip_kernel_provider::cast<ComputeVecType>(beta));

            hip_kernel_provider::batchnorm::accumulate(dbias, value1);
            hip_kernel_provider::batchnorm::accumulateMad(dscale, xhat, value1);
        }
    }

    if constexpr(!hip_plugin_bn_config::USE_AMDGCN
                 || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP0 // NOLINT(readability-static-accessed-through-instance)
                        > 1 // NOLINT(readability-static-accessed-through-instance)
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ ComputeChannelVecType s_lcl_data[2 * hip_plugin_bn_config::LDS_SIZE];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
            dscale,
            dbias,
            hip_kernel_provider::cast<ComputeType>(1.0),
            s_lcl_data,
            xgrpSize,
            xlid,
            ylid + zlid * ygrpSize,
            ygrpSize * zgrpSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeChannelVecType, ComputeType, hip_plugin_bn_config::LDS_SIZE>(
                dscale, dbias, hip_kernel_provider::cast<ComputeType>(1.0), ylid + zlid * ygrpSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        const unsigned int stashIndex = hip_plugin_bn_config::USE_SAVED ? 0 : 2;
        hip_kernel_provider::batchnorm::storeToStash(
            dscale,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            stashIndex,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
        hip_kernel_provider::batchnorm::storeToStash(
            dbias,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            stashIndex + 1,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            ystride / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            xstride);
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM_FINAL.BOUNDS)
    batchNormBwdSpatialFinalDScaleDBias(
        const InputType* __restrict meanvarbuff, // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
        ScaleBiasType* __restrict dscale,
        ScaleBiasType* __restrict dbias)
{
    const unsigned int xlid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ylid = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zlid = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpId = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgid
        = blockDim.x * blockIdx.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgrpSize
        = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygrpSize
        = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgrpSize
        = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
    constexpr unsigned int YSTRIDE
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;
    constexpr unsigned int STASH_INDEX = hip_plugin_bn_config::USE_SAVED ? 0 : 2;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    auto pdscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pdbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    for(unsigned int zoffset = zlid;
        zoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                  LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                      .GRP2; // NOLINT(readability-static-accessed-through-instance)
        zoffset += zgrpSize)
    {
        for(unsigned int yoffset = ylid;
            yoffset < hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                      LAUNCH_GROUPS // NOLINT(readability-static-accessed-through-instance)
                          .GRP1; // NOLINT(readability-static-accessed-through-instance)
            yoffset += ygrpSize)
        {
            pdscale += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                STASH_INDEX,
                hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                    LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                        .GRP2 // NOLINT(readability-static-accessed-through-instance)
                    * zoffset // NOLINT(readability-static-accessed-through-instance)
                    * hip_plugin_bn_config::BATCH_ELEMENTS,
                hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                    LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                        .GRP1 // NOLINT(readability-static-accessed-through-instance)
                    * yoffset // NOLINT(readability-static-accessed-through-instance)
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                XSTRIDE);
            pdbias += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                STASH_INDEX + 1,
                hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                    LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                        .GRP2 // NOLINT(readability-static-accessed-through-instance)
                    * zoffset // NOLINT(readability-static-accessed-through-instance)
                    * hip_plugin_bn_config::BATCH_ELEMENTS,
                hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                    LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                        .GRP1 // NOLINT(readability-static-accessed-through-instance)
                    * yoffset // NOLINT(readability-static-accessed-through-instance)
                    * hip_plugin_bn_config::
                        VEC_SIZE_Y, // NOLINT(readability-static-accessed-through-instance)
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xgrpSize,
                xgrpId,
                xlid,
                XSTRIDE);
        }
    }

    if constexpr(!hip_plugin_bn_config::USE_AMDGCN
                 || hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
                        LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
                            .GRP0 // NOLINT(readability-static-accessed-through-instance)
                        > 1 // NOLINT(readability-static-accessed-through-instance)
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ ComputeChannelVecType s_lcl_data
            [2
             * hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
               LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                   .BOUNDS]; // NOLINT(readability-static-accessed-through-instance)
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(
            pdscale,
            pdbias,
            hip_kernel_provider::cast<ComputeType>(1.0),
            s_lcl_data,
            xgrpSize,
            xlid,
            ylid + zlid * ygrpSize,
            ygrpSize * zgrpSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<
            ComputeChannelVecType,
            ComputeType,
            hip_plugin_bn_config:: // NOLINT(readability-static-accessed-through-instance)
            LAUNCH_DIM_FINAL // NOLINT(readability-static-accessed-through-instance)
                .BOUNDS>( // NOLINT(readability-static-accessed-through-instance)
            pdscale,
            pdbias,
            hip_kernel_provider::cast<ComputeType>(1.0),
            ylid + zlid * ygrpSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<ScaleBiasChannelVecType*>(dscale)[xgid]
            = hip_kernel_provider::cast<ScaleBiasChannelVecType>(pdscale);
        reinterpret_cast<ScaleBiasChannelVecType*>(dbias)[xgid]
            = hip_kernel_provider::cast<ScaleBiasChannelVecType>(pdbias);
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.BOUNDS)
    batchNormBwdSpatialDX(const InputType* __restrict xIn,
                          const OutputType* __restrict dyIn,
                          InputType* __restrict dxOut,
                          const ScaleBiasType* __restrict bnScale,
                          const ScaleBiasType* __restrict bnBias,
                          const ScaleBiasType* __restrict dscale,
                          const ScaleBiasType* __restrict dbias,
#if HIP_PLUGIN_BN_USESAVED == 1
                          const MeanVarType* __restrict savedMean,
                          const MeanVarType* __restrict savedInvVariance,
#endif
                          ComputeType alpha,
                          ComputeType beta)
{
    const unsigned int xlid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ylid = threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zlid = threadIdx.z; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int xgid
        = blockDim.x * blockIdx.x // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int ygid
        = blockDim.y * blockIdx.y // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.y; // NOLINT(readability-static-accessed-through-instance)
    const unsigned int zgid
        = blockDim.z * blockIdx.z // NOLINT(readability-static-accessed-through-instance)
          + threadIdx.z; // NOLINT(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? 1 : hip_plugin_bn_config::SPATIAL_SIZE;
    constexpr unsigned int YSTRIDE
        = hip_plugin_config::LAYOUT_CHANNEL_LAST ? hip_plugin_bn_config::CHANNELS : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::CHANNELS)
    {
        return;
    }

    ComputeChannelVecType mean;
    ComputeChannelVecType invVar;
    ComputeChannelVecType pscale;
    ComputeChannelVecType pdscale;
    ComputeChannelVecType pdbias;
    auto pbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    __shared__ ComputeChannelVecType s_lscale
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_ldscale
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_ldbias
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_lmean
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
    __shared__ ComputeChannelVecType s_livar
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ ComputeChannelVecType s_lbias
        [hip_plugin_bn_config::LAUNCH_DIM // NOLINT(readability-static-accessed-through-instance)
             .GRP0]; // NOLINT(readability-static-accessed-through-instance)
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        const unsigned int xgrpId
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpId
            = blockIdx.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpId
            = blockIdx.z; // NOLINT(readability-static-accessed-through-instance)

        const unsigned int xgrpSize
            = blockDim.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int ygrpSize
            = blockDim.y; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int zgrpSize
            = blockDim.z; // NOLINT(readability-static-accessed-through-instance)

        s_lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(dxOut),
            0,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            XSTRIDE);
        s_livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(dxOut),
            1,
            zgrpSize * zgrpId * hip_plugin_bn_config::BATCH_ELEMENTS,
            ygrpSize * ygrpId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xgrpSize,
            xgrpId,
            xlid,
            XSTRIDE);
#else
        s_lmean[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedMean)[xgid]);
        s_livar[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedInvVariance)[xgid]);
#endif
        s_lscale[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(bnScale)[xgid]);
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        s_lbias[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(bnBias)[xgid]);
#endif
        s_ldscale[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(dscale)[xgid]);
        s_ldbias[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(dbias)[xgid]);
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::SPATIAL_SIZE
       && zgid < hip_plugin_bn_config::BATCHES)
    {
        mean = s_lmean[xlid];
        invVar = s_livar[xlid];
        pscale = s_lscale[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = s_lbias[xlid];
#endif
        pdscale = s_ldscale[xlid];
        pdbias = s_ldbias[xlid];

        const unsigned int indexBase
            = (zgid * hip_plugin_bn_config::BATCH_ELEMENTS) * hip_plugin_bn_config::BATCH_SIZE
              + ygid * YSTRIDE * hip_plugin_bn_config::VEC_SIZE_Y
              + xgid * XSTRIDE * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::BATCH_ELEMENTS; batch++)
        { // apply normalization
            const unsigned int index = indexBase + batch * hip_plugin_bn_config::BATCH_SIZE;
            auto xI = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(xIn + index));
            const ComputeVecType xhat = (xI - mean) * invVar; // recalculating this again...
            auto value1 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const OutputVecType*>(dyIn + index));
            value1 = hip_kernel_provider::batchnorm::bwdActivationOp<ComputeVecType,
                                                                     hip_plugin_config::NEURON_OP>(
                value1,
                xhat,
                hip_kernel_provider::cast<ComputeVecType>(pscale),
                hip_kernel_provider::cast<ComputeVecType>(pbias),
                hip_kernel_provider::cast<ComputeVecType>(alpha),
                hip_kernel_provider::cast<ComputeVecType>(beta));

            *reinterpret_cast<InputVecType*>(dxOut + index)
                = hip_kernel_provider::cast<InputVecType>(
                    batchBwdNormalizationVar2(value1,
                                              xhat,
                                              hip_kernel_provider::cast<ComputeVecType>(pdbias),
                                              hip_kernel_provider::cast<ComputeVecType>(pdscale),
                                              hip_kernel_provider::cast<ComputeVecType>(pscale),
                                              hip_kernel_provider::cast<ComputeVecType>(invVar)));
        }
    }
}

#endif
