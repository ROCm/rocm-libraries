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
        hip_kernel_provider::cast<ComputeWriteVecType>(hip_plugin_bn_config::channel_slice_size),
        value,
        hip_kernel_provider::cast<ComputeWriteVecType>(-dbias));
    ComputeWriteVecType tmp2 = -xhat * hip_kernel_provider::cast<ComputeWriteVecType>(dscale);
    ComputeType tmp3 = pscale * invVariance * hip_plugin_bn_config::i_channel_slice_size;
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
        hip_kernel_provider::cast<ComputeWriteVecType>(hip_plugin_bn_config::channel_slice_size),
        value,
        -dbias);
    ComputeWriteVecType tmp2 = -xhat * dscale;
    ComputeWriteVecType tmp3 = pscale * invVariance * hip_plugin_bn_config::i_channel_slice_size;
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
    auto constexpr size = hip_kernel_provider::mapped_vector_info<ComputeReadVecType>::size;
    if constexpr(size == 4)
    {
        ComputeReadVecType out;
        out.x = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        out.z = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.z, xnorm.z, scale, bias, alpha, beta);
        out.w = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.w, xnorm.w, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(size == 2)
    {
        ComputeReadVecType out;
        out.x = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                  hip_plugin_config::neuron_op>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(size == 1)
    {
        return hip_kernel_provider::batchnorm::bwd_activation_op<ComputeType,
                                                                 hip_plugin_config::neuron_op>(
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
    static constexpr unsigned int segtmp1
        = hip_plugin_bn_config::launch_dim.grp0 / hip_plugin_bn_config::spatial_size;
    static constexpr unsigned int segtmp2 = segtmp1 == 0 ? 1 : segtmp1;
    static constexpr unsigned int segtmp = hip_plugin_bn_config::spatial_size * segtmp2;
    static constexpr unsigned int segment = segtmp > hip_plugin_bn_config::channel_slice_size
                                                ? hip_plugin_bn_config::channel_slice_size
                                                : segtmp;
    static constexpr unsigned int batchLoop
        = (hip_plugin_bn_config::channel_slice_size + segment - 1) / segment;
    static constexpr unsigned int segiSpatial = segment / hip_plugin_bn_config::spatial_size;
    static_assert(batchLoop > 0);
    static constexpr unsigned int batchLoopm = batchLoop - 1;
    static constexpr unsigned int snSpatial = batchLoopm * segiSpatial;

    __forceinline__ __device__ void operator()(const InputType* __restrict x_in,
                                               const OutputType* __restrict dy_in,
                                               InputType* __restrict dx_out,
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

        ComputeType batchValues[batchLoop];
        ComputeType dyValues[batchLoop];

        __shared__ ComputeType lbnScale;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        __shared__ ComputeType lbnBias;
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        __shared__ ComputeType lmean, lvar;
#endif
        unsigned int index = 0;
        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int channelSpatialId = grpid * hip_plugin_bn_config::spatial_size
                                        + (lid % hip_plugin_bn_config::spatial_size);
        unsigned int lidiSpatial = lid / hip_plugin_bn_config::spatial_size;
        unsigned int batchId = 0;

        if(lid == 0)
        {
            lbnScale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
            lbnBias = bnBias[grpid];
#endif
        }

#if(HIP_PLUGIN_BN_USESAVED == 1)
        if(lid == 0)
        {
            lmean = savedMean[grpid];
            lvar = savedInvVariance[grpid];
        }
        __syncthreads();
        mean = lmean;
        invVariance = lvar;
#else // recalc mean and variance below \
    // == RECALC MEAN AND VARIANCE ===========
        if(lid < segment)
        {
            for(unsigned int batch = 0; batch < batchLoopm; ++batch)
            {
                batchId = batch * segiSpatial + lidiSpatial;
                index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
                batchValues[batch] = hip_kernel_provider::cast<ComputeType>(x_in[index]);
                mean += batchValues[batch];
                variance = fma(batchValues[batch], batchValues[batch], variance);
            }
            batchId = snSpatial + lidiSpatial;
            index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
            batchValues[batchLoopm] = index < hip_plugin_bn_config::tensor_size
                                          ? hip_kernel_provider::cast<ComputeType>(x_in[index])
                                          : 0;
            mean += batchValues[batchLoopm];
            variance = fma(batchValues[batchLoopm], batchValues[batchLoopm], variance);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lid);

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon);

#endif // end -- Recalc mean and variance \
    //-------------------------------------------
        pscale = lbnScale;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = lbnBias;
#endif

        //==== CALC DB and DS =========================================
        if(lid < segment)
        {
            for(unsigned int batch = 0; batch < batchLoopm; ++batch)
            {
                batchId = batch * segiSpatial + lidiSpatial;
                index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
                dyValues[batch] = hip_kernel_provider::cast<ComputeType>(dy_in[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                batchValues[batch]
                    = (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean) * invVariance;
#else
                batchValues[batch] = (batchValues[batch] - mean) * invVariance;
#endif
                dyValues[batch] = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
                    dyValues[batch], batchValues[batch], pscale, pbias, alpha, beta);
                // batchValues is now xhat
                pdbias += dyValues[batch];
                pdscale = fma(batchValues[batch], dyValues[batch], pdscale);
            }
            batchId = snSpatial + lidiSpatial;
            index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
            dyValues[batchLoopm] = index < hip_plugin_bn_config::tensor_size
                                       ? hip_kernel_provider::cast<ComputeType>(dy_in[index])
                                       : hip_kernel_provider::cast<ComputeType>(0.0);

#if(HIP_PLUGIN_BN_USESAVED == 1)
            batchValues[batchLoopm]
                = index < hip_plugin_bn_config::tensor_size
                      ? (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean) * invVariance
                      : 0;
#else
            batchValues[batchLoopm] = (batchValues[batchLoopm] - mean) * invVariance;
#endif
            dyValues[batchLoopm] = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
                dyValues[batchLoopm], batchValues[batchLoopm], pscale, pbias, alpha, beta);
            // batchValues is now xhat
            pdbias += dyValues[batchLoopm];
            pdscale = fma(batchValues[batchLoopm], dyValues[batchLoopm], pdscale);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        if(lid < segment)
        {
            //==== CALC NORM =======================
            for(unsigned int batch = 0; batch < batchLoopm; batch++)
            {
                batchId = batch * segiSpatial + lidiSpatial;
                index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
                dx_out[index] = hip_kernel_provider::cast<OutputType>(batchBwdNormalization(
                    dyValues[batch], batchValues[batch], pdbias, pdscale, pscale, invVariance));
            } // end for
            batchId = snSpatial + lidiSpatial;
            index = batchId * hip_plugin_bn_config::batch_size + channelSpatialId;
            if(index < hip_plugin_bn_config::tensor_size)
            {
                dx_out[index] = hip_kernel_provider::cast<OutputType>(
                    batchBwdNormalization(dyValues[batchLoopm],
                                          batchValues[batchLoopm],
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
    static constexpr unsigned int read_size = hip_plugin_config::layout_channel_last ? 1 : 4;
    static constexpr unsigned int write_size = hip_plugin_config::layout_channel_last ? 1 : 2;

    using InputReadVecType = typename mapped_vector_type<InputType, read_size>::type;
    using OutputReadVecType = typename mapped_vector_type<OutputType, read_size>::type;
    using ComputeReadVecType = typename mapped_vector_type<ComputeType, read_size>::type;
    using InputWriteVecType = typename mapped_vector_type<InputType, write_size>::type;
    using OutputWriteVecType = typename mapped_vector_type<OutputType, write_size>::type;
    using ComputeWriteVecType = typename mapped_vector_type<ComputeType, write_size>::type;

    static constexpr unsigned int read_block = 1;
    static constexpr unsigned int grprd
        = hip_plugin_bn_config::launch_dim.grp0 * read_block * read_size;
    static constexpr unsigned int rem4
        = hip_plugin_bn_config::channel_slice_size
          - (hip_plugin_bn_config::channel_slice_size / grprd) * grprd;
    static constexpr unsigned int less4 = hip_plugin_bn_config::channel_slice_size - rem4;
    static constexpr unsigned int rem
        = hip_plugin_bn_config::channel_slice_size
          - (hip_plugin_bn_config::channel_slice_size / hip_plugin_bn_config::launch_dim.grp0)
                * hip_plugin_bn_config::launch_dim.grp0;
    static constexpr unsigned int less = hip_plugin_bn_config::channel_slice_size - rem;
    static constexpr unsigned int chunk = write_size * hip_plugin_bn_config::launch_dim.grp0;
    static constexpr unsigned int remout
        = hip_plugin_bn_config::channel_slice_size
          - ((hip_plugin_bn_config::channel_slice_size / chunk) * chunk);
    static constexpr unsigned int lessout = hip_plugin_bn_config::channel_slice_size - remout;

    __forceinline__ __device__ unsigned int getTensorIndex(unsigned int loopIndex)
    {
        unsigned int grpid = blockIdx.x;
        unsigned int channelSpatialId = grpid * hip_plugin_bn_config::spatial_size;
        unsigned int batchIdx = loopIndex / hip_plugin_bn_config::spatial_size;
        unsigned int spatialIdx = loopIndex - (batchIdx * hip_plugin_bn_config::spatial_size);
        return hip_plugin_config::layout_channel_last
                   ? batchIdx * hip_plugin_bn_config::batch_size
                         + spatialIdx * hip_plugin_bn_config::channels + grpid
                   : batchIdx * hip_plugin_bn_config::batch_size + channelSpatialId + spatialIdx;
    }

    __forceinline__ __device__ void operator()(const InputType* __restrict x_in,
                                               const OutputType* __restrict dy_in,
                                               InputType* __restrict dx_out,
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
        ComputeType xhat = 0;

        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int channelSpatialId = grpid * hip_plugin_bn_config::spatial_size;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 0)
        //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
        ComputeType variance = 0;
        if constexpr(!hip_plugin_config::layout_channel_last
                     && hip_plugin_bn_config::spatial_size >= 4096)
        {
            ComputeReadVecType read4;
            for(unsigned int k = lid << 2; k < less4; k += grprd)
            {
                read4 = hip_kernel_provider::cast<ComputeReadVecType>(
                    *reinterpret_cast<const InputReadVecType*>(x_in + getTensorIndex(k)));
                hip_kernel_provider::batchnorm::_accumulate(mean, read4);
                hip_kernel_provider::batchnorm::_accumulate_mad(variance, read4, read4);
            }

            if constexpr(rem4 > 0)
            {
                if(lid < rem4)
                {
                    unsigned int index = getTensorIndex((lid << 2) + less4);
                    if(index + read_size - 1 < hip_plugin_bn_config::tensor_size)
                    {
                        read4 = hip_kernel_provider::cast<ComputeReadVecType>(
                            *reinterpret_cast<const InputReadVecType*>(x_in + index));
                        hip_kernel_provider::batchnorm::_accumulate(mean, read4);
                        hip_kernel_provider::batchnorm::_accumulate_mad(variance, read4, read4);
                    }
                }
            }
        }
        else
        {
            for(unsigned int k = lid; k < less; k += hip_plugin_bn_config::launch_dim.grp0)
            {
                auto in = hip_kernel_provider::cast<ComputeType>(x_in[getTensorIndex(k)]);
                mean += in;
                variance = fma(in, in, variance);
            }
            if constexpr(rem > 0)
            {
                if(lid < rem)
                {
                    unsigned int index = getTensorIndex(lid + less);
                    auto in = (index < hip_plugin_bn_config::tensor_size)
                                  ? hip_kernel_provider::cast<ComputeType>(x_in[index])
                                  : hip_kernel_provider::cast<ComputeType>(0.0);
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lid);

        // REDUCTION COMPLETE ---------------------------
        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon);

#else // HIP_PLUGIN_BN_USESAVED == 1
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#endif

        constexpr unsigned int readUnrollHint
            = hip_plugin_bn_config::batches > hip_plugin_bn_config::loop_unroll_max_batch ? 4 : 2;
        static_unroll_count<unsigned int, 0, less4, grprd, readUnrollHint>{[&](unsigned int k) {
            unsigned int l = k + (lid << 2 * (1 - hip_plugin_config::layout_channel_last));
            if(l < less4)
            {
                unsigned int index = getTensorIndex(l);
                auto xread = *reinterpret_cast<const InputReadVecType*>(x_in + index);
                auto dyRead = *reinterpret_cast<const OutputReadVecType*>(dy_in + index);
                auto dyValue = hip_kernel_provider::cast<ComputeReadVecType>(dyRead);
                ComputeReadVecType xhat
                    = (hip_kernel_provider::cast<ComputeReadVecType>(xread) - mean) * invVariance;

                dyValue = vectorizedBwdActivationOp<ComputeReadVecType>(
                    dyValue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::_accumulate(pdbias, dyValue);
                hip_kernel_provider::batchnorm::_accumulate_mad(pdscale, xhat, dyValue);
            }
        }};

        if constexpr(rem4 > 0)
        {
            unsigned int index
                = getTensorIndex((lid << 2 * (1 - hip_plugin_config::layout_channel_last)) + less4);
            if(index + read_size - 1 < hip_plugin_bn_config::tensor_size)
            {
                auto xread = *reinterpret_cast<const InputReadVecType*>(x_in + index);
                auto dyRead = *reinterpret_cast<const OutputReadVecType*>(dy_in + index);
                auto dyValue = hip_kernel_provider::cast<ComputeReadVecType>(dyRead);
                auto xhat
                    = (hip_kernel_provider::cast<ComputeReadVecType>(xread) - mean) * invVariance;

                dyValue = vectorizedBwdActivationOp<ComputeReadVecType>(
                    dyValue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::_accumulate(pdbias, dyValue);
                hip_kernel_provider::batchnorm::_accumulate_mad(pdscale, xhat, dyValue);
            }
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        __syncthreads();

        if(lid == 0)
        {
            dbias[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdbias);
            dscale[grpid] = hip_kernel_provider::cast<ScaleBiasType>(pdscale);
        }

        constexpr unsigned int writeUnrollHint
            = hip_plugin_bn_config::batches > hip_plugin_bn_config::loop_unroll_max_batch ? 2 : 1;
        static_unroll_count<unsigned int, 0, lessout, chunk, writeUnrollHint>{[&](unsigned int k) {
            // Unrolling the loop requires forcing explicit data vectorization otherwise the
            // compiler will start splitting global loads into smaller chunks resulting in
            // significant slowdown.
            ComputeWriteVecType vals;
            unsigned int l = k + (write_size * lid);
            unsigned int index = getTensorIndex(l);
            if(l < lessout)
            {
                auto xread = *reinterpret_cast<const InputWriteVecType*>(x_in + index);
                auto dyRead = *reinterpret_cast<const OutputWriteVecType*>(dy_in + index);
                auto value1 = hip_kernel_provider::cast<ComputeWriteVecType>(dyRead);
                ComputeWriteVecType xhat1
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

            if(l < lessout)
            {
                *reinterpret_cast<InputWriteVecType*>(dx_out + index)
                    = hip_kernel_provider::cast<InputWriteVecType>(vals);
            }
        }};

        if constexpr(remout > 0)
        {
            unsigned int remkeyout = (write_size * lid) + lessout;
            for(unsigned int j = 0; j < write_size; j++)
            {
                unsigned int index = getTensorIndex(remkeyout + j);
                if(index < hip_plugin_bn_config::tensor_size)
                {
                    auto value1 = hip_kernel_provider::cast<ComputeType>(dy_in[index]);
                    ComputeType xhat = (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean)
                                       * invVariance;

                    value1 = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
                        value1, xhat, pscale, pbias, alpha, beta);

                    dx_out[index] = hip_kernel_provider::cast<InputType>(
                        batchBwdNormalization(value1, xhat, pdbias, pdscale, pscale, invVariance));
                }
            }
        }
    }
};

template <>
struct BatchNormBwdSpatialImpl<3>
{
    __forceinline__ __device__ void operator()(const InputType* __restrict x_in,
                                               const OutputType* __restrict dy_in,
                                               InputType* __restrict dx_out,
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

        // Unused if hip_plugin_bn_config::batches >= hip_plugin_bn_config::max_batches
        ComputeType batchValues[hip_plugin_bn_config::batches];
        ComputeType dyValues[hip_plugin_bn_config::batches];

        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int index;
        unsigned int channelIdx = grpid * hip_plugin_bn_config::spatial_size;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#else // recalc mean and variance

        if(lid < hip_plugin_bn_config::spatial_size)
        {
            for(int batch = 0; batch < hip_plugin_bn_config::batches; batch++)
            {
                index = batch * hip_plugin_bn_config::batch_size + channelIdx + lid;
                if constexpr(hip_plugin_bn_config::batches < hip_plugin_bn_config::max_batches)
                {
                    batchValues[batch] = hip_kernel_provider::cast<ComputeType>(x_in[index]);
                    mean += batchValues[batch];
                    variance = fma(batchValues[batch], batchValues[batch], variance);
                }
                else
                {
                    auto in = hip_kernel_provider::cast<ComputeType>(x_in[index]);
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
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                mean,
                variance,
                static_cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
                lid);

        // REDUCTION COMPLETE -----------------------
        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon);

// RECALC of MEAN and VARIANCE complete
//===========================================
#endif

        if(lid < hip_plugin_bn_config::spatial_size)
        {
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::batches; batch++)
            {
                index = batch * hip_plugin_bn_config::batch_size + channelIdx + lid;
                if constexpr(hip_plugin_bn_config::batches < hip_plugin_bn_config::max_batches)
                {
                    dyValues[batch] = hip_kernel_provider::cast<ComputeType>(dy_in[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                    batchValues[batch]
                        = (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean)
                          * invVariance;
#else
                    batchValues[batch] = (batchValues[batch] - mean) * invVariance;
#endif // batchValues is now xhat

                    dyValues[batch] = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
                        dyValues[batch], batchValues[batch], pscale, pbias, alpha, beta);

                    pdbias += dyValues[batch];
                    pdscale = fma(batchValues[batch], dyValues[batch], pdscale);
                }
                else
                {
                    auto dyValue = hip_kernel_provider::cast<ComputeType>(dy_in[index]);
                    ComputeType xhat = (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean)
                                       * invVariance;

                    dyValue = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
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
            reduce2<ComputeType, ComputeType, hip_plugin_bn_config::lds_size>(
                pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), lid);

        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxL/NxHxW/NxDxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < hip_plugin_bn_config::spatial_size)
        {
            for(unsigned int batch = 0; batch < hip_plugin_bn_config::batches; batch++)
            {
                index = batch * hip_plugin_bn_config::batch_size + channelIdx + lid;
                ComputeType dyValue;
                ComputeType xhat;
                if constexpr(hip_plugin_bn_config::batches < hip_plugin_bn_config::max_batches)
                {
                    dyValue = dyValues[batch];
                    xhat = batchValues[batch];
                }
                else
                {
                    dyValue = hip_kernel_provider::cast<ComputeType>(dy_in[index]);
                    xhat = (hip_kernel_provider::cast<ComputeType>(x_in[index]) - mean)
                           * invVariance;

                    dyValue = bwd_activation_op<ComputeType, hip_plugin_config::neuron_op>(
                        dyValue, xhat, pscale, pbias, alpha, beta);
                }

                dx_out[index] = hip_kernel_provider::cast<InputType>(
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

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormBwdSpatial(const InputType* __restrict x_in,
                        const OutputType* __restrict dy_in,
                        InputType* __restrict dx_out,
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
        = hip_kernel_provider::batchnorm::BatchNormBwdSpatialImpl<hip_plugin_bn_config::variant>;

#if(HIP_PLUGIN_BN_USESAVED == 0)
    BwdSpatialHIPImpl{}(x_in, dy_in, dx_out, bnScale, bnBias, dscale, dbias, epsilon, alpha, beta);
#elif(HIP_PLUGIN_BN_USESAVED == 1)
    BwdSpatialHIPImpl{}(x_in,
                        dy_in,
                        dx_out,
                        bnScale,
                        bnBias,
                        dscale,
                        dbias,
                        savedMean,
                        savedInvVariance,
                        alpha,
                        beta);
#endif
}

#else

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim_final.bounds)
    BatchNormBwdSpatialFinalMeanVariance(
        InputType* __restrict meanvarbuff, // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
        double epsilon)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_size = blockDim.x;
    unsigned int ygrp_size = blockDim.y;
    unsigned int zgrp_size = blockDim.z;

    unsigned int xstride
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
    unsigned int ystride
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
    {
        return;
    }

    auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    ComputeChannelVecType invVariance;

    for(unsigned int zoffset = zlid; zoffset < hip_plugin_bn_config::launch_groups.grp2;
        zoffset += zgrp_size)
    {
        for(unsigned int yoffset = ylid; yoffset < hip_plugin_bn_config::launch_groups.grp1;
            yoffset += ygrp_size)
        {
            mean += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                0,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
            variance += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                1,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ ComputeChannelVecType
            lcl_data[2 * hip_plugin_bn_config::launch_dim_final.bounds];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(
            mean,
            variance,
            hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
            lcl_data,
            xgrp_size,
            xlid,
            ylid + zlid * ygrp_size,
            ygrp_size * zgrp_size);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<
            ComputeChannelVecType,
            ComputeType,
            hip_plugin_bn_config::launch_dim_final.bounds>(
            mean,
            variance,
            hip_kernel_provider::cast<ComputeType>(hip_plugin_bn_config::i_channel_slice_size),
            ylid + zlid * ygrp_size);
    }

    variance = hip_kernel_provider::fma(-mean, mean, variance);
    variance
        = hip_kernel_provider::max(variance, hip_kernel_provider::cast<ComputeChannelVecType>(0.0));
    invVariance = hip_kernel_provider::rsqrt(
        variance + hip_kernel_provider::cast<ComputeChannelVecType>(epsilon));

    for(unsigned int zoffset = zlid; zoffset < hip_plugin_bn_config::launch_groups.grp2;
        zoffset += zgrp_size)
    {
        for(unsigned int yoffset = ylid; yoffset < hip_plugin_bn_config::launch_groups.grp1;
            yoffset += ygrp_size)
        {
            // Replicate mean and variance for all y groups because stash == dx_out and
            // BatchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            hip_kernel_provider::batchnorm::storeToStash(
                mean,
                reinterpret_cast<InputChannelVecType*>(meanvarbuff),
                0,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
            hip_kernel_provider::batchnorm::storeToStash(
                invVariance,
                reinterpret_cast<InputChannelVecType*>(meanvarbuff),
                1,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
        }
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormBwdSpatialMeanVariance(
        const InputType* __restrict in,
        InputType* __restrict meanvarbuff) // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int ygrp_id = blockIdx.y;
    unsigned int zgrp_id = blockIdx.z;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int xgrp_size = blockDim.x;
    unsigned int ygrp_size = blockDim.y;
    unsigned int zgrp_size = blockDim.z;

    unsigned int xstride
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
    unsigned int ystride
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
    {
        return;
    }

    auto variance = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto mean = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::spatial_size
       && zgid < hip_plugin_bn_config::batches)
    {
        unsigned int index_base
            = zgid * hip_plugin_bn_config::batch_elements * hip_plugin_bn_config::batch_size
              + ygid * ystride * hip_plugin_bn_config::vec_size_y
              + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
        {
            unsigned int index = index_base + batch * hip_plugin_bn_config::batch_size;
            auto value = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(in + index));

            hip_kernel_provider::batchnorm::_accumulate(mean, value);
            hip_kernel_provider::batchnorm::_accumulate_mad(variance, value, value);
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
            hip_kernel_provider::cast<ComputeChannelVecType>(1.0),
            lcl_data,
            xgrp_size,
            xlid,
            ylid + zlid * ygrp_size,
            ygrp_size * zgrp_size);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<ComputeChannelVecType,
                                                           ComputeType,
                                                           hip_plugin_bn_config::lds_size>(
            mean, variance, hip_kernel_provider::cast<ComputeType>(1.0), ylid + zlid * ygrp_size);
    }

    if(ylid == 0 && zlid == 0)
    {
        hip_kernel_provider::batchnorm::storeToStash(
            mean,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            0,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
        hip_kernel_provider::batchnorm::storeToStash(
            variance,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            1,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
    }
} // end spatial mean kernel

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormBwdSpatialDScaleDBias(
        const InputType* __restrict x_in,
        const OutputType* __restrict dy_in,
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
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int ygrp_id = blockIdx.y;
    unsigned int zgrp_id = blockIdx.z;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int xgrp_size = blockDim.x;
    unsigned int ygrp_size = blockDim.y;
    unsigned int zgrp_size = blockDim.z;

    unsigned int xstride
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
    unsigned int ystride
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
    {
        return;
    }

    ComputeChannelVecType mean, invVar;
    auto dscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto dbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    __shared__ ComputeChannelVecType lmean[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType livar[hip_plugin_bn_config::launch_dim.grp0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ ComputeChannelVecType lcl_scale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType lcl_bias[hip_plugin_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
            0,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
            1,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedMean)[xgid]);
        livar[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedInvVariance)[xgid]);
#endif
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        lcl_scale[xlid] = reinterpret_cast<const ComputeChannelVecType*>(bnScale)[xgid];
        lcl_bias[xlid] = reinterpret_cast<const ComputeChannelVecType*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::spatial_size
       && zgid < hip_plugin_bn_config::batches)
    {
        mean = lmean[xlid];
        invVar = livar[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pscale = lcl_scale[xlid];
        pbias = lcl_bias[xlid];
#endif

        unsigned int index_base
            = (zgid * hip_plugin_bn_config::batch_elements) * hip_plugin_bn_config::batch_size
              + ygid * ystride * hip_plugin_bn_config::vec_size_y
              + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
        {
            unsigned int index = index_base + batch * hip_plugin_bn_config::batch_size;
            auto value1 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const OutputVecType*>(dy_in + index));
            auto value2 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(x_in + index));
            ComputeVecType xhat = (value2 - mean) * invVar;
            // apply activation function on dy
            value1
                = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeVecType,
                                                                    hip_plugin_config::neuron_op>(
                    value1,
                    xhat,
                    hip_kernel_provider::cast<ComputeVecType>(pscale),
                    hip_kernel_provider::cast<ComputeVecType>(pbias),
                    hip_kernel_provider::cast<ComputeVecType>(alpha),
                    hip_kernel_provider::cast<ComputeVecType>(beta));

            hip_kernel_provider::batchnorm::_accumulate(dbias, value1);
            hip_kernel_provider::batchnorm::_accumulate_mad(dscale, xhat, value1);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ ComputeChannelVecType lcl_data[2 * hip_plugin_bn_config::lds_size];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(
            dscale,
            dbias,
            hip_kernel_provider::cast<ComputeType>(1.0),
            lcl_data,
            xgrp_size,
            xlid,
            ylid + zlid * ygrp_size,
            ygrp_size * zgrp_size);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<ComputeChannelVecType,
                                                           ComputeType,
                                                           hip_plugin_bn_config::lds_size>(
            dscale, dbias, hip_kernel_provider::cast<ComputeType>(1.0), ylid + zlid * ygrp_size);
    }

    if(ylid == 0 && zlid == 0)
    {
        const unsigned int stash_index = hip_plugin_bn_config::use_saved ? 0 : 2;
        hip_kernel_provider::batchnorm::storeToStash(
            dscale,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            stash_index,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
        hip_kernel_provider::batchnorm::storeToStash(
            dbias,
            reinterpret_cast<InputChannelVecType*>(meanvarbuff),
            stash_index + 1,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim_final.bounds)
    BatchNormBwdSpatialFinalDScaleDBias(
        const InputType* __restrict meanvarbuff, // meanvarbuff uses the dx buffer as a workspace, so InputType instead of MeanVarType
        ScaleBiasType* __restrict dscale,
        ScaleBiasType* __restrict dbias)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_size = blockDim.x;
    unsigned int ygrp_size = blockDim.y;
    unsigned int zgrp_size = blockDim.z;

    constexpr unsigned int xstride
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
    constexpr unsigned int ystride
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;
    constexpr unsigned int stash_index = hip_plugin_bn_config::use_saved ? 0 : 2;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
    {
        return;
    }

    auto pdscale = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);
    auto pdbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    for(unsigned int zoffset = zlid; zoffset < hip_plugin_bn_config::launch_groups.grp2;
        zoffset += zgrp_size)
    {
        for(unsigned int yoffset = ylid; yoffset < hip_plugin_bn_config::launch_groups.grp1;
            yoffset += ygrp_size)
        {
            pdscale += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                stash_index,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
            pdbias += hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
                reinterpret_cast<const InputChannelVecType*>(meanvarbuff),
                stash_index + 1,
                hip_plugin_bn_config::launch_dim.grp2 * zoffset
                    * hip_plugin_bn_config::batch_elements,
                hip_plugin_bn_config::launch_dim.grp1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_size,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ ComputeChannelVecType
            lcl_data[2 * hip_plugin_bn_config::launch_dim_final.bounds];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(
            pdscale,
            pdbias,
            hip_kernel_provider::cast<ComputeType>(1.0),
            lcl_data,
            xgrp_size,
            xlid,
            ylid + zlid * ygrp_size,
            ygrp_size * zgrp_size);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<
            ComputeChannelVecType,
            ComputeType,
            hip_plugin_bn_config::launch_dim_final.bounds>(
            pdscale, pdbias, hip_kernel_provider::cast<ComputeType>(1.0), ylid + zlid * ygrp_size);
    }

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<ScaleBiasChannelVecType*>(dscale)[xgid]
            = hip_kernel_provider::cast<ScaleBiasChannelVecType>(pdscale);
        reinterpret_cast<ScaleBiasChannelVecType*>(dbias)[xgid]
            = hip_kernel_provider::cast<ScaleBiasChannelVecType>(pdbias);
    }
}

extern "C" __global__ void __launch_bounds__(hip_plugin_bn_config::launch_dim.bounds)
    BatchNormBwdSpatialDX(const InputType* __restrict x_in,
                          const OutputType* __restrict dy_in,
                          InputType* __restrict dx_out,
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
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;

    constexpr unsigned int xstride
        = hip_plugin_config::layout_channel_last ? 1 : hip_plugin_bn_config::spatial_size;
    constexpr unsigned int ystride
        = hip_plugin_config::layout_channel_last ? hip_plugin_bn_config::channels : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::channels)
    {
        return;
    }

    ComputeChannelVecType mean, invVar;
    ComputeChannelVecType pscale, pdscale, pdbias;
    auto pbias = hip_kernel_provider::cast<ComputeChannelVecType>(0.0);

    __shared__ ComputeChannelVecType lscale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType ldscale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType ldbias[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType lmean[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ ComputeChannelVecType livar[hip_plugin_bn_config::launch_dim.grp0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ ComputeChannelVecType lbias[hip_plugin_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_size = blockDim.x;
        unsigned int ygrp_size = blockDim.y;
        unsigned int zgrp_size = blockDim.z;

        lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(dx_out),
            0,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<ComputeChannelVecType>(
            reinterpret_cast<const InputChannelVecType*>(dx_out),
            1,
            zgrp_size * zgrp_id * hip_plugin_bn_config::batch_elements,
            ygrp_size * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_size,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedMean)[xgid]);
        livar[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const MeanVarChannelVecType*>(savedInvVariance)[xgid]);
#endif
        lscale[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(bnScale)[xgid]);
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        lbias[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(bnBias)[xgid]);
#endif
        ldscale[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(dscale)[xgid]);
        ldbias[xlid] = static_cast<ComputeChannelVecType>(
            reinterpret_cast<const ScaleBiasChannelVecType*>(dbias)[xgid]);
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::spatial_size
       && zgid < hip_plugin_bn_config::batches)
    {
        mean = lmean[xlid];
        invVar = livar[xlid];
        pscale = lscale[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = lbias[xlid];
#endif
        pdscale = ldscale[xlid];
        pdbias = ldbias[xlid];

        unsigned int index_base
            = (zgid * hip_plugin_bn_config::batch_elements) * hip_plugin_bn_config::batch_size
              + ygid * ystride * hip_plugin_bn_config::vec_size_y
              + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int batch = 0; batch < hip_plugin_bn_config::batch_elements; batch++)
        { // apply normalization
            unsigned int index = index_base + batch * hip_plugin_bn_config::batch_size;
            auto x_i = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const InputVecType*>(x_in + index));
            ComputeVecType xhat = (x_i - mean) * invVar; // recalculating this again...
            auto value1 = hip_kernel_provider::cast<ComputeVecType>(
                *reinterpret_cast<const OutputVecType*>(dy_in + index));
            value1
                = hip_kernel_provider::batchnorm::bwd_activation_op<ComputeVecType,
                                                                    hip_plugin_config::neuron_op>(
                    value1,
                    xhat,
                    hip_kernel_provider::cast<ComputeVecType>(pscale),
                    hip_kernel_provider::cast<ComputeVecType>(pbias),
                    hip_kernel_provider::cast<ComputeVecType>(alpha),
                    hip_kernel_provider::cast<ComputeVecType>(beta));

            *reinterpret_cast<InputVecType*>(dx_out + index)
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
