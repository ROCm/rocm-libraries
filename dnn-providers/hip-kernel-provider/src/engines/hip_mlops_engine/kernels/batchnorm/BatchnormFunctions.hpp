// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Configuration.hpp"
#include "HipKernelMath.hpp"
#include "VectorTypes.hpp"

namespace hip_kernel_provider
{

namespace batchnorm
{

namespace detail
{

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate2(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate4(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
    a += cast<T1>(b.z);
    a += cast<T1>(b.w);
}

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ __host__ void _accumulate_mad2(T1& a, T2 const& b, T3 const& c)
{
    a = fma(cast<T1>(b.x), cast<T1>(c.x), a);
    a = fma(cast<T1>(b.y), cast<T1>(c.y), a);
}

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ __host__ void _accumulate_mad4(T1& a, T2 const& b, T3 const& c)
{
    a = fma(cast<T1>(b.x), cast<T1>(c.x), a);
    a = fma(cast<T1>(b.y), cast<T1>(c.y), a);
    a = fma(cast<T1>(b.z), cast<T1>(c.z), a);
    a = fma(cast<T1>(b.w), cast<T1>(c.w), a);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void _accumulate8(T1& a, T2 const& b)
{
    a += cast<T1>(b.s0);
    a += cast<T1>(b.s1);
    a += cast<T1>(b.s2);
    a += cast<T1>(b.s3);
    a += cast<T1>(b.s4);
    a += cast<T1>(b.s5);
    a += cast<T1>(b.s6);
    a += cast<T1>(b.s7);
}

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ __host__ void _accumulate_mad8(T1& a, T2 const& b, T3 const& c)
{
    a = fma(cast<T1>(b.s0), cast<T1>(c.s0), a);
    a = fma(cast<T1>(b.s1), cast<T1>(c.s1), a);
    a = fma(cast<T1>(b.s2), cast<T1>(c.s2), a);
    a = fma(cast<T1>(b.s3), cast<T1>(c.s3), a);
    a = fma(cast<T1>(b.s4), cast<T1>(c.s4), a);
    a = fma(cast<T1>(b.s5), cast<T1>(c.s5), a);
    a = fma(cast<T1>(b.s6), cast<T1>(c.s6), a);
    a = fma(cast<T1>(b.s7), cast<T1>(c.s7), a);
}
} // namespace detail

template <typename TAccum, typename T>
__forceinline__ __device__ __host__ void _accumulate_mad(TAccum& a, T const& b, T const& c)
{
    constexpr auto TaccumSize = mapped_vector_info<TAccum>::size;
    constexpr auto TSize = mapped_vector_info<T>::size;

    if constexpr(TSize == TaccumSize)
    {
        a = fma(b, c, a);
    }
    else if constexpr(TaccumSize == 1 && TSize == 2)
    {
        detail::_accumulate_mad2(a, b, c);
    }
    else if constexpr(TaccumSize == 1 && TSize == 4)
    {
        detail::_accumulate_mad4(a, b, c);
    }
    else if constexpr(TaccumSize == 1 && TSize == 8)
    {
        detail::_accumulate_mad8(a, b, c);
    }
    else
    {
        static_assert(false, "Invalid input types for _accumulate_mad.");
    }
}

template <typename TAccum, typename T>
__forceinline__ __device__ __host__ void _accumulate(TAccum& a, T const& b)
{
    constexpr auto TaccumSize = mapped_vector_info<TAccum>::size;
    constexpr auto TSize = mapped_vector_info<T>::size;

    if constexpr(TaccumSize == 1 && TSize == 8)
    {
        detail::_accumulate8(a, b);
    }
    else if constexpr(TaccumSize == 1 && TSize == 4)
    {
        detail::_accumulate4(a, b);
    }
    else if constexpr(TaccumSize == 1 && TSize == 2)
    {
        detail::_accumulate2(a, b);
    }
    else if constexpr(TaccumSize == TSize)
    {
        a += cast<TAccum>(b);
    }
    else
    {
        static_assert(false, "Invalid input types for _accumulate.");
    }
}

__forceinline__ __device__ unsigned int getStashIndex(unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride,
                                                      unsigned int nstride)
{
    if constexpr(HIP_PLUGIN_USE_FPMIX)
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::layout_nhwc)
        {
            if constexpr(config::c % 2 == 0)
            {
                // xgrp_sz values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C and xgrp_sz is that they
                // must be even.
                return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                       + (vindex * 2 + xlid % 2) * nstride + ygroupoffset * ystride
                       + (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                       + (vindex * 2) * nstride + ygroupoffset * ystride
                       + (xgrp_sz * xgrp_id + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                   + ((vindex * 2) * nstride + ygroupoffset * ystride
                      + (xgrp_sz * xgrp_id + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else if constexpr(HIP_PLUGIN_USE_BFPMIX)
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::layout_nhwc)
        {
            if constexpr(config::c % 2 == 0)
            {
                // xgrp_sz values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C and xgrp_sz is that they
                // must be even.
                return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                       + (vindex * 2 + xlid % 2) * nstride + ygroupoffset * ystride
                       + (xgrp_sz * xgrp_id + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                       + (vindex * 2) * nstride + ygroupoffset * ystride
                       + (xgrp_sz * xgrp_id + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::c / config::vec_size_x * config::hw)
                   + ((vindex * 2) * nstride + ygroupoffset * ystride
                      + (xgrp_sz * xgrp_id + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else
    {
        return zgroupoffset * (config::c / config::vec_size_x * config::hw) + vindex * nstride
               + ygroupoffset * ystride + (xgrp_sz * xgrp_id + xlid) * xstride;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ FpPrecType_C loadFromStash(const FpType_C* stash,
                                                      unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_sz,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{
    unsigned int nstride
        = config::stash_method == 0 ? ystride : config::c / config::vec_size_x * config::hw;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride, nstride);

    if constexpr(config::stash_method == 0 || config::stash_method == 1)
    {
        return *((const FpPrecType_C*)(stash + index));
    }
    else
    {
        FpPrecType_C value;
        *(reinterpret_cast<FpType_C*>(&value)) = stash[index];
        *(reinterpret_cast<FpType_C*>(&value) + 1) = stash[index + nstride];
        return value;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ void storeToStash(FpPrecType_C value,
                                             FpType_C* stash,
                                             unsigned int vindex,
                                             unsigned int zgroupoffset,
                                             unsigned int ygroupoffset,
                                             unsigned int ystride,
                                             unsigned int xgrp_sz,
                                             unsigned int xgrp_id,
                                             unsigned int xlid,
                                             unsigned int xstride)
{
    unsigned int nstride
        = config::stash_method == 0 ? ystride : config::c / config::vec_size_x * config::hw;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrp_sz, xgrp_id, xlid, xstride, nstride);

    if constexpr(config::stash_method == 0 || config::stash_method == 1)
    {
        *(reinterpret_cast<config::fp_prec_c_type*>(stash + index)) = value;
    }
    else
    {
        stash[index] = *(reinterpret_cast<FpType_C*>(&value));
        stash[index + nstride] = *(reinterpret_cast<FpType_C*>(&value) + 1);
    }
}

template <typename FpAccumType_C>
struct StashUpdater
{
    FpAccumType_C const mean;
    FpAccumType_C const variance;
    FpAccumType_C const expAvgFactor;

    __device__ StashUpdater(FpAccumType_C m, FpAccumType_C v, FpAccumType_C e)
        : mean(m)
        , variance(v)
        , expAvgFactor(e)
    {
    }

    __forceinline__ __device__ void operator()(FpAccumType_C& runningMean,
                                               FpAccumType_C& runningVariance) const
    {
        const FpAccumType_C newRunningMean = fma(-expAvgFactor, runningMean, runningMean);
        runningMean = fma(mean, expAvgFactor, newRunningMean);

        const FpAccumType_C adjust
            = (config::nhw == 1)
                  ? variance
                  : variance
                        * (cast<FpAccumType_C>(config::nhw) / cast<FpAccumType_C>(config::nhw - 1));

        runningVariance
            = fma(cast<FpAccumType_C>(1.0) - expAvgFactor, runningVariance, expAvgFactor * adjust);
    }
};

template <typename FpAccumType_C>
struct StashUpdaterPA
{
    FpAccumType_C const mean;
    FpAccumType_C const variance;
    FpAccumType_C const expAvgFactor;

    __device__ StashUpdaterPA(FpAccumType_C m, FpAccumType_C v, FpAccumType_C e)
        : mean(m)
        , variance(v)
        , expAvgFactor(e)
    {
    }

    __forceinline__ __device__ void operator()(FpAccumType_C& runningMean,
                                               FpAccumType_C& runningVariance) const
    {
        const FpAccumType_C newRunningMean = fma(-expAvgFactor, runningMean, runningMean);
        runningMean = fma(mean, expAvgFactor, newRunningMean);

        const FpAccumType_C adjust
            = (config::n == 1)
                  ? variance
                  : variance
                        * (cast<FpAccumType_C>(config::n) / cast<FpAccumType_C>(config::n - 1));

        runningVariance
            = fma(cast<FpAccumType_C>(1.0) - expAvgFactor, runningVariance, expAvgFactor * adjust);
    }
};

template <typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void saved_stash(FpPrecType_C* __restrict resultSaveMean,
                                            FpPrecType_C* __restrict resultSaveInvVariance,
                                            FpAccumType_C mean,
                                            FpAccumType_C invVariance,
                                            unsigned int channel)
{
    resultSaveMean[channel] = cast<FpPrecType_C>(mean);
    resultSaveInvVariance[channel] = cast<FpPrecType_C>(invVariance);
}

template <typename FpAccumType_C, typename FpPrecType_C, typename Updater>
__forceinline__ __device__ void running_stash(const FpPrecType_C* __restrict prevRunningMean,
                                              const FpPrecType_C* __restrict prevRunningVariance,
                                              FpPrecType_C* __restrict nextRunningMean,
                                              FpPrecType_C* __restrict nextRunningVariance,
                                              Updater const& update,
                                              unsigned int channel)
{
    // Variant 4 is not used any more. There used to be a special updater for that case deleted when
    // porting kernels to HIP.
    static_assert(hip_kernel_provider::batchnorm::config::variant != 4,
                  "running_stash is only compiled when HIP_PLUGIN_BN_VARIANT != 4.");

    auto pvt_runMean = cast<FpAccumType_C>(prevRunningMean[channel]);
    auto pvt_runVariance = cast<FpAccumType_C>(prevRunningVariance[channel]);

    update(pvt_runMean, pvt_runVariance);

    saved_stash(nextRunningMean, nextRunningVariance, pvt_runMean, pvt_runVariance, channel);
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::pasthru>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return tmp;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::pasthru>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&)
{
    return dy;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return max(tmp, hip_kernel_provider::cast<FpPrecType>(0.));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const&,
                                                                 FpPrecType const&)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > 0) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clipped_relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const&)
{
    return min(_alpha, max(tmp, hip_kernel_provider::cast<FpPrecType>(0.)));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clipped_relu>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const& alpha,
                                                                 FpPrecType const&)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > 0 && macro_tmp <= alpha) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clamp>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const& _beta)
{
    return hip_kernel_provider::max(_alpha, hip_kernel_provider::min(_beta, tmp));
}

template <typename FpPrecType,
          hip_kernel_provider::neuron_op_type NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if<NrnOpType == neuron_op_type::clamp>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwd_activation_op(FpPrecType const& dy,
                                                                 FpPrecType const& xnorm,
                                                                 FpPrecType const& scale,
                                                                 FpPrecType const& bias,
                                                                 FpPrecType const& alpha,
                                                                 FpPrecType const& beta)
{
    FpPrecType macro_tmp = scale * xnorm + bias;
    return (macro_tmp > alpha && macro_tmp <= beta) ? dy : 0;
}

} // namespace batchnorm

} // namespace hip_kernel_provider
