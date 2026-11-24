// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "Configuration.hpp"
#include "HipKernelMath.hpp"
#include "ReductionFunctions.hpp"
#include "VectorTypes.hpp"

// NOTE on bf16 rounding: narrowing casts to __bf16 in these kernels go through
// hip_kernel_provider::cast<T> (VectorTypes.hpp), which does a plain static_cast with no custom
// rounding logic - the actual rounding mode is whatever the compiler's native float->__bf16
// conversion implements. MIOpen explicitly passes -DMIOPEN_USE_RNE_BFLOAT16 for its BatchNorm
// kernel builds (PR #13998) so its device conversion round-to-nearest-even (RNE) matches its CPU
// reference, since a plain truncating conversion would otherwise be a real mismatch.
//
// Empirically verified on this provider's actual mechanism (static_cast<__bf16>(float), no
// compile-time fast-math/ffp-contract flags are set for these kernels) on gfx90a: the compiler's
// native conversion already performs RNE, not truncation - confirmed against MIOpen's
// float_to_bfloat16 RNE bit-manipulation oracle (bfloat16_dev.hpp) across round-half-to-even tie
// cases (including a mean~1000 case matching this file's Welford fix), 0 mismatches. This matches
// hipDNN's CPU reference (hipdnn_data_sdk::types::bfloat16, which defaults to RNE). So no RNE
// compile-define/kernel change is needed here; if this provider's minimum compiler version or
// -ffast-math-style flags ever change, re-verify this empirically rather than assuming it still
// holds, since native-cast rounding behavior isn't guaranteed by the language standard. Verified
// only on gfx90a - gfx942/gfx1201 are presumed to use the same codegen path but are unconfirmed.
namespace hip_kernel_provider::batchnorm
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

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ unsigned int getStashIndex(unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_size,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride,
                                                      unsigned int batchStride)
{
    // For upcast, two _FLOAT values can only be stored in one _FLOAT_PREC value if _FLOAT is not the same type as _FLOAT_PREC (i.e. only during backwards)
    if constexpr(hip_kernel_provider::config::input_type_strategy == type_strategy::fpmix
                 || (hip_kernel_provider::config::input_type_strategy == type_strategy::fpupcast
                     && !std::is_same_v<FpPrecType_C, FpType_C>))
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::layout_channel_last)
        {
            if constexpr((config::channels / config::vec_size_x) % 2 == 0)
            {
                // xgrp_size values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C / vec_size_x and xgrp_size
                // is that they must be even. Must match the parity test in getStashMethod.
                return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                       + (vindex * 2 + xlid % 2) * batchStride + ygroupoffset * ystride
                       + (xgrp_size * xgrp_id + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                       + (vindex * 2) * batchStride + ygroupoffset * ystride
                       + (xgrp_size * xgrp_id + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                   + ((vindex * 2) * batchStride + ygroupoffset * ystride
                      + (xgrp_size * xgrp_id + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else if constexpr(hip_kernel_provider::config::input_type_strategy == type_strategy::bfpmix
                      || (hip_kernel_provider::config::input_type_strategy
                              == type_strategy::bfpupcast
                          && !std::is_same_v<FpPrecType_C, FpType_C>))
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::layout_channel_last)
        {
            if constexpr((config::channels / config::vec_size_x) % 2 == 0)
            {
                // xgrp_size values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C / vec_size_x and xgrp_size
                // is that they must be even. Must match the parity test in getStashMethod.
                return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                       + (vindex * 2 + xlid % 2) * batchStride + ygroupoffset * ystride
                       + (xgrp_size * xgrp_id + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                       + (vindex * 2) * batchStride + ygroupoffset * ystride
                       + (xgrp_size * xgrp_id + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
                   + ((vindex * 2) * batchStride + ygroupoffset * ystride
                      + (xgrp_size * xgrp_id + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else
    {
        return zgroupoffset * (config::channels / config::vec_size_x * config::spatial_size)
               + vindex * batchStride + ygroupoffset * ystride
               + (xgrp_size * xgrp_id + xlid) * xstride;
    }
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ FpPrecType_C loadFromStash(const FpType_C* stash,
                                                      unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrp_size,
                                                      unsigned int xgrp_id,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{
    unsigned int batchStride = config::stash_method == 0
                                   ? ystride
                                   : config::channels / config::vec_size_x * config::spatial_size;

    unsigned int index = getStashIndex<FpPrecType_C, FpType_C>(vindex,
                                                               zgroupoffset,
                                                               ygroupoffset,
                                                               ystride,
                                                               xgrp_size,
                                                               xgrp_id,
                                                               xlid,
                                                               xstride,
                                                               batchStride);

    if constexpr(config::stash_method == 0 || config::stash_method == 1)
    {
        return *((const FpPrecType_C*)(stash + index));
    }
    else
    {
        FpPrecType_C value;
        *(reinterpret_cast<FpType_C*>(&value)) = stash[index];
        *(reinterpret_cast<FpType_C*>(&value) + 1) = stash[index + batchStride];
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
                                             unsigned int xgrp_size,
                                             unsigned int xgrp_id,
                                             unsigned int xlid,
                                             unsigned int xstride)
{
    unsigned int batchStride = config::stash_method == 0
                                   ? ystride
                                   : config::channels / config::vec_size_x * config::spatial_size;

    unsigned int index = getStashIndex<FpPrecType_C, FpType_C>(vindex,
                                                               zgroupoffset,
                                                               ygroupoffset,
                                                               ystride,
                                                               xgrp_size,
                                                               xgrp_id,
                                                               xlid,
                                                               xstride,
                                                               batchStride);

    if constexpr(config::stash_method == 0 || config::stash_method == 1)
    {
        *(reinterpret_cast<FpPrecType_C*>(stash + index)) = value;
    }
    else
    {
        stash[index] = *(reinterpret_cast<FpType_C*>(&value));
        stash[index + batchStride] = *(reinterpret_cast<FpType_C*>(&value) + 1);
    }
}

#if(HIP_PLUGIN_BN_VARIANT == 2)
namespace detail
{
constexpr unsigned int statsXStride
    = hip_kernel_provider::config::layout_channel_last ? 1 : config::spatial_size;
constexpr unsigned int statsYStride
    = hip_kernel_provider::config::layout_channel_last ? config::channels : 1;
constexpr unsigned int statsTileY = config::launch_dim.grp1 * config::vec_size_y;
constexpr unsigned int statsTileZ = config::launch_dim.grp2 * config::batch_elements;
} // namespace detail

// The output buffer already provides two FP32 slots per channel and tile. Store the tile's
// mean and M2 there; the final kernel derives its count from the tile bounds, including tails.
template <typename StashType>
__forceinline__ __device__ void stash_partial_stats(const config::input_type* __restrict in,
                                                    StashType* __restrict stash)
{
    using StashVec = typename mapped_vector_type<StashType, config::vec_size_x>::type;
    using InputVec = config::input_vec_type;
    using ComputeVec = config::compute_vec_type;
    constexpr auto dim = config::launch_dim;
    const unsigned int x = (blockIdx.x * dim.grp0 + threadIdx.x) * config::vec_size_x;
    const unsigned int y = (blockIdx.y * dim.grp1 + threadIdx.y) * config::vec_size_y;
    const unsigned int z = (blockIdx.z * dim.grp2 + threadIdx.z) * config::batch_elements;
    const bool validChannel = x < config::channels;

    reduction::Welford<config::compute_channel_vec_type> stats{};
    if(validChannel && y < config::spatial_size && z < config::batches)
    {
        const unsigned int base
            = x * detail::statsXStride + y * detail::statsYStride + z * config::batch_size;
        for(unsigned int batch = 0; batch < config::batch_elements; ++batch)
        {
            if constexpr(config::batches % config::batch_elements != 0)
            {
                if(z + batch >= config::batches)
                {
                    break;
                }
            }
            const unsigned int index = base + batch * config::batch_size;
            const auto sample = *reinterpret_cast<const InputVec*>(in + index);
            stats.add(cast<ComputeVec>(sample));
        }
    }
    reduction::reduce_welford_2d<dim.grp0, dim.bounds>(
        stats, threadIdx.x, threadIdx.y + threadIdx.z * dim.grp1);

    if(validChannel && threadIdx.y == 0 && threadIdx.z == 0)
    {
        auto store = [&](auto value, unsigned int slot) {
            storeToStash(value,
                         reinterpret_cast<StashVec*>(stash),
                         slot,
                         blockIdx.z * detail::statsTileZ,
                         blockIdx.y * detail::statsTileY,
                         detail::statsYStride / config::vec_size_x,
                         dim.grp0,
                         blockIdx.x,
                         threadIdx.x,
                         detail::statsXStride);
        };
        store(stats.mean, 0);
        store(stats.m2, 1);
    }
}

template <typename StashType>
__forceinline__ __device__ auto finalize_stashed_stats(StashType* stash, double epsilon)
{
    using StashVec = typename mapped_vector_type<StashType, config::vec_size_x>::type;
    using ComputeVec = config::compute_channel_vec_type;
    constexpr auto dim = config::launch_dim_final;
    const bool validChannel
        = (blockIdx.x * dim.grp0 + threadIdx.x) * config::vec_size_x < config::channels;
    reduction::Welford<ComputeVec> stats{};
    if(validChannel)
    {
        for(unsigned int zGroup = threadIdx.z; zGroup < config::launch_groups.grp2;
            zGroup += dim.grp2)
        {
            const unsigned int z = zGroup * detail::statsTileZ;
            const unsigned int batchCount = detail::statsTileZ < config::batches - z
                                                ? detail::statsTileZ
                                                : config::batches - z;
            for(unsigned int yGroup = threadIdx.y; yGroup < config::launch_groups.grp1;
                yGroup += dim.grp1)
            {
                const unsigned int y = yGroup * detail::statsTileY;
                const unsigned int spatialCount = detail::statsTileY < config::spatial_size - y
                                                      ? detail::statsTileY
                                                      : config::spatial_size - y;
                auto load = [&](unsigned int slot) {
                    return loadFromStash<ComputeVec>(reinterpret_cast<const StashVec*>(stash),
                                                     slot,
                                                     z,
                                                     y,
                                                     detail::statsYStride / config::vec_size_x,
                                                     dim.grp0,
                                                     blockIdx.x,
                                                     threadIdx.x,
                                                     detail::statsXStride);
                };
                stats.merge(load(0), load(1), static_cast<float>(batchCount * spatialCount));
            }
        }
    }

    reduction::reduce_welford_2d<dim.grp0, dim.bounds>(
        stats, threadIdx.x, threadIdx.y + threadIdx.z * dim.grp1);

    const ComputeVec variance = stats.m2 * config::i_channel_slice_size;
    const ComputeVec invVariance = rsqrt(variance + cast<ComputeVec>(epsilon));
    if(validChannel)
    {
        // The normalization kernel reads the final statistics from its own tile, then
        // overwrites the output buffer. Do not publish them until all partials were read.
        for(unsigned int zGroup = threadIdx.z; zGroup < config::launch_groups.grp2;
            zGroup += dim.grp2)
        {
            for(unsigned int yGroup = threadIdx.y; yGroup < config::launch_groups.grp1;
                yGroup += dim.grp1)
            {
                auto store = [&](ComputeVec value, unsigned int slot) {
                    storeToStash(value,
                                 reinterpret_cast<StashVec*>(stash),
                                 slot,
                                 zGroup * detail::statsTileZ,
                                 yGroup * detail::statsTileY,
                                 detail::statsYStride / config::vec_size_x,
                                 dim.grp0,
                                 blockIdx.x,
                                 threadIdx.x,
                                 detail::statsXStride);
                };
                store(stats.mean, 0);
                store(invVariance, 1);
            }
        }
    }
    struct FinalStats
    {
        ComputeVec mean;
        ComputeVec variance;
        ComputeVec invVariance;
    };
    return FinalStats{stats.mean, variance, invVariance};
}
#endif

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
            = (config::channel_slice_size == 1)
                  ? variance
                  : variance
                        * (cast<FpAccumType_C>(config::channel_slice_size)
                           / cast<FpAccumType_C>(config::channel_slice_size - 1));

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

        const FpAccumType_C adjust = (config::batches == 1)
                                         ? variance
                                         : variance
                                               * (cast<FpAccumType_C>(config::batches)
                                                  / cast<FpAccumType_C>(config::batches - 1));

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
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::PASTHRU>* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return tmp;
}

template <typename FpPrecType,
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::PASTHRU>* = nullptr>
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
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::RELU>* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const&,
                                                             FpPrecType const&)
{
    return max(tmp, hip_kernel_provider::cast<FpPrecType>(0.));
}

template <typename FpPrecType,
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::RELU>* = nullptr>
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
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::CLIPPED_RELU>* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const&)
{
    return min(_alpha, max(tmp, hip_kernel_provider::cast<FpPrecType>(0.)));
}

template <typename FpPrecType,
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::CLIPPED_RELU>* = nullptr>
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
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::CLAMP>* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activation_op(FpPrecType const& tmp,
                                                             FpPrecType const& _alpha,
                                                             FpPrecType const& _beta)
{
    return hip_kernel_provider::max(_alpha, hip_kernel_provider::min(_beta, tmp));
}

template <typename FpPrecType,
          hip_kernel_provider::ActivationMode NrnOpType = hip_kernel_provider::config::neuron_op,
          typename std::enable_if_t<NrnOpType == ActivationMode::CLAMP>* = nullptr>
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

} // namespace hip_kernel_provider::batchnorm
