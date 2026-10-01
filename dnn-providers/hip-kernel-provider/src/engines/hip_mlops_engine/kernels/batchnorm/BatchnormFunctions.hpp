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
__forceinline__ __device__ __host__ void accumulate2(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void accumulate4(T1& a, T2 const& b)
{
    a += cast<T1>(b.x);
    a += cast<T1>(b.y);
    a += cast<T1>(b.z);
    a += cast<T1>(b.w);
}

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ __host__ void accumulateMad2(T1& a, T2 const& b, T3 const& c)
{
    a = fma(cast<T1>(b.x), cast<T1>(c.x), a);
    a = fma(cast<T1>(b.y), cast<T1>(c.y), a);
}

template <typename T1, typename T2, typename T3>
__forceinline__ __device__ __host__ void accumulateMad4(T1& a, T2 const& b, T3 const& c)
{
    a = fma(cast<T1>(b.x), cast<T1>(c.x), a);
    a = fma(cast<T1>(b.y), cast<T1>(c.y), a);
    a = fma(cast<T1>(b.z), cast<T1>(c.z), a);
    a = fma(cast<T1>(b.w), cast<T1>(c.w), a);
}

template <typename T1, typename T2>
__forceinline__ __device__ __host__ void accumulate8(T1& a, T2 const& b)
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
__forceinline__ __device__ __host__ void accumulateMad8(T1& a, T2 const& b, T3 const& c)
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
__forceinline__ __device__ __host__ void accumulateMad(TAccum& a, T const& b, T const& c)
{
    constexpr auto TACCUM_SIZE = MappedVectorInfo<TAccum>::size;
    constexpr auto T_SIZE = MappedVectorInfo<T>::size;

    if constexpr(T_SIZE == TACCUM_SIZE)
    {
        a = fma(b, c, a);
    }
    else if constexpr(TACCUM_SIZE == 1 && T_SIZE == 2)
    {
        detail::accumulateMad2(a, b, c);
    }
    else if constexpr(TACCUM_SIZE == 1 && T_SIZE == 4)
    {
        detail::accumulateMad4(a, b, c);
    }
    else if constexpr(TACCUM_SIZE == 1 && T_SIZE == 8)
    {
        detail::accumulateMad8(a, b, c);
    }
    else
    {
        static_assert(false, "Invalid input types for accumulateMad.");
    }
}

template <typename TAccum, typename T>
__forceinline__ __device__ __host__ void accumulate(TAccum& a, T const& b)
{
    constexpr auto TACCUM_SIZE = MappedVectorInfo<TAccum>::size;
    constexpr auto T_SIZE = MappedVectorInfo<T>::size;

    if constexpr(TACCUM_SIZE == 1 && T_SIZE == 8)
    {
        detail::accumulate8(a, b);
    }
    else if constexpr(TACCUM_SIZE == 1 && T_SIZE == 4)
    {
        detail::accumulate4(a, b);
    }
    else if constexpr(TACCUM_SIZE == 1 && T_SIZE == 2)
    {
        detail::accumulate2(a, b);
    }
    else if constexpr(TACCUM_SIZE == T_SIZE)
    {
        a += cast<TAccum>(b);
    }
    else
    {
        static_assert(false, "Invalid input types for accumulate.");
    }
}

__forceinline__ __device__ unsigned int getStashIndex(unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrpSz,
                                                      unsigned int xgrpId,
                                                      unsigned int xlid,
                                                      unsigned int xstride,
                                                      unsigned int nstride)
{
    // NOLINTBEGIN(bugprone-branch-clone)
    if constexpr(HIP_PLUGIN_USE_FPMIX)
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::LAYOUT_NHWC)
        {
            if constexpr(config::C % 2 == 0)
            {
                // xgrp_sz values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C and xgrp_sz is that they
                // must be even.
                return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                       + (vindex * 2 + xlid % 2) * nstride + ygroupoffset * ystride
                       + (xgrpSz * xgrpId + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                       + (vindex * 2) * nstride + ygroupoffset * ystride
                       + (xgrpSz * xgrpId + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                   + ((vindex * 2) * nstride + ygroupoffset * ystride
                      + (xgrpSz * xgrpId + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else if constexpr(HIP_PLUGIN_USE_BFPMIX)
    {
        // 2 _FLOAT values are used to store 1 _FLOAT_PREC value.
        if constexpr(hip_kernel_provider::config::LAYOUT_NHWC)
        {
            if constexpr(config::C % 2 == 0)
            {
                // xgrp_sz values are split in two parts: even threads use 2 values at even rows,
                // odd threads - at odd rows. The only restriction for C and xgrp_sz is that they
                // must be even.
                return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                       + (vindex * 2 + xlid % 2) * nstride + ygroupoffset * ystride
                       + (xgrpSz * xgrpId + xlid / 2 * 2) * xstride;
            }
            else
            {
                // Values are stored consecutively in y dim.
                return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                       + (vindex * 2) * nstride + ygroupoffset * ystride
                       + (xgrpSz * xgrpId + xlid) * xstride;
            }
        }
        else
        {
            // Values are stored consecutively in y dim, indices are aligned up by 2 (_FLOAT_PREC).
            return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW)
                   + ((vindex * 2) * nstride + ygroupoffset * ystride
                      + (xgrpSz * xgrpId + xlid) * xstride + 1)
                         / 2 * 2;
        }
    }
    else
    {
        return zgroupoffset * (config::C / config::VEC_SIZE_X * config::HW) + vindex * nstride
               + ygroupoffset * ystride + (xgrpSz * xgrpId + xlid) * xstride;
    }
    // NOLINTEND(bugprone-branch-clone)
}

template <typename FpPrecType_C, typename FpType_C>
__forceinline__ __device__ FpPrecType_C loadFromStash(const FpType_C* stash,
                                                      unsigned int vindex,
                                                      unsigned int zgroupoffset,
                                                      unsigned int ygroupoffset,
                                                      unsigned int ystride,
                                                      unsigned int xgrpSz,
                                                      unsigned int xgrpId,
                                                      unsigned int xlid,
                                                      unsigned int xstride)
{
    unsigned int nstride
        = config::STASH_METHOD == 0 ? ystride : config::C / config::VEC_SIZE_X * config::HW;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrpSz, xgrpId, xlid, xstride, nstride);

    if constexpr(config::STASH_METHOD == 0 || config::STASH_METHOD == 1)
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
                                             unsigned int xgrpSz,
                                             unsigned int xgrpId,
                                             unsigned int xlid,
                                             unsigned int xstride)
{
    unsigned int nstride
        = config::STASH_METHOD == 0 ? ystride : config::C / config::VEC_SIZE_X * config::HW;

    unsigned int index = getStashIndex(
        vindex, zgroupoffset, ygroupoffset, ystride, xgrpSz, xgrpId, xlid, xstride, nstride);

    if constexpr(config::STASH_METHOD == 0 || config::STASH_METHOD == 1)
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
            = (config::NHW == 1)
                  ? variance
                  : variance
                        * (cast<FpAccumType_C>(config::NHW) / cast<FpAccumType_C>(config::NHW - 1));

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
            = (config::N == 1)
                  ? variance
                  : variance
                        * (cast<FpAccumType_C>(config::N) / cast<FpAccumType_C>(config::N - 1));

        runningVariance
            = fma(cast<FpAccumType_C>(1.0) - expAvgFactor, runningVariance, expAvgFactor * adjust);
    }
};

template <typename FpAccumType_C, typename FpPrecType_C>
__forceinline__ __device__ void savedStash(FpPrecType_C* __restrict resultSaveMean,
                                           FpPrecType_C* __restrict resultSaveInvVariance,
                                           FpAccumType_C mean,
                                           FpAccumType_C invVariance,
                                           unsigned int channel)
{
    resultSaveMean[channel] = cast<FpPrecType_C>(mean);
    resultSaveInvVariance[channel] = cast<FpPrecType_C>(invVariance);
}

template <typename FpAccumType_C, typename FpPrecType_C, typename Updater>
__forceinline__ __device__ void runningStash(const FpPrecType_C* __restrict prevRunningMean,
                                             const FpPrecType_C* __restrict prevRunningVariance,
                                             FpPrecType_C* __restrict nextRunningMean,
                                             FpPrecType_C* __restrict nextRunningVariance,
                                             Updater const& update,
                                             unsigned int channel)
{
    // Variant 4 is not used any more. There used to be a special updater for that case deleted when
    // porting kernels to HIP.
    static_assert(hip_kernel_provider::batchnorm::config::VARIANT != 4,
                  "running_stash is only compiled when HIP_PLUGIN_BN_VARIANT != 4.");

    auto pvtRunMean = cast<FpAccumType_C>(prevRunningMean[channel]);
    auto pvtRunVariance = cast<FpAccumType_C>(prevRunningVariance[channel]);

    update(pvtRunMean, pvtRunVariance);

    savedStash(nextRunningMean, nextRunningVariance, pvtRunMean, pvtRunVariance, channel);
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::PASTHRU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activationOp(FpPrecType const& tmp,
                                                            FpPrecType const& /* unused */,
                                                            FpPrecType const& /* unused */)
{
    return tmp;
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::PASTHRU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwdActivationOp(FpPrecType const& dy,
                                                               FpPrecType const& /* unused */,
                                                               FpPrecType const& /* unused */,
                                                               FpPrecType const& /* unused */,
                                                               FpPrecType const& /* unused */,
                                                               FpPrecType const& /* unused */)
{
    return dy;
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::RELU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activationOp(FpPrecType const& tmp,
                                                            FpPrecType const& /* unused */,
                                                            FpPrecType const& /* unused */)
{
    return max(tmp, hip_kernel_provider::cast<FpPrecType>(0.));
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::RELU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwdActivationOp(FpPrecType const& dy,
                                                               FpPrecType const& xnorm,
                                                               FpPrecType const& scale,
                                                               FpPrecType const& bias,
                                                               FpPrecType const& /* unused */,
                                                               FpPrecType const& /* unused */)
{
    FpPrecType macroTmp = scale * xnorm + bias;
    return (macroTmp > 0) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::CLIPPED_RELU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activationOp(FpPrecType const& tmp,
                                                            FpPrecType const& alpha,
                                                            FpPrecType const& /* unused */)
{
    return min(alpha, max(tmp, hip_kernel_provider::cast<FpPrecType>(0.)));
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::CLIPPED_RELU>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwdActivationOp(FpPrecType const& dy,
                                                               FpPrecType const& xnorm,
                                                               FpPrecType const& scale,
                                                               FpPrecType const& bias,
                                                               FpPrecType const& alpha,
                                                               FpPrecType const& /*unused*/)
{
    FpPrecType macroTmp = scale * xnorm + bias;
    return (macroTmp > 0 && macroTmp <= alpha) ? dy : 0;
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::CLAMP>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType activationOp(FpPrecType const& tmp,
                                                            FpPrecType const& alpha,
                                                            FpPrecType const& beta)
{
    return hip_kernel_provider::max(alpha, hip_kernel_provider::min(beta, tmp));
}

template <typename FpPrecType,
          hip_kernel_provider::NeuronOpType NrnOpType = hip_kernel_provider::config::NEURON_OP,
          typename std::enable_if<NrnOpType == NeuronOpType::CLAMP>::type* = nullptr>
__forceinline__ __host__ __device__ FpPrecType bwdActivationOp(FpPrecType const& dy,
                                                               FpPrecType const& xnorm,
                                                               FpPrecType const& scale,
                                                               FpPrecType const& bias,
                                                               FpPrecType const& alpha,
                                                               FpPrecType const& beta)
{
    FpPrecType macroTmp = scale * xnorm + bias;
    return (macroTmp > alpha && macroTmp <= beta) ? dy : 0;
}

} // namespace batchnorm

} // namespace hip_kernel_provider
