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
} // namespace

namespace hip_kernel_provider
{

namespace batchnorm
{

template <int BnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormFwdTrainSpatialImpl
{
    static_assert(false, "this variant is not supported.");
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 0
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormFwdTrainSpatialImpl<0, FpType, FpPrecType, FpAccumType>
{
    // These are the configs for this variant
    static constexpr unsigned int SEGTMP_1
        = hip_plugin_bn_config::LAUNCH_DIM.GRP0 / hip_plugin_bn_config::HW;
    static constexpr unsigned int SEGTMP_2 = (SEGTMP_1 == 0) ? 1 : SEGTMP_1;
    static constexpr unsigned int SEGTMP = hip_plugin_bn_config::HW * SEGTMP_2;
    static constexpr unsigned int SEGMENT
        = (SEGTMP > hip_plugin_bn_config::NHW) ? hip_plugin_bn_config::NHW : SEGTMP;
    static constexpr unsigned int NLOOP = (hip_plugin_bn_config::NHW + SEGMENT - 1) / SEGMENT;
    static constexpr unsigned int SEGIHW = SEGMENT / hip_plugin_bn_config::HW;
    static constexpr unsigned int NLOOPM = NLOOP - 1;
    static constexpr unsigned int SNHW = NLOOPM * SEGIHW;

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType inhw,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        // SPATIAL
        mean = cast<FpAccumType>(0.);
        variance = cast<FpAccumType>(0.);
        invVariance = cast<FpAccumType>(0.);
        FpType batchvalues[NLOOP]; // NOLINT(modernize-avoid-c-arrays)
        FpAccumType temp;

        __shared__ FpPrecType s_lcl_bias;
        __shared__ FpPrecType s_lcl_scale;

        unsigned int index = 0;
        unsigned int lid = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        unsigned int grpid = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        unsigned int chwid = grpid * hip_plugin_bn_config::HW + (lid % hip_plugin_bn_config::HW);
        unsigned int lidihw = lid / hip_plugin_bn_config::HW;
        unsigned int nid = 0;

        if(lid == 0)
        {
            s_lcl_scale = *(scale + grpid);
            s_lcl_bias = *(bias + grpid);
        }

        __syncthreads();

        if(lid < SEGMENT)
        {
            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int UNROLL_HINT
                = hip_plugin_config::INPUT_TYPE_STRATEGY == TypeStrategy::FP16 ? 2 : 1;
            StaticUnrollCount<unsigned int, 0, NLOOPM, 1, UNROLL_HINT>{[&](unsigned int n) {
                nid = n * SEGIHW + lidihw;
                index = nid * hip_plugin_bn_config::CHW + chwid;
                batchvalues[n] = *(in + index);
                temp = cast<FpAccumType>(*(in + index));
                mean += temp;
                variance = fma(temp, temp, variance);
            }};
            nid = SNHW + lidihw;
            index = nid * hip_plugin_bn_config::CHW + chwid;
            batchvalues[NLOOPM]
                = (index < hip_plugin_bn_config::NCHW) ? (*(in + index)) : cast<FpType>(0.);
            temp = cast<FpAccumType>(batchvalues[NLOOPM]);
            mean += temp;
            variance = fma(temp, temp, variance);
        }
        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ FpAccumType s_lcl_data_x[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        __shared__ FpAccumType s_lcl_data_y[LCL_DATA_SIZE]; // NOLINT(modernize-avoid-c-arrays)
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        // Reduction complete

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = hip_kernel_provider::rsqrt(variance + cast<FpAccumType>(epsilon));

        FpAccumType pvscale = cast<FpAccumType>(s_lcl_scale);
        FpAccumType pvbias = cast<FpAccumType>(s_lcl_bias);

        if(lid < SEGMENT)
        {
            // Calculate norm

            FpAccumType inhat = cast<FpAccumType>(0.);
            FpPrecType value;

            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int UNROLL_HINT
                = hip_plugin_config::INPUT_TYPE_STRATEGY == TypeStrategy::FP16 ? 2 : 1;

            StaticUnrollCount<unsigned int, 0, NLOOPM, 1, UNROLL_HINT>{[&](unsigned int n) {
                // Apply normalization
                inhat = (cast<FpAccumType>(batchvalues[n]) - mean) * invVariance;
                nid = n * SEGIHW + lidihw;
                index = nid * hip_plugin_bn_config::CHW + chwid;
                value = cast<FpPrecType>(fma(pvscale, inhat, pvbias));
                out[index] = cast<FpType>(
                    hip_kernel_provider::applyActivation<FpPrecType,
                                                         hip_kernel_provider::ActivationMode{
                                                             HIP_PLUGIN_BN_NRN_OP_ID}>(
                        value, alpha, beta));
            }};

            // Tail of loop
            inhat = (cast<FpAccumType>(batchvalues[NLOOPM]) - mean) * invVariance;
            nid = SNHW + lidihw;
            index = nid * hip_plugin_bn_config::CHW + chwid;
            if(index < hip_plugin_bn_config::NCHW)
            {
                value = cast<FpPrecType>(fma(pvscale, inhat, pvbias));
                out[index] = cast<FpType>(
                    hip_kernel_provider::applyActivation<FpPrecType,
                                                         hip_kernel_provider::ActivationMode{
                                                             HIP_PLUGIN_BN_NRN_OP_ID}>(
                        value, alpha, beta));
            }
        }
    }
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 1
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormFwdTrainSpatialImpl<1, FpType, FpPrecType, FpAccumType>
{
    // These are the configs for this variant
    static constexpr unsigned int MAX_READ
        = hip_plugin_config::LAYOUT_NHWC ? 1 : (hip_plugin_bn_config::HW >= 4096 ? 3 : 2);
    static constexpr unsigned int RD_BLK = 1;
    static constexpr unsigned int GRPRD
        = hip_plugin_config::LAYOUT_NHWC ? (hip_plugin_bn_config::LAUNCH_DIM.GRP0 * RD_BLK)
                                         : (hip_plugin_bn_config::LAUNCH_DIM.GRP0 * RD_BLK * 4);
    static constexpr unsigned int REM4
        = hip_plugin_bn_config::NHW - ((hip_plugin_bn_config::NHW / GRPRD) * GRPRD);
    static constexpr unsigned int LESS4 = hip_plugin_bn_config::NHW - REM4;
    static constexpr unsigned int CHUNK4 = MAX_READ * GRPRD;
    static constexpr unsigned int REMOUT4
        = hip_plugin_bn_config::NHW - ((hip_plugin_bn_config::NHW / CHUNK4) * CHUNK4);
    static constexpr unsigned int LESSOUT4 = hip_plugin_bn_config::NHW - REMOUT4;
    static constexpr unsigned int REM
        = hip_plugin_bn_config::NHW
          - ((hip_plugin_bn_config::NHW / hip_plugin_bn_config::LAUNCH_DIM.GRP0)
             * hip_plugin_bn_config::LAUNCH_DIM.GRP0);
    static constexpr unsigned int LESS = hip_plugin_bn_config::NHW - REM;
    static constexpr unsigned int CHUNK = MAX_READ * hip_plugin_bn_config::LAUNCH_DIM.GRP0;
    static constexpr unsigned int REMOUT
        = hip_plugin_bn_config::NHW - ((hip_plugin_bn_config::NHW / CHUNK) * CHUNK);
    static constexpr unsigned int LESSOUT = hip_plugin_bn_config::NHW - REMOUT;

    // Kernel
    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType inhw,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType pvscale, pvbias;

        mean = 0;
        variance = 0;
        invVariance = 0;

        __shared__ FpPrecType s_lcl_bias;
        __shared__ FpPrecType s_lcl_scale;

        unsigned int index = 0;
        const unsigned int lid = threadIdx.x;
        const unsigned int grpid = blockIdx.x;

        // Note: this variable is only used when hip_plugin_config::LAYOUT_NHWC is false.
        unsigned int chwid;
        if constexpr(!hip_plugin_config::LAYOUT_NHWC)
        {
            chwid = grpid * hip_plugin_bn_config::HW;
        }

        unsigned int nidx = 0;
        unsigned int hwidx = 0;

        if(lid == 0)
        {
            s_lcl_scale = *(scale + grpid);
            s_lcl_bias = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!hip_plugin_config::LAYOUT_NHWC && hip_plugin_bn_config::HW >= 4096)
        {
            using fp_type4 = typename MappedVectorType<FpType, 4>::type;
            fp_type4 read4;

            StaticUnrollCount<unsigned int, 0, LESS4, GRPRD, 2>{[&](unsigned int k) {
                if((k + (lid << 2)) < LESS4)
                {
                    nidx = (k + (lid << 2)) / hip_plugin_bn_config::HW;
                    hwidx = (k + (lid << 2)) - (nidx * hip_plugin_bn_config::HW);
                    index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    hip_kernel_provider::batchnorm::accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                }
            }};

            if constexpr(REM4 > 0u)
            {
                const unsigned int remkey = (lid << 2) + LESS4;
                nidx = remkey / hip_plugin_bn_config::HW;
                hwidx = remkey - (nidx * hip_plugin_bn_config::HW);
                index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;

                // index is unsigned int, so if the result would normally end up negative,
                // the value wraps around and the check fails. Improves on the
                // previous way of handling which was: if(index < (hip_plugin_bn_config::NCHW - 3))
                if(index + 3 < (hip_plugin_bn_config::NCHW))
                {
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    hip_kernel_provider::batchnorm::accumulate(mean, read4);
                    hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                }
            }
        }
        else
        {
            StaticUnrollCount<unsigned int, 0, LESS, hip_plugin_bn_config::LAUNCH_DIM.GRP0, 4>{
                [&](unsigned int k) {
                    if(k + lid < LESS)
                    {
                        nidx = (k + lid) / hip_plugin_bn_config::HW;
                        hwidx = (k + lid) - (nidx * hip_plugin_bn_config::HW);
                        if constexpr(hip_plugin_config::LAYOUT_NHWC)
                        {
                            index = nidx * hip_plugin_bn_config::CHW
                                    + hwidx * hip_plugin_bn_config::C + grpid;
                        }
                        else
                        {
                            index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                        }
                        const auto xin = cast<FpPrecType>(in[index]);
                        mean += xin;
                        variance = fma(xin, xin, variance);
                    }
                }};

            if constexpr(REM > 0u)
            {
                // Note: hip compiler has a bug, it throws compiler warning for comparing unsigned
                // int with 0 value, when rem is 0. but when rem is 0, this code block should not be
                // compiled due to the if constexpr used above.
                if(lid < REM)
                {
                    unsigned int remkey = lid + LESS;
                    nidx = remkey / hip_plugin_bn_config::HW;
                    hwidx = remkey - (nidx * hip_plugin_bn_config::HW);
                    if constexpr(hip_plugin_config::LAYOUT_NHWC)
                    {
                        index = nidx * hip_plugin_bn_config::CHW + hwidx * hip_plugin_bn_config::C
                                + grpid;
                    }
                    else
                    {
                        index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                    }

                    const auto xin = index < hip_plugin_bn_config::NCHW
                                         ? cast<FpPrecType>(in[index])
                                         : FpPrecType{0};
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ FpAccumType s_lcl_data_x[LCL_DATA_SIZE];
        __shared__ FpAccumType s_lcl_data_y[LCL_DATA_SIZE];
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        // REDUCTION COMPLETE ---------------------------
        variance = fma(-mean, mean, variance);
        if(variance < FpPrecType{0})
        {
            variance = FpPrecType{0};
        }

        // unsafe: casting double to FpPrecType
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<FpPrecType>(epsilon));
        pvscale = s_lcl_scale;
        pvbias = s_lcl_bias;
        if constexpr(hip_plugin_config::LAYOUT_NHWC || REM == 0)
        {
            constexpr unsigned int K_LIMIT
                = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::NHW : LESS;

            StaticUnrollCount<unsigned int, 0, K_LIMIT, hip_plugin_bn_config::LAUNCH_DIM.GRP0, 2>{
                [&](unsigned int k) {
                    if(k + lid < K_LIMIT)
                    {
                        nidx = (k + lid) / hip_plugin_bn_config::HW;
                        hwidx = (k + lid) - (nidx * hip_plugin_bn_config::HW);
                        if constexpr(hip_plugin_config::LAYOUT_NHWC)
                        {
                            index = nidx * hip_plugin_bn_config::CHW
                                    + hwidx * hip_plugin_bn_config::C + grpid;
                        }
                        else
                        {
                            index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                        }

                        out[index] = cast<FpType>(
                            hip_kernel_provider::applyActivation<
                                FpPrecType,
                                hip_kernel_provider::ActivationMode{HIP_PLUGIN_BN_NRN_OP_ID}>(
                                fma(pvscale,
                                    (cast<FpPrecType>(in[index]) - mean) * invVariance,
                                    pvbias),
                                alpha,
                                beta));
                    }
                }};
        }
        else
        {
            FpPrecType xhat[MAX_READ];

            StaticUnrollCount<unsigned int, 0, LESSOUT, CHUNK, 2>{[&](unsigned int k) {
                if(k + (MAX_READ * lid) < LESSOUT)
                {
                    for(unsigned int j = 0; j < MAX_READ; ++j)
                    {
                        const unsigned int l = k + (MAX_READ * lid) + j;
                        nidx = l / hip_plugin_bn_config::HW;
                        hwidx = l - (nidx * hip_plugin_bn_config::HW);
                        index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                        xhat[j] = (cast<FpPrecType>(in[index]) - mean) * invVariance;
                    }

                    __syncthreads();

                    for(unsigned int j = 0; j < MAX_READ; ++j) // This part takes 0.405
                    {
                        const unsigned int l = k + (MAX_READ * lid) + j;
                        nidx = l / hip_plugin_bn_config::HW;
                        hwidx = l - (nidx * hip_plugin_bn_config::HW);
                        index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                        out[index] = cast<FpType>(
                            hip_kernel_provider::applyActivation<
                                FpPrecType,
                                hip_kernel_provider::ActivationMode{HIP_PLUGIN_BN_NRN_OP_ID}>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }};

            if constexpr(REMOUT > 0u)
            {
                const unsigned int remkeyout = (MAX_READ * lid) + LESSOUT;
                for(unsigned int j = 0; j < MAX_READ; ++j)
                {
                    unsigned int l = remkeyout + j;
                    nidx = l / hip_plugin_bn_config::HW;
                    hwidx = l - (nidx * hip_plugin_bn_config::HW);
                    index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
                    // TODO: comparing different types
                    const auto xin = (index < hip_plugin_bn_config::NCHW)
                                         ? cast<FpPrecType>(in[index])
                                         : FpPrecType{0};
                    xhat[j] = (xin - cast<FpPrecType>(mean)) * cast<FpPrecType>(invVariance);
                }

                __syncthreads();
                for(unsigned int j = 0; j < MAX_READ; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    nidx = l / hip_plugin_bn_config::HW;
                    hwidx = l - (nidx * hip_plugin_bn_config::HW);
                    index = nidx * hip_plugin_bn_config::CHW + chwid + hwidx;

                    if(index < hip_plugin_bn_config::NCHW)
                    {
                        out[index] = cast<FpType>(
                            hip_kernel_provider::applyActivation<
                                FpPrecType,
                                hip_kernel_provider::ActivationMode{HIP_PLUGIN_BN_NRN_OP_ID}>(
                                fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }
        }
        return;
    }
};

// This is the instance for HIP_PLUGIN_BN_VARIANT == 3
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormFwdTrainSpatialImpl<3, FpType, FpPrecType, FpAccumType>
{

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType inhw,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        // SPATIAL
        mean = cast<FpPrecType>(0.);
        variance = cast<FpPrecType>(0.);
        invVariance = cast<FpPrecType>(0.);
        FpPrecType inhat = cast<FpPrecType>(0.);
        FpPrecType pvscale = cast<FpPrecType>(0.);
        FpPrecType pvbias = cast<FpPrecType>(0.);
        FpPrecType xin = cast<FpPrecType>(0.);

        __shared__ FpPrecType s_lcl_bias;
        __shared__ FpPrecType s_lcl_scale;

        unsigned int index = 0;
        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int cidx = grpid * hip_plugin_bn_config::HW;

        // Unused if hip_plugin_bn_config::N >= hip_plugin_bn_config::MAX_N
        FpType minibatch[HIP_PLUGIN_BN_N];

        if(lid == 0)
        {
            s_lcl_scale = *(scale + grpid);
            s_lcl_bias = *(bias + grpid);
        }

        if(lid < hip_plugin_bn_config::HW)
        {
            StaticUnrollCount<unsigned int, 0, hip_plugin_bn_config::N, 1, 2>{[&](unsigned int n) {
                index = n * hip_plugin_bn_config::CHW + cidx + lid;
                xin = cast<FpPrecType>(*(in + index));
                mean += xin;
                variance = fma(xin, xin, variance);

                if constexpr(hip_plugin_bn_config::N < hip_plugin_bn_config::MAX_N)
                {
                    minibatch[n] = (*(in + index));
                }
            }};
        }
        __syncthreads();

        constexpr auto LCL_DATA_SIZE = hip_plugin_bn_config::USE_AMDGCN
                                           ? hip_plugin_bn_config::LDS_GCN_SIZE
                                           : hip_plugin_bn_config::LDS_SIZE;
        __shared__ FpAccumType s_lcl_data_x[LCL_DATA_SIZE];
        __shared__ FpAccumType s_lcl_data_y[LCL_DATA_SIZE];
        if constexpr(hip_plugin_bn_config::USE_AMDGCN)
        {
            hip_kernel_provider::batchnorm::reduction::gcnReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }
        else
        {
            hip_kernel_provider::batchnorm::reduction::ldsReduce2<FpAccumType, LCL_DATA_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(inhw),
                s_lcl_data_x,
                s_lcl_data_y,
                lid);
        }

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = hip_kernel_provider::rsqrt(variance + static_cast<FpPrecType>(epsilon));

        if(lid < hip_plugin_bn_config::HW)
        {
            pvscale = s_lcl_scale;
            pvbias = s_lcl_bias;
            StaticUnrollCount<unsigned int, 0, hip_plugin_bn_config::N, 1, 2>{
                [&](unsigned int n) { // apply normalization
                    index = n * hip_plugin_bn_config::CHW + cidx + lid;
                    if constexpr(hip_plugin_bn_config::N < hip_plugin_bn_config::MAX_N)
                    {
                        inhat = (cast<FpPrecType>(minibatch[n]) - mean)
                                * invVariance; // (in[index] - mean) * invVariance;
                    }
                    else
                    {
                        inhat = (cast<FpPrecType>(*(in + index)) - mean) * invVariance;
                    }
                    out[index] = cast<FpType>(
                        hip_kernel_provider::applyActivation<FpPrecType,
                                                             hip_kernel_provider::ActivationMode{
                                                                 HIP_PLUGIN_BN_NRN_OP_ID}>(
                            fma(pvscale, inhat, pvbias), alpha, beta));
                }}; // end for

        } // end if
    }
};

// these are the kernels for HIP_PLUGIN_BN_VARIANT == 2
#if(HIP_PLUGIN_BN_VARIANT == 2)

template <typename FpType,
          typename FpType_C,
          typename FpLsType,
          typename FpPrecType,
          typename FpPrecType_C,
          typename FpPrecLsType,
          typename FpAccumType,
          typename FpAccumCType>
struct BatchNormFwdTrainSpatialImplVar2
{
    static constexpr unsigned int ngrps = HIP_PLUGIN_BN_NGRPS;
    static constexpr unsigned int ngrps2 = HIP_PLUGIN_BN_NGRPS2;

    static constexpr __forceinline__ __device__ void Norm(const FpType* __restrict__ in,
                                                          FpType* __restrict__ out,
                                                          const FpPrecType* scale,
                                                          const FpPrecType* bias,
                                                          FpPrecType alpha,
                                                          FpPrecType beta)
    {
        unsigned int xstride = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
        unsigned int ystride = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP0;
        unsigned int ygrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP1;
        unsigned int zgrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP2;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xgid = xgrp_id * xgrp_sz + xlid;
        unsigned int ygid = ygrp_id * ygrp_sz + ylid;
        unsigned int zgid = zgrp_id * zgrp_sz + zlid;

        unsigned int index;

        FpPrecType_C mean;
        FpPrecType_C invVariance;
        FpPrecLsType inhat; // this is a float4 when not vectorizing; for FPMIX FpPrecC is float,
            // FpType is __half
        FpPrecType_C pvt_scale;
        FpPrecType_C pvt_bias;
        FpLsType value;

        __shared__ FpPrecType_C lcl_bias[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
        __shared__ FpPrecType_C lcl_scale[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
        __shared__ FpPrecType_C lcl_mean[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
        __shared__ FpPrecType_C lcl_ivar[hip_plugin_bn_config::LAUNCH_DIM.GRP0];

        if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
            return;

        // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        if(ylid == 0 && zlid == 0)
        {
            lcl_scale[xlid]
                = *((const FpPrecType_C*)(scale + xgid * hip_plugin_bn_config::VEC_SIZE_X));
            lcl_bias[xlid]
                = *((const FpPrecType_C*)(bias + xgid * hip_plugin_bn_config::VEC_SIZE_X));
            lcl_mean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<FpPrecType_C, FpType_C>(
                (const FpType_C*)(out),
                0,
                zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                ygrp_sz * ygrp_id * hip_plugin_bn_config::VEC_SIZE_Y,
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            lcl_ivar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<FpPrecType_C, FpType_C>(
                (const FpType_C*)(out),
                1,
                zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                ygrp_sz * ygrp_id * hip_plugin_bn_config::VEC_SIZE_Y,
                ystride / hip_plugin_bn_config::VEC_SIZE_X,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
        __syncthreads();

        if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::HW
           && zgid < hip_plugin_bn_config::N)
        {
            mean = lcl_mean[xlid];
            invVariance = lcl_ivar[xlid];
            pvt_scale = lcl_scale[xlid];
            pvt_bias = lcl_bias[xlid];
            unsigned int index_base = zgid * HIP_PLUGIN_BN_N_ELEMENTS * HIP_PLUGIN_BN_CHW
                                      + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
                                      + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;

            // The original OpenCL kernel unrolled the loop only when this condition was met.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr unsigned int unrollHint
                = hip_plugin_bn_config::HW > hip_plugin_bn_config::LOOP_UNROLL_MAX_HW ? 1 : 2;

            // This method of unrolling is used as opposed to
            // the struct static_unroll_count due to a bug where
            // the kernel writes a tensor for the output twice bigger
            // than it should be, and it repeats values equal to the number of channels
            // in the examined test cases that were failing.

#if(HIP_PLUGIN_BN_HW > HIP_PLUGIN_BN_LOOP_UNROLL_MAXHW)
            for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
#else
#pragma unroll(2)
            for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
#endif
            {
                index = index_base + n * HIP_PLUGIN_BN_CHW;
                value = *((const FpLsType*)(in + index));
                inhat = cast<FpPrecLsType>(value);
                inhat = (inhat - mean) * invVariance;
                inhat = hip_kernel_provider::fma(
                    cast<FpPrecLsType>(pvt_scale), inhat, cast<FpPrecLsType>(pvt_bias));

                value = cast<FpLsType>(
                    hip_kernel_provider::applyActivation<FpPrecLsType,
                                                         hip_kernel_provider::ActivationMode{
                                                             HIP_PLUGIN_BN_NRN_OP_ID}>(
                        inhat, cast<FpPrecLsType>(alpha), cast<FpPrecLsType>(beta)));

                *((FpLsType*)(out + index)) = value;
            } // end for(n)

        } // end if(inImgIndex)
    } // end spatial norm

    static constexpr __forceinline__ __device__ void
        FinalMeanVariance(FpType* __restrict__ meanvarbuff,
                          FpPrecType INHW,
                          double epsilon,
                          const unsigned int& xgid,
                          const unsigned int& ygid,
                          const unsigned int& zgid,
                          unsigned int& commitID,
                          FpPrecType_C& mean,
                          FpPrecType_C& variance,
                          FpPrecType_C& invVariance)
    {
        variance = cast<FpPrecType_C>(0.);
        invVariance = cast<FpPrecType_C>(0.);
        mean = cast<FpPrecType_C>(0.);

        unsigned int xgrp_id = blockIdx.x;

        // These values (?grp_sz) cannot be substituted with hip_plugin_bn_config::LAUNCH_DIM.grp? because
        // the dimensions of the blocks for this kernel may be different from the other
        // kernels that take part in the operation. Given that the launch dimensions are
        // rounded up, blockDim would return a consistent value.

        unsigned int xgrp_sz = blockDim.x;
        unsigned int ygrp_sz = blockDim.y;
        unsigned int zgrp_sz = blockDim.z;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xstride = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
        unsigned int ystride = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

        commitID = 0;

        for(unsigned int zoffset = zlid; zoffset < ngrps2; zoffset += zgrp_sz)
        {
            for(unsigned int yoffset = ylid; yoffset < ngrps; yoffset += ygrp_sz)
            {
                mean += hip_kernel_provider::batchnorm::loadFromStash<FpPrecType_C>(
                    (FpType_C*)(meanvarbuff),
                    0,
                    hip_plugin_bn_config::LAUNCH_DIM.GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                    hip_plugin_bn_config::LAUNCH_DIM.GRP1 * yoffset
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrp_sz,
                    xgrp_id,
                    xlid,
                    xstride);
                variance += hip_kernel_provider::batchnorm::loadFromStash<FpPrecType_C>(
                    (FpType_C*)(meanvarbuff),
                    1,
                    hip_plugin_bn_config::LAUNCH_DIM.GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                    hip_plugin_bn_config::LAUNCH_DIM.GRP1 * yoffset
                        * hip_plugin_bn_config::VEC_SIZE_Y,
                    ystride / hip_plugin_bn_config::VEC_SIZE_X,
                    xgrp_sz,
                    xgrp_id,
                    xlid,
                    xstride);
            }
        }

        // Total workgroup size for final kernel - reused in condition and array declarations
        constexpr auto grp_final_total
            = HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;

        if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                     || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                     || hip_plugin_bn_config::VEC_SIZE_X > 1 || (grp_final_total < 64))
        {
            __shared__ FpAccumCType lcl_data[2 * grp_final_total];

            hip_kernel_provider::batchnorm::reduction::ldsReduce22d(mean,
                                                                    variance,
                                                                    INHW,
                                                                    lcl_data,
                                                                    xgrp_sz,
                                                                    xlid,
                                                                    ylid + zlid * ygrp_sz,
                                                                    ygrp_sz * zgrp_sz);
        }
        else
        {
            // C++17 idiomatic: ensure array size is never zero using constexpr ternary
            constexpr auto lds_gcn_array_size = grp_final_total >= 64 ? grp_final_total / 64 : 1;

            commitID = 64;
            __shared__ FpAccumCType lcl_data_x[lds_gcn_array_size];
            __shared__ FpAccumCType lcl_data_y[lds_gcn_array_size];
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(
                mean, variance, INHW, lcl_data_x, lcl_data_y, ylid + zlid * ygrp_sz);
        }

        variance = hip_kernel_provider::fma(-mean, mean, variance);
        variance = hip_kernel_provider::max(variance, cast<FpPrecType_C>(0.));
        invVariance = hip_kernel_provider::rsqrt(variance + cast<FpPrecType_C>(epsilon));

        for(unsigned int zoffset = zlid; zoffset < ngrps2; zoffset += zgrp_sz)
        {
            for(unsigned int yoffset = ylid; yoffset < ngrps; yoffset += ygrp_sz)
            {

                storeToStash(mean,
                             (FpType_C*)(meanvarbuff),
                             0,
                             HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                             HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                             ystride / hip_plugin_bn_config::VEC_SIZE_X,
                             xgrp_sz,
                             xgrp_id,
                             xlid,
                             xstride);
                storeToStash(invVariance,
                             (FpType_C*)(meanvarbuff),
                             1,
                             HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                             HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                             ystride / hip_plugin_bn_config::VEC_SIZE_X,
                             xgrp_sz,
                             xgrp_id,
                             xlid,
                             xstride);
            }
        }
    }

    static constexpr __forceinline__ __device__ void MeanVariance(const FpType* __restrict__ in,
                                                                  FpType* __restrict__ mvbuff)
    {
        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP0;
        unsigned int ygrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP1;
        unsigned int zgrp_sz = hip_plugin_bn_config::LAUNCH_DIM.GRP2;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xgid = xgrp_id * xgrp_sz + xlid;
        unsigned int ygid = ygrp_id * ygrp_sz + ylid;
        unsigned int zgid = zgrp_id * zgrp_sz + zlid;

        unsigned int xstride = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
        unsigned int ystride = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

        unsigned int index;

        FpPrecType_C mean = cast<FpPrecType_C>(0.);
        FpPrecType_C variance = cast<FpPrecType_C>(0.);
        FpPrecLsType value;

        if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
            return;

        if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::HW
           && zgid < hip_plugin_bn_config::N)
        {
            unsigned int index_base = zgid * HIP_PLUGIN_BN_N_ELEMENTS * hip_plugin_bn_config::CHW
                                      + ygid * ystride * hip_plugin_bn_config::VEC_SIZE_Y
                                      + xgid * xstride * hip_plugin_bn_config::VEC_SIZE_X;
            FpLsType read4;
            for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
            {
                index = index_base + n * hip_plugin_bn_config::CHW;
                read4 = *((const FpLsType*)(in + index));
                value = cast<FpPrecLsType>(read4);

                hip_kernel_provider::batchnorm::accumulate(mean, value);
                hip_kernel_provider::batchnorm::accumulateMad(variance, value, value);
            }
        }

        if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                     || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                     || hip_plugin_bn_config::VEC_SIZE_X > 1)
        {
            __shared__ FpAccumCType lcl_data[2 * hip_plugin_bn_config::LDS_SIZE];
            hip_kernel_provider::batchnorm::reduction::ldsReduce22d(mean,
                                                                    variance,
                                                                    cast<FpAccumType>(1.0),
                                                                    lcl_data,
                                                                    xgrp_sz,
                                                                    xlid,
                                                                    ylid + zlid * ygrp_sz,
                                                                    ygrp_sz * zgrp_sz);
        }
        else
        {
            __shared__ FpAccumCType lcl_data_x[hip_plugin_bn_config::LDS_GCN_SIZE];
            __shared__ FpAccumCType lcl_data_y[hip_plugin_bn_config::LDS_GCN_SIZE];
            hip_kernel_provider::batchnorm::reduction::gcnReduce2(mean,
                                                                  variance,
                                                                  cast<FpAccumType>(1.0),
                                                                  lcl_data_x,
                                                                  lcl_data_y,
                                                                  ylid + zlid * ygrp_sz);
        }

        if(ylid == 0 && zlid == 0)
        {
            storeToStash(mean,
                         (FpType_C*)(mvbuff),
                         0,
                         zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                         ygrp_sz * ygrp_id * hip_plugin_bn_config::VEC_SIZE_Y,
                         ystride / hip_plugin_bn_config::VEC_SIZE_X,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
            storeToStash(variance,
                         (FpType_C*)(mvbuff),
                         1,
                         zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                         ygrp_sz * ygrp_id * hip_plugin_bn_config::VEC_SIZE_Y,
                         ystride / hip_plugin_bn_config::VEC_SIZE_X,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
        }
    }
};

using BNFwdTrainSpatialVar2 = hip_kernel_provider::batchnorm::BatchNormFwdTrainSpatialImplVar2<
    hip_plugin_bn_config::fp_type,
    hip_plugin_bn_config::fp_c_type,
    hip_plugin_bn_config::fp_ls_type,
    hip_plugin_bn_config::fp_prec_type,
    hip_plugin_bn_config::fp_prec_c_type,
    hip_plugin_bn_config::fp_prec_ls_type,
    hip_plugin_bn_config::fp_accum_type,
    hip_plugin_bn_config::fp_accum_c_type>;

#endif // HIP_PLUGIN_BN_VARIANT == 2

} // namespace batchnorm
} // namespace hip_kernel_provider

/// C interfaces

// TODO: This can be removed after every variant has been implemnted
// [[deprecated]]
#if(HIP_PLUGIN_BN_VARIANT != 2)
extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormFwdTrainSpatial(
            const typename hip_plugin_bn_config::fp_type* __restrict in,
            typename hip_plugin_bn_config::fp_type* __restrict out,
            const typename hip_plugin_bn_config::fp_prec_type* __restrict scale,
            const typename hip_plugin_bn_config::fp_prec_type* __restrict bias,
            typename hip_plugin_bn_config::fp_prec_type inhw,
// TODO: should find a better way of doing this
// but it's hard becasue C does not support function
// overloads.
// [[deprecated]]
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
            double expAvgFactor,
            const typename hip_plugin_bn_config::fp_prec_type* __restrict prevResultRunningMean,
            const typename hip_plugin_bn_config::fp_prec_type* __restrict prevResultRunningVariance,
            typename hip_plugin_bn_config::fp_prec_type* __restrict nextResultRunningMean,
            typename hip_plugin_bn_config::fp_prec_type* __restrict nextResultRunningVariance,
#endif
            double epsilon
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
            ,
            typename hip_plugin_bn_config::fp_prec_type* __restrict resultSaveMean,
            typename hip_plugin_bn_config::fp_prec_type* __restrict resultSaveInvVariance
#endif
            ,
            typename hip_plugin_bn_config::fp_prec_type alpha,
            typename hip_plugin_bn_config::fp_prec_type beta)
{
    using fp_type = typename hip_plugin_bn_config::fp_type;
    using fp_prec_type = typename hip_plugin_bn_config::fp_prec_type;
    using fp_accum_type = typename hip_plugin_bn_config::fp_accum_type;
    using fp_accum_c_type = typename hip_plugin_bn_config::fp_accum_c_type;
    using fp_prec_c_type = typename hip_plugin_bn_config::fp_prec_c_type;
    constexpr auto VARIANT = hip_plugin_bn_config::VARIANT;

    using forward_train_spatial_impl = hip_kernel_provider::batchnorm::
        BatchNormFwdTrainSpatialImpl<VARIANT, fp_type, fp_prec_type, fp_accum_type>;

    fp_prec_type mean, variance, invVariance;
    const unsigned int lid = threadIdx.x;
    const unsigned int grpid = blockIdx.x;

    forward_train_spatial_impl{}(
        in, out, scale, bias, inhw, epsilon, mean, variance, invVariance, alpha, beta);

    if(lid == 0)
    {
// TODO: this should also be removed, but using constexpr can lead compile error
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<fp_accum_c_type>;
        StashUpdater updater(static_cast<fp_accum_c_type>(mean),
                             static_cast<fp_accum_c_type>(variance),
                             static_cast<fp_accum_c_type>(expAvgFactor));

        hip_kernel_provider::batchnorm::runningStash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            prevResultRunningMean,
            prevResultRunningVariance,
            nextResultRunningMean,
            nextResultRunningVariance,
            updater,
            grpid);
#endif
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::savedStash<fp_accum_c_type, fp_prec_c_type>(
            resultSaveMean,
            resultSaveInvVariance,
            static_cast<fp_accum_c_type>(mean),
            static_cast<fp_accum_c_type>(invVariance),
            grpid);
#endif
    }
}

#else

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormFwdTrainSpatialNorm(const hip_plugin_bn_config::fp_type* __restrict__ in,
                                     hip_plugin_bn_config::fp_type* __restrict__ out,
                                     const hip_plugin_bn_config::fp_prec_type* scale,
                                     const hip_plugin_bn_config::fp_prec_type* bias,
                                     hip_plugin_bn_config::fp_prec_type alpha,
                                     hip_plugin_bn_config::fp_prec_type beta)
{
    hip_kernel_provider::batchnorm::BNFwdTrainSpatialVar2{}.Norm(in, out, scale, bias, alpha, beta);
}

extern "C" __global__ void
    __launch_bounds__(HIP_PLUGIN_BN_GRP0_FINAL* HIP_PLUGIN_BN_GRP1_FINAL* HIP_PLUGIN_BN_GRP2_FINAL)
        batchNormFwdTrainSpatialFinalMeanVariance(
            hip_plugin_bn_config::fp_type* __restrict__ meanvarbuff,
            hip_plugin_bn_config::fp_prec_type INHW
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
            ,
            double expAvgFactor /* input momentum */
            ,
            const hip_plugin_bn_config::fp_prec_type* __restrict__ prevResultRunningMean,
            const hip_plugin_bn_config::fp_prec_type* __restrict__ prevResultRunningVariance,
            hip_plugin_bn_config::fp_prec_type* __restrict__ nextResultRunningMean,
            hip_plugin_bn_config::fp_prec_type* __restrict__ nextResultRunningVariance
#endif
            ,
            double epsilon
#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
            ,
            hip_plugin_bn_config::fp_prec_type* __restrict__ resultSaveMean /*output only*/
            ,
            hip_plugin_bn_config::fp_prec_type* __restrict__ resultSaveInvVariance
#endif
        )
{
    // mean, variance, invVariance

    using fp_prec_c_type = hip_plugin_bn_config::fp_prec_c_type;
    using fp_accum_type = hip_plugin_bn_config::fp_accum_type;
    using fp_accum_c_type = hip_plugin_bn_config::fp_accum_c_type;

    fp_prec_c_type mean;
    fp_prec_c_type variance;
    fp_prec_c_type invVariance;

    unsigned int xgid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int ygid = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int zgid = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int commitID;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
        return;

    hip_kernel_provider::batchnorm::BNFwdTrainSpatialVar2{}.FinalMeanVariance(
        meanvarbuff, INHW, epsilon, xgid, ygid, zgid, commitID, mean, variance, invVariance);
    // Save mean and calculate and save running mean
    if(ygid == commitID && zgid == 0)
    {
#if(HIP_PLUGIN_BN_RUNNING_RESULT == 1)
        using StashUpdater = hip_kernel_provider::batchnorm::StashUpdater<fp_accum_c_type>;
        StashUpdater updater(hip_kernel_provider::cast<fp_accum_c_type>(mean),
                             hip_kernel_provider::cast<fp_accum_c_type>(variance),
                             hip_kernel_provider::cast<fp_accum_c_type>(expAvgFactor));

        hip_kernel_provider::batchnorm::runningStash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            (const hip_plugin_bn_config::fp_prec_c_type*)prevResultRunningMean,
            (const hip_plugin_bn_config::fp_prec_c_type*)prevResultRunningVariance,
            (hip_plugin_bn_config::fp_prec_c_type*)nextResultRunningMean,
            (hip_plugin_bn_config::fp_prec_c_type*)nextResultRunningVariance,
            updater,
            xgid);
#endif

#if(HIP_PLUGIN_BN_SAVE_MEAN_VARIANCE == 1)
        hip_kernel_provider::batchnorm::savedStash<fp_accum_c_type, fp_prec_c_type>(
            (hip_plugin_bn_config::fp_prec_c_type*)resultSaveMean,
            (hip_plugin_bn_config::fp_prec_c_type*)resultSaveInvVariance,
            mean,
            invVariance,
            xgid);
#endif
    }
}

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormFwdTrainSpatialMeanVariance(const hip_plugin_bn_config::fp_type* __restrict__ in,
                                             hip_plugin_bn_config::fp_type* __restrict__ mvbuff)
{
    hip_kernel_provider::batchnorm::BNFwdTrainSpatialVar2{}.MeanVariance(in, mvbuff);
}

#endif
