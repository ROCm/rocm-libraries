// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// NOTE: These included headers should be standalone, they shouldn't rely on each other,
// otherwise the dependencies will be pretty messed up.
// TODO: actually these headers are not that independent due to the Macros that being used
// currently. we should remove as more macros as we can.
#include "batchnorm_functions.hpp"
#include "bnorm_spatial_activation_functions.hpp"
#include "activation_functions.hpp"
#include "configuration.hpp"
#include "reduction_functions.hpp"
#include "static_unroll.hpp"
#include "vector_types.hpp"
#include "miopen_math.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

namespace miopen {
namespace batchnorm {

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialImpl
{
    static_assert(false, "this variant is not supported.");
};

// This is the instance for MIO_BN_VARIANT == 0
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialImpl<0, FpType, FpPrecType, FpAccumType>
{
    // These are the configs for this variant
    static constexpr unsigned int segtmp_1 = mio_bn_config::launch_dim.grp0 / mio_bn_config::hw;
    static constexpr unsigned int segtmp_2 = (segtmp_1 == 0) ? 1 : segtmp_1;
    static constexpr unsigned int segtmp   = mio_bn_config::hw * segtmp_2;
    static constexpr unsigned int segment =
        (segtmp > mio_bn_config::nhw) ? mio_bn_config::nhw : segtmp;
    static constexpr unsigned int nloop  = (mio_bn_config::nhw + segment - 1) / segment;
    static constexpr unsigned int segihw = segment / mio_bn_config::hw;
    static constexpr unsigned int nloopm = nloop - 1;
    static constexpr unsigned int snhw   = nloopm * segihw;

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType INHW,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType _alpha,
                                                         FpPrecType _beta)
    {
        // ACTIVATION_SET()

        // SPATIAL
        mean        = cast<FpAccumType>(0.);
        variance    = cast<FpAccumType>(0.);
        invVariance = cast<FpAccumType>(0.);
        FpType batchvalues[nloop];
        FpAccumType temp;

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;

        unsigned int index  = 0;
        unsigned int lid    = threadIdx.x;
        unsigned int grpid  = blockIdx.x;
        unsigned int chwid  = grpid * mio_bn_config::hw + (lid % mio_bn_config::hw);
        unsigned int lidihw = lid / mio_bn_config::hw;
        unsigned int nid    = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        __syncthreads();

        if(lid < segment)
        {
            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int unrollHint =
                mio_config::input_type_strategy == type_strategy::fp16 ? 2 : 1;
            static_unroll_count<unsigned int, 0, nloopm, 1, unrollHint>{[&](unsigned int n) {
                nid            = n * segihw + lidihw;
                index          = nid * mio_bn_config::chw + chwid;
                batchvalues[n] = *(in + index);
                temp           = cast<FpAccumType>(*(in + index));
                mean += temp;
                variance = fma(temp, temp, variance);
            }};
            nid   = snhw + lidihw;
            index = nid * mio_bn_config::chw + chwid;
            batchvalues[nloopm] =
                (index < mio_bn_config::nchw) ? (*(in + index)) : cast<FpType>(0.);
            temp = cast<FpAccumType>(batchvalues[nloopm]);
            mean += temp;
            variance = fma(temp, temp, variance);
        }
        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            cast<FpAccumType>(INHW),
            lid);

        // Reduction complete

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = miopen::rsqrt(variance + cast<FpAccumType>(epsilon));

        FpAccumType pvscale = cast<FpAccumType>(lcl_scale);
        FpAccumType pvbias  = cast<FpAccumType>(lcl_bias);

        if(lid < segment)
        {
            // Calculate norm

            FpAccumType inhat = cast<FpAccumType>(0.);
            FpPrecType value;

            // The original OpenCL kernel unrolled the loop with a hint of 2 when using FP16.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr int unrollHint =
                mio_config::input_type_strategy == type_strategy::fp16 ? 2 : 1;

            static_unroll_count<unsigned int, 0, nloopm, 1, unrollHint>{[&](unsigned int n) {
                // Apply normalization
                inhat      = (cast<FpAccumType>(batchvalues[n]) - mean) * invVariance;
                nid        = n * segihw + lidihw;
                index      = nid * mio_bn_config::chw + chwid;
                value      = cast<FpPrecType>(fma(pvscale, inhat, pvbias));
                out[index] = cast<FpType>(miopen::batchnorm::activation_op(value, _alpha, _beta));
            }};

            // Tail of loop
            inhat = (cast<FpAccumType>(batchvalues[nloopm]) - mean) * invVariance;
            nid   = snhw + lidihw;
            index = nid * mio_bn_config::chw + chwid;
            if(index < mio_bn_config::nchw)
            {
                value      = cast<FpPrecType>(fma(pvscale, inhat, pvbias));
                out[index] = cast<FpType>(miopen::batchnorm::activation_op(value, _alpha, _beta));
            }
        }
    }
};

// This is the instance for MIO_BN_VARIANT == 1
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialImpl<1, FpType, FpPrecType, FpAccumType>
{
    // These are the configs for this variant
    static constexpr unsigned int max_read =
        mio_config::layout_nhwc ? 1 : (mio_bn_config::hw >= 4096 ? 3 : 2);
    static constexpr unsigned int rd_blk = 1;
    static constexpr unsigned int grprd  = mio_config::layout_nhwc
                                               ? (mio_bn_config::launch_dim.grp0 * rd_blk)
                                               : (mio_bn_config::launch_dim.grp0 * rd_blk * 4);
    static constexpr unsigned int rem4 =
        mio_bn_config::nhw - ((mio_bn_config::nhw / grprd) * grprd);
    static constexpr unsigned int less4  = mio_bn_config::nhw - rem4;
    static constexpr unsigned int chunk4 = max_read * grprd;
    static constexpr unsigned int remout4 =
        mio_bn_config::nhw - ((mio_bn_config::nhw / chunk4) * chunk4);
    static constexpr unsigned int lessout4 = mio_bn_config::nhw - remout4;
    static constexpr unsigned int rem =
        mio_bn_config::nhw -
        ((mio_bn_config::nhw / mio_bn_config::launch_dim.grp0) * mio_bn_config::launch_dim.grp0);
    static constexpr unsigned int less  = mio_bn_config::nhw - rem;
    static constexpr unsigned int chunk = max_read * mio_bn_config::launch_dim.grp0;
    static constexpr unsigned int remout =
        mio_bn_config::nhw - ((mio_bn_config::nhw / chunk) * chunk);
    static constexpr unsigned int lessout = mio_bn_config::nhw - remout;

    // Kernel
    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType INHW,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType pvscale, pvbias;

        mean        = 0;
        variance    = 0;
        invVariance = 0;

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;

        unsigned int index       = 0;
        const unsigned int lid   = threadIdx.x;
        const unsigned int grpid = blockIdx.x;

        // Note: this variable is only used when mio_config::layout_nhwc is false.
        unsigned int chwid;
        if constexpr(!mio_config::layout_nhwc)
        {
            chwid = grpid * mio_bn_config::hw;
        }

        unsigned int nidx  = 0;
        unsigned int hwidx = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!mio_config::layout_nhwc && mio_bn_config::hw >= 4096)
        {
            using fp_type4 = typename mapped_vector_type<FpType, 4>::type;
            fp_type4 read4;

            static_unroll_count<unsigned int, 0, less4, grprd, 2>{[&](unsigned int k) {
                if((k + (lid << 2)) < less4)
                {
                    nidx  = (k + (lid << 2)) / mio_bn_config::hw;
                    hwidx = (k + (lid << 2)) - (nidx * mio_bn_config::hw);
                    index = nidx * mio_bn_config::chw + chwid + hwidx;
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    miopen::batchnorm::_accumulate(mean, read4);
                    miopen::batchnorm::_accumulate_mad(variance, read4, read4);
                }
            }};

            if constexpr(rem4 > 0u)
            {
                const unsigned int remkey = (lid << 2) + less4;
                nidx                      = remkey / mio_bn_config::hw;
                hwidx                     = remkey - (nidx * mio_bn_config::hw);
                index                     = nidx * mio_bn_config::chw + chwid + hwidx;

                // index is unsigned int, so if the result would normally end up negative,
                // the value wraps around and the check fails. Improves on the
                // previous way of handling which was: if(index < (mio_bn_config::nchw - 3))
                if(index + 3 < (mio_bn_config::nchw))
                {
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    miopen::batchnorm::_accumulate(mean, read4);
                    miopen::batchnorm::_accumulate_mad(variance, read4, read4);
                }
            }
        }
        else
        {
            static_unroll_count<unsigned int, 0, less, mio_bn_config::launch_dim.grp0, 4>{
                [&](unsigned int k) {
                    if(k + lid < less)
                    {
                        nidx  = (k + lid) / mio_bn_config::hw;
                        hwidx = (k + lid) - (nidx * mio_bn_config::hw);
                        if constexpr(mio_config::layout_nhwc)
                        {
                            index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                        }
                        else
                        {
                            index = nidx * mio_bn_config::chw + chwid + hwidx;
                        }
                        const auto xin = cast<FpPrecType>(in[index]);
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
                    nidx                = remkey / mio_bn_config::hw;
                    hwidx               = remkey - (nidx * mio_bn_config::hw);
                    if constexpr(mio_config::layout_nhwc)
                    {
                        index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                    }
                    else
                    {
                        index = nidx * mio_bn_config::chw + chwid + hwidx;
                    }

                    const auto xin =
                        index < mio_bn_config::nchw ? cast<FpPrecType>(in[index]) : FpPrecType{0};
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
            lid);

        // REDUCTION COMPLETE ---------------------------
        variance = fma(-mean, mean, variance);
        if(variance < FpPrecType{0})
        {
            variance = FpPrecType{0};
        }

        // unsafe: casting double to FpPrecType
        invVariance = miopen::rsqrt(variance + static_cast<FpPrecType>(epsilon));
        pvscale     = lcl_scale;
        pvbias      = lcl_bias;
        if constexpr(mio_config::layout_nhwc || rem == 0)
        {
            constexpr unsigned int k_limit = mio_config::layout_nhwc ? mio_bn_config::nhw : less;

            static_unroll_count<unsigned int, 0, k_limit, mio_bn_config::launch_dim.grp0, 2>{
                [&](unsigned int k) {
                    if(k + lid < k_limit)
                    {
                        nidx  = (k + lid) / mio_bn_config::hw;
                        hwidx = (k + lid) - (nidx * mio_bn_config::hw);
                        if constexpr(mio_config::layout_nhwc)
                        {
                            index = nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid;
                        }
                        else
                        {
                            index = nidx * mio_bn_config::chw + chwid + hwidx;
                        }

                        out[index] = cast<FpType>(miopen::batchnorm::activation_op(
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
            FpPrecType xhat[max_read];

            static_unroll_count<unsigned int, 0, lessout, chunk, 2>{[&](unsigned int k) {
                if(k + (max_read * lid) < lessout)
                {
                    for(unsigned int j = 0; j < max_read; ++j)
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        nidx                 = l / mio_bn_config::hw;
                        hwidx                = l - (nidx * mio_bn_config::hw);
                        index                = nidx * mio_bn_config::chw + chwid + hwidx;
                        xhat[j]              = (cast<FpPrecType>(in[index]) - mean) * invVariance;
                    }

                    // Synchronization is not required for correctness but enhances performance.
                    //
                    // Loop is memory bound as it iterates across all the batches in the tensor,
                    // and has memory access strides of CHW size once all the elements in a single
                    // sample have been processed, which may be large.
                    //
                    // `__syncthreads()` helps to coalesce memory accesses as each work-item
                    // accesses adjacent elements to its neighbours on the same loop iteration,
                    // leading to contiguous memory access across all the waves in a workgroup. By
                    // keeping all the waves on the same loop iteration it prevents waves on
                    // different loop iterations from stalling as they wait for memory.
                    //
                    // This can be seen by profiling the kernel with rocprofv3 and comparing the
                    // `TCP_PENDING_STALL_CYCLES_sum` counter and also looking at a thread trace in
                    // compute viewer and seeing the impact on occupancy.
                    //
                    // TODO: This call is within the scope of an `if` condition, but it is not clear
                    // that this control flow is guanteed to be uniform across all threads in a
                    // workgroup, risking deadlock. Further investigation is needed.
                    __syncthreads();

                    for(unsigned int j = 0; j < max_read; ++j)
                    {
                        const unsigned int l = k + (max_read * lid) + j;
                        nidx                 = l / mio_bn_config::hw;
                        hwidx                = l - (nidx * mio_bn_config::hw);
                        index                = nidx * mio_bn_config::chw + chwid + hwidx;
                        out[index]           = cast<FpType>(miopen::batchnorm::activation_op(
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
                    nidx           = l / mio_bn_config::hw;
                    hwidx          = l - (nidx * mio_bn_config::hw);
                    index          = nidx * mio_bn_config::chw + chwid + hwidx;
                    // TODO: comparing different types
                    const auto xin =
                        (index < mio_bn_config::nchw) ? cast<FpPrecType>(in[index]) : FpPrecType{0};
                    xhat[j] = (xin - cast<FpPrecType>(mean)) * cast<FpPrecType>(invVariance);
                }

                __syncthreads();
                for(unsigned int j = 0; j < max_read; ++j)
                {
                    const unsigned int l = remkeyout + j;
                    nidx                 = l / mio_bn_config::hw;
                    hwidx                = l - (nidx * mio_bn_config::hw);
                    index                = nidx * mio_bn_config::chw + chwid + hwidx;

                    if(index < mio_bn_config::nchw)
                    {
                        out[index] = cast<FpType>(miopen::batchnorm::activation_op(
                            fma(pvscale, xhat[j], pvbias), alpha, beta));
                    }
                }
            }
        }
        return;
    }
};

// This is the instance for MIO_BN_VARIANT == 3
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialImpl<3, FpType, FpPrecType, FpAccumType>
{

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict scale,
                                                         const FpPrecType* __restrict bias,
                                                         FpPrecType INHW,
                                                         double epsilon,
                                                         FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        // SPATIAL
        mean               = cast<FpPrecType>(0.);
        variance           = cast<FpPrecType>(0.);
        invVariance        = cast<FpPrecType>(0.);
        FpPrecType inhat   = cast<FpPrecType>(0.);
        FpPrecType pvscale = cast<FpPrecType>(0.);
        FpPrecType pvbias  = cast<FpPrecType>(0.);
        FpPrecType xin     = cast<FpPrecType>(0.);

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;

        unsigned int index = 0;
        unsigned int lid   = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int cidx  = grpid * mio_bn_config::hw;

#if(MIO_BN_N < MIO_BN_MAXN)
        FpType minibatch[MIO_BN_N];
#endif

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        if(lid < mio_bn_config::hw)
        {
            static_unroll_count<unsigned int, 0, mio_bn_config::n, 1, 2>{[&](unsigned int n) {
                index = n * mio_bn_config::chw + cidx + lid;
                xin   = cast<FpPrecType>(*(in + index));
                mean += xin;
                variance = fma(xin, xin, variance);

#if(MIO_BN_N < MIO_BN_MAXN)
                minibatch[n] = (*(in + index));
#endif
            }};
        }
        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
            lid);

        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = miopen::rsqrt(variance + (FpPrecType)epsilon);

        if(lid < mio_bn_config::hw)
        {
            pvscale = lcl_scale;
            pvbias  = lcl_bias;
            static_unroll_count<unsigned int, 0, mio_bn_config::n, 1, 2>{
                [&](unsigned int n) { // apply normalization
                    index = n * mio_bn_config::chw + cidx + lid;
#if(MIO_BN_N < MIO_BN_MAXN)
                    inhat = (cast<FpPrecType>(minibatch[n]) - mean) *
                            invVariance; // (in[index] - mean) * invVariance;
#else
                    inhat = (cast<FpPrecType>(*(in + index)) - mean) * invVariance;
#endif
                    out[index] = cast<FpType>(
                        miopen::batchnorm::activation_op(fma(pvscale, inhat, pvbias), alpha, beta));
                }}; // end for

        } // end if
    }
};

// these are the kernels for MIO_BN_VARIANT == 2
#if(MIO_BN_VARIANT == 2)

template <typename FpType,
          typename FpType_C,
          typename FpLsType,
          typename FpPrecType,
          typename FpPrecType_C,
          typename FpPrecLsType,
          typename FpAccumType,
          typename FpAccumCType>
struct MIOpenBatchNormFwdTrainSpatialImplVar2
{
    static constexpr unsigned int ngrps  = MIO_BN_NGRPS;
    static constexpr unsigned int ngrps2 = MIO_BN_NGRPS2;

    static constexpr __forceinline__ __device__ void Norm(const FpType* __restrict__ in,
                                                          FpType* __restrict__ out,
                                                          const FpPrecType* scale,
                                                          const FpPrecType* bias,
                                                          FpPrecType alpha,
                                                          FpPrecType beta)
    {
        unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
        unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_sz = mio_bn_config::launch_dim.grp0;
        unsigned int ygrp_sz = mio_bn_config::launch_dim.grp1;
        unsigned int zgrp_sz = mio_bn_config::launch_dim.grp2;

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

        __shared__ FpPrecType_C lcl_bias[mio_bn_config::launch_dim.grp0];
        __shared__ FpPrecType_C lcl_scale[mio_bn_config::launch_dim.grp0];
        __shared__ FpPrecType_C lcl_mean[mio_bn_config::launch_dim.grp0];
        __shared__ FpPrecType_C lcl_ivar[mio_bn_config::launch_dim.grp0];

        if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
            return;

        // #4 apply the normalization :: x_hat = (x_i - mean) / sqrt(variance_accum + epsilon)
        if(ylid == 0 && zlid == 0)
        {
            lcl_scale[xlid] = *((const FpPrecType_C*)(scale + xgid * mio_bn_config::vec_size_x));
            lcl_bias[xlid]  = *((const FpPrecType_C*)(bias + xgid * mio_bn_config::vec_size_x));
            lcl_mean[xlid]  = miopen::batchnorm::loadFromStash<FpPrecType_C, FpType_C>(
                (const FpType_C*)(out),
                0,
                zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                ystride / mio_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            lcl_ivar[xlid] = miopen::batchnorm::loadFromStash<FpPrecType_C, FpType_C>(
                (const FpType_C*)(out),
                1,
                zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                ystride / mio_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
        __syncthreads();

        if(ygid * mio_bn_config::vec_size_y < mio_bn_config::hw && zgid < mio_bn_config::n)
        {
            mean                    = lcl_mean[xlid];
            invVariance             = lcl_ivar[xlid];
            pvt_scale               = lcl_scale[xlid];
            pvt_bias                = lcl_bias[xlid];
            unsigned int index_base = zgid * MIO_BN_N_ELEMENTS * MIO_BN_CHW +
                                      ygid * ystride * mio_bn_config::vec_size_y +
                                      xgid * xstride * mio_bn_config::vec_size_x;

            // The original OpenCL kernel unrolled the loop only when this condition was met.
            // Using this unrollHint and the static_unroll_count struct replicates this.
            constexpr unsigned int unrollHint =
                mio_bn_config::hw > mio_bn_config::loop_unroll_max_hw ? 1 : 2;

            // This method of unrolling is used as opposed to
            // the struct static_unroll_count due to a bug where
            // the kernel writes a tensor for the output twice bigger
            // than it should be, and it repeats values equal to the number of channels
            // in the examined test cases that were failing.

#if(MIO_BN_HW > MIO_BN_LOOP_UNROLL_MAXHW)
            for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
#else
#pragma unroll(2)
            for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
#endif
            {
                index = index_base + n * MIO_BN_CHW;
                value = *((const FpLsType*)(in + index));
                inhat = cast<FpPrecLsType>(value);
                inhat = (inhat - mean) * invVariance;
                inhat =
                    miopen::fma(cast<FpPrecLsType>(pvt_scale), inhat, cast<FpPrecLsType>(pvt_bias));

                value = cast<FpLsType>(miopen::batchnorm::activation_op(
                    inhat, cast<FpPrecLsType>(alpha), cast<FpPrecLsType>(beta)));

                *((FpLsType*)(out + index)) = value;
            } // end for(n)

        } // end if(inImgIndex)
    } // end spatial norm

    static constexpr __forceinline__ __device__ void
    FinalMeanVariance(FpType* __restrict__ meanvarbuff,
                      FpPrecType INHW,
                      double epsilon,
                      unsigned int& xgid,
                      unsigned int& ygid,
                      unsigned int& zgid,
                      unsigned int& commitID,
                      FpPrecType_C& mean,
                      FpPrecType_C& variance,
                      FpPrecType_C& invVariance)
    {
        variance    = cast<FpPrecType_C>(0.);
        invVariance = cast<FpPrecType_C>(0.);
        mean        = cast<FpPrecType_C>(0.);

        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        // These values (?grp_sz) cannot be substituted with mio_bn_config::launch_dim.grp? because
        // the dimensions of the blocks for this kernel may be different from the other
        // kernels that take part in the operation. Given that the launch dimensions are
        // rounded up, blockDim would return a consistent value.

        unsigned int xgrp_sz = blockDim.x;
        unsigned int ygrp_sz = blockDim.y;
        unsigned int zgrp_sz = blockDim.z;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        xgid = xgrp_id * xgrp_sz + xlid;
        ygid = ygrp_id * ygrp_sz + ylid;
        zgid = zgrp_id * zgrp_sz + zlid;

        unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
        unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

        commitID = 0;

        if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
            return;

        for(unsigned int zoffset = zlid; zoffset < ngrps2; zoffset += zgrp_sz)
        {
            for(unsigned int yoffset = ylid; yoffset < ngrps; yoffset += ygrp_sz)
            {
                mean += miopen::batchnorm::loadFromStash<FpPrecType_C>(
                    (FpType_C*)(meanvarbuff),
                    0,
                    mio_bn_config::launch_dim.grp2 * zoffset * MIO_BN_N_ELEMENTS,
                    mio_bn_config::launch_dim.grp1 * yoffset * mio_bn_config::vec_size_y,
                    ystride / mio_bn_config::vec_size_x,
                    xgrp_sz,
                    xgrp_id,
                    xlid,
                    xstride);
                variance += miopen::batchnorm::loadFromStash<FpPrecType_C>(
                    (FpType_C*)(meanvarbuff),
                    1,
                    mio_bn_config::launch_dim.grp2 * zoffset * MIO_BN_N_ELEMENTS,
                    mio_bn_config::launch_dim.grp1 * yoffset * mio_bn_config::vec_size_y,
                    ystride / mio_bn_config::vec_size_x,
                    xgrp_sz,
                    xgrp_id,
                    xlid,
                    xstride);
            }
        }

        // Total workgroup size for final kernel - reused in condition and array declarations
        constexpr auto grp_final_total = MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL;

        if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                     (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1 ||
                     (grp_final_total < 64))
        {
            __shared__ FpAccumCType lcl_data[2 * grp_final_total];

            miopen::reduction::lds_reduce2_2d(mean,
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
            miopen::reduction::reduce2<FpPrecType_C, grp_final_total>(
                mean, variance, static_cast<FpPrecType_C>(INHW), ylid + zlid * ygrp_sz);
        }

        variance    = miopen::fma(-mean, mean, variance);
        variance    = miopen::max(variance, cast<FpPrecType_C>(0.));
        invVariance = miopen::rsqrt(variance + cast<FpPrecType_C>(epsilon));

        for(unsigned int zoffset = zlid; zoffset < ngrps2; zoffset += zgrp_sz)
        {
            for(unsigned int yoffset = ylid; yoffset < ngrps; yoffset += ygrp_sz)
            {

                storeToStash(mean,
                             (FpType_C*)(meanvarbuff),
                             0,
                             MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                             MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                             ystride / mio_bn_config::vec_size_x,
                             xgrp_sz,
                             xgrp_id,
                             xlid,
                             xstride);
                storeToStash(invVariance,
                             (FpType_C*)(meanvarbuff),
                             1,
                             MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                             MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                             ystride / mio_bn_config::vec_size_x,
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

        unsigned int xgrp_sz = mio_bn_config::launch_dim.grp0;
        unsigned int ygrp_sz = mio_bn_config::launch_dim.grp1;
        unsigned int zgrp_sz = mio_bn_config::launch_dim.grp2;

        unsigned int xlid = threadIdx.x;
        unsigned int ylid = threadIdx.y;
        unsigned int zlid = threadIdx.z;

        unsigned int xgid = xgrp_id * xgrp_sz + xlid;
        unsigned int ygid = ygrp_id * ygrp_sz + ylid;
        unsigned int zgid = zgrp_id * zgrp_sz + zlid;

        unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
        unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

        unsigned int index;

        FpPrecType_C mean     = cast<FpPrecType_C>(0.);
        FpPrecType_C variance = cast<FpPrecType_C>(0.);
        FpPrecLsType value;

        if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
            return;

        if(ygid * mio_bn_config::vec_size_y < mio_bn_config::hw && zgid < mio_bn_config::n)
        {
            unsigned int index_base = zgid * MIO_BN_N_ELEMENTS * mio_bn_config::chw +
                                      ygid * ystride * mio_bn_config::vec_size_y +
                                      xgid * xstride * mio_bn_config::vec_size_x;
            FpLsType read4;
            for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
            {
                index = index_base + n * mio_bn_config::chw;
                read4 = *((const FpLsType*)(in + index));
                value = cast<FpPrecLsType>(read4);

                miopen::batchnorm::_accumulate(mean, value);
                miopen::batchnorm::_accumulate_mad(variance, value, value);
            }
        }

        if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                     (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1)
        {
            __shared__ FpAccumCType lcl_data[2 * mio_bn_config::lds_size];
            miopen::reduction::lds_reduce2_2d(mean,
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
            miopen::reduction::reduce2<FpPrecType_C, mio_bn_config::lds_size>(
                mean, variance, static_cast<FpPrecType_C>(1.0), ylid + zlid * ygrp_sz);
        }

        if(ylid == 0 && zlid == 0)
        {
            storeToStash(mean,
                         (FpType_C*)(mvbuff),
                         0,
                         zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                         ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                         ystride / mio_bn_config::vec_size_x,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
            storeToStash(variance,
                         (FpType_C*)(mvbuff),
                         1,
                         zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                         ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                         ystride / mio_bn_config::vec_size_x,
                         xgrp_sz,
                         xgrp_id,
                         xlid,
                         xstride);
        }
    }
};

using MIOpenBNFwdTrainSpatialVar2 =
    miopen::batchnorm::MIOpenBatchNormFwdTrainSpatialImplVar2<mio_bn_config::fp_type,
                                                              mio_bn_config::fp_c_type,
                                                              mio_bn_config::fp_ls_type,
                                                              mio_bn_config::fp_prec_type,
                                                              mio_bn_config::fp_prec_c_type,
                                                              mio_bn_config::fp_prec_ls_type,
                                                              mio_bn_config::fp_accum_type,
                                                              mio_bn_config::fp_accum_c_type>;

#endif // MIO_BN_VARIANT == 2

} // namespace batchnorm
} // namespace miopen

/// C interfaces

// TODO: This can be removed after every variant has been implemnted
// [[deprecated]]
#if(MIO_BN_VARIANT != 2)
extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormFwdTrainSpatial(
        const typename mio_bn_config::fp_type* __restrict in,
        typename mio_bn_config::fp_type* __restrict out,
        const typename mio_bn_config::fp_prec_type* __restrict scale,
        const typename mio_bn_config::fp_prec_type* __restrict bias,
        typename mio_bn_config::fp_prec_type INHW,
// TODO: should find a better way of doing this
// but it's hard becasue C does not support function
// overloads.
// [[deprecated]]
#if(MIO_RUNNING_RESULT == 1)
        double expAvgFactor,
        const typename mio_bn_config::fp_prec_type* __restrict prevResultRunningMean,
        const typename mio_bn_config::fp_prec_type* __restrict prevResultRunningVariance,
        typename mio_bn_config::fp_prec_type* __restrict nextResultRunningMean,
        typename mio_bn_config::fp_prec_type* __restrict nextResultRunningVariance,
#endif
        double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict resultSaveMean,
        typename mio_bn_config::fp_prec_type* __restrict resultSaveInvVariance
#endif
        ,
        typename mio_bn_config::fp_prec_type alpha,
        typename mio_bn_config::fp_prec_type beta)
{
    using fp_type          = typename mio_bn_config::fp_type;
    using fp_prec_type     = typename mio_bn_config::fp_prec_type;
    using fp_accum_type    = typename mio_bn_config::fp_accum_type;
    using fp_accum_c_type  = typename mio_bn_config::fp_accum_c_type;
    using fp_prec_c_type   = typename mio_bn_config::fp_prec_c_type;
    constexpr auto variant = mio_bn_config::variant;

    using forward_train_spatial_impl = miopen::batchnorm::
        MIOpenBatchNormFwdTrainSpatialImpl<variant, fp_type, fp_prec_type, fp_accum_type>;

    fp_prec_type mean, variance, invVariance;
    const unsigned int lid   = threadIdx.x;
    const unsigned int grpid = blockIdx.x;

    forward_train_spatial_impl{}(
        in, out, scale, bias, INHW, epsilon, mean, variance, invVariance, alpha, beta);

    if(lid == 0)
    {
// TODO: this should also be removed, but using constexpr can lead compile error
#if(MIO_RUNNING_RESULT == 1)
        using StashUpdater = miopen::batchnorm::StashUpdater<fp_accum_c_type>;
        StashUpdater updater(static_cast<fp_accum_c_type>(mean),
                             static_cast<fp_accum_c_type>(variance),
                             static_cast<fp_accum_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            prevResultRunningMean,
            prevResultRunningVariance,
            nextResultRunningMean,
            nextResultRunningVariance,
            updater,
            grpid);
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            resultSaveMean,
            resultSaveInvVariance,
            static_cast<fp_accum_c_type>(mean),
            static_cast<fp_accum_c_type>(invVariance),
            grpid);
#endif
    }
}

#else

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormFwdTrainSpatialNorm(const mio_bn_config::fp_type* __restrict__ in,
                                       mio_bn_config::fp_type* __restrict__ out,
                                       const mio_bn_config::fp_prec_type* scale,
                                       const mio_bn_config::fp_prec_type* bias,
                                       mio_bn_config::fp_prec_type alpha,
                                       mio_bn_config::fp_prec_type beta)
{
    miopen::batchnorm::MIOpenBNFwdTrainSpatialVar2{}.Norm(in, out, scale, bias, alpha, beta);
}

extern "C" __global__ void
__launch_bounds__(MIO_BN_GRP0_FINAL* MIO_BN_GRP1_FINAL* MIO_BN_GRP2_FINAL)
    MIOpenBatchNormFwdTrainSpatialFinalMeanVariance(
        mio_bn_config::fp_type* __restrict__ meanvarbuff,
        mio_bn_config::fp_prec_type INHW
#if(MIO_RUNNING_RESULT == 1)
        ,
        double expAvgFactor /* input momentum */
        ,
        const mio_bn_config::fp_prec_type* __restrict__ prevResultRunningMean,
        const mio_bn_config::fp_prec_type* __restrict__ prevResultRunningVariance,
        mio_bn_config::fp_prec_type* __restrict__ nextResultRunningMean,
        mio_bn_config::fp_prec_type* __restrict__ nextResultRunningVariance
#endif
        ,
        double epsilon
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        mio_bn_config::fp_prec_type* __restrict__ resultSaveMean /*output only*/
        ,
        mio_bn_config::fp_prec_type* __restrict__ resultSaveInvVariance
#endif
    )
{
    // mean, variance, invVariance

    using fp_prec_c_type  = mio_bn_config::fp_prec_c_type;
    using fp_accum_type   = mio_bn_config::fp_accum_type;
    using fp_accum_c_type = mio_bn_config::fp_accum_c_type;

    fp_prec_c_type mean;
    fp_prec_c_type variance;
    fp_prec_c_type invVariance;

    unsigned int xgid;
    unsigned int ygid;
    unsigned int zgid;
    unsigned int commitID;

    miopen::batchnorm::MIOpenBNFwdTrainSpatialVar2{}.FinalMeanVariance(
        meanvarbuff, INHW, epsilon, xgid, ygid, zgid, commitID, mean, variance, invVariance);
    // Save mean and calculate and save running mean
    if(ygid == commitID && zgid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        using StashUpdater = miopen::batchnorm::StashUpdater<fp_accum_c_type>;
        StashUpdater updater(miopen::cast<fp_accum_c_type>(mean),
                             miopen::cast<fp_accum_c_type>(variance),
                             miopen::cast<fp_accum_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            (const mio_bn_config::fp_prec_c_type*)prevResultRunningMean,
            (const mio_bn_config::fp_prec_c_type*)prevResultRunningVariance,
            (mio_bn_config::fp_prec_c_type*)nextResultRunningMean,
            (mio_bn_config::fp_prec_c_type*)nextResultRunningVariance,
            updater,
            xgid);
#endif

#if(MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            (mio_bn_config::fp_prec_c_type*)resultSaveMean,
            (mio_bn_config::fp_prec_c_type*)resultSaveInvVariance,
            mean,
            invVariance,
            xgid);
#endif
    }
}

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormFwdTrainSpatialMeanVariance(const mio_bn_config::fp_type* __restrict__ in,
                                               mio_bn_config::fp_type* __restrict__ mvbuff)
{
    miopen::batchnorm::MIOpenBNFwdTrainSpatialVar2{}.MeanVariance(in, mvbuff);
}

#endif
