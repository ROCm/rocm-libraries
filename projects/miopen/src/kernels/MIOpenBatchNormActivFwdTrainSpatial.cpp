// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "miopen_math.hpp"
#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"
#include "reduction_functions.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

namespace miopen {
namespace batchnorm {

template <typename...>
constexpr bool dependent_false = false;

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl
{
    static_assert(dependent_false<FpType>, "This variant is not supported.");
};

// Without the macro guards the compiler fails on the constant expressions. Somehow these
// definitions spill over to versions defined later in the source file causing weird errors.
// TODO: Find a way to remove them.
#if(MIO_BN_VARIANT == 0)

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl<0, FpType, FpPrecType, FpAccumType>
{
    static constexpr unsigned int segtmp =
        mio_bn_config::hw * (mio_bn_config::launch_dim.grp0 / mio_bn_config::hw);
    static constexpr unsigned int segment =
        (segtmp > mio_bn_config::nhw) ? mio_bn_config::nhw : segtmp;
    static constexpr unsigned int nloop  = (mio_bn_config::nhw + segment - 1) / segment;
    static constexpr unsigned int segihw = segment / mio_bn_config::hw;
    static constexpr unsigned int nloopm = nloop - 1;
    static constexpr unsigned int snhw   = nloopm * segihw;

    constexpr __forceinline__ __device__ void operator()(FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         float INHW,
                                                         const FpType alpha,
                                                         const FpType beta,
                                                         const FpType gamma,
                                                         double epsilon,
                                                         const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict bias,
                                                         const FpPrecType* __restrict scale)
    {
        mean        = 0;
        variance    = 0;
        invVariance = 0;

        FpPrecType batchvalues[nloop];

        unsigned int lid    = threadIdx.x;
        unsigned int grpid  = blockIdx.x;
        unsigned int chwid  = grpid * mio_bn_config::hw + (lid % mio_bn_config::hw);
        unsigned int lidihw = lid / mio_bn_config::hw;
        unsigned int nid    = 0;
        unsigned int index  = 0;

        FpPrecType bn_out  = 0.;
        FpPrecType act_out = 0.;

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;
        if(lid == 0)
        {
            lcl_scale = scale[grpid];
            lcl_bias  = bias[grpid];
        }

        __syncthreads();

        if(lid < segment)
        {
            for(unsigned int n = 0; n < nloopm; ++n)
            {
                nid            = n * segihw + lidihw;
                index          = nid * mio_bn_config::chw + chwid;
                batchvalues[n] = cast<FpPrecType>(in[index]);
                mean += batchvalues[n];
                variance = fma(batchvalues[n], batchvalues[n], variance);
            }
            nid                 = snhw + lidihw;
            index               = nid * mio_bn_config::chw + chwid;
            batchvalues[nloopm] = (index < mio_bn_config::nchw) ? cast<FpPrecType>(in[index]) : 0;
            mean += batchvalues[nloopm];
            variance = fma(batchvalues[nloopm], batchvalues[nloopm], variance);
        }

        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
            lid);

        variance           = fma(-mean, mean, variance);
        invVariance        = miopen::rsqrt(variance + static_cast<FpPrecType>(epsilon));
        FpPrecType pvscale = lcl_scale;
        FpPrecType pvbias  = lcl_bias;

        if(lid < segment)
        {
            FpPrecType inhat = 0;

            // Apply normalization
            for(unsigned int n = 0; n < nloopm; n++)
            {
                inhat             = (batchvalues[n] - mean) * invVariance;
                nid               = n * segihw + lidihw;
                index             = nid * mio_bn_config::chw + chwid;
                FpPrecType bn_out = fma(pvscale, inhat, pvbias);
                ActivationFunction<FpPrecType, 1>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                                  *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                                  cast<FpPrecType>(gamma),
                                                  cast<FpPrecType>(beta),
                                                  cast<FpPrecType>(alpha));
                out[index] = cast<FpPrecType>(act_out);
            }

            // Tail of loop
            inhat = (batchvalues[nloopm] - mean) * invVariance;
            nid   = snhw + lidihw;
            index = nid * mio_bn_config::chw + chwid;
            if(index < mio_bn_config::nchw)
            {
                FpPrecType bn_out = fma(pvscale, inhat, pvbias);
                ActivationFunction<FpPrecType, 1>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                                  *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                                  cast<FpPrecType>(gamma),
                                                  cast<FpPrecType>(beta),
                                                  cast<FpPrecType>(alpha));
                out[index] = cast<FpPrecType>(act_out);
            }
        }
    }
};

#elif(MIO_BN_VARIANT == 1)

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl<1, FpType, FpPrecType, FpAccumType>
{
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

    constexpr __forceinline__ __device__ void operator()(FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         float INHW,
                                                         const FpType alpha,
                                                         const FpType beta,
                                                         const FpType gamma,
                                                         double epsilon,
                                                         const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict bias,
                                                         const FpPrecType* __restrict scale)
    {
        mean        = 0;
        variance    = 0;
        invVariance = 0;

        FpPrecType bn_out, act_out;

        unsigned int index = 0;
        unsigned int nidx  = 0;
        unsigned int hwidx = 0;

        unsigned int lid   = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * mio_bn_config::hw;

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;
        if(lid == 0)
        {
            lcl_scale = scale[grpid];
            lcl_bias  = bias[grpid];
        }

        __syncthreads();

        if constexpr(mio_bn_config::hw >= 4096)
        {
            using fp_type4 = typename mapped_vector_type<FpType, 4>::type;
            fp_type4 read4;

            /*__attribute__((opencl_unroll_hint(2)))*/
            // TODO: Loop start / and / stride must be constexprs to unroll
            for(unsigned int k = lid << 2; k < less4; k += grprd)
            {
                nidx  = k / mio_bn_config::hw;
                hwidx = k - (nidx * mio_bn_config::hw);
                index = nidx * mio_bn_config::chw + chwid + hwidx;
                read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                _accumulate(mean, read4);
                _accumulate_mad(variance, read4, read4);
            }

            if constexpr(rem4 > 0u)
            {
                unsigned int remkey = (lid << 2) + less4;
                nidx                = remkey / mio_bn_config::hw;
                hwidx               = remkey - (nidx * mio_bn_config::hw);
                index               = nidx * mio_bn_config::chw + chwid + hwidx;
                if(index + 3 < mio_bn_config::nchw)
                {
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    _accumulate(mean, read4);
                    _accumulate_mad(variance, read4, read4);
                }
            }
        }
        else
        {
            /*__attribute__((opencl_unroll_hint(4))) */
            // TODO: Loop start / and / stride must be constexprs to unroll
            for(unsigned int k = lid; k < less; k += mio_bn_config::launch_dim.grp0)
            {
                nidx           = k / mio_bn_config::hw;
                hwidx          = k - (nidx * mio_bn_config::hw);
                index          = nidx * mio_bn_config::chw + chwid + hwidx;
                FpPrecType xin = cast<FpPrecType>(in[index]);
                mean += xin;
                variance = fma(xin, xin, variance);
            }
            if constexpr(rem > 0u)
            {
                if(lid < rem)
                {
                    unsigned int remkey = lid + less;
                    nidx                = remkey / mio_bn_config::hw;
                    hwidx               = remkey - (nidx * mio_bn_config::hw);
                    index               = nidx * mio_bn_config::chw + chwid + hwidx;
                    FpPrecType xin =
                        (index < mio_bn_config::nchw) ? cast<FpPrecType>(in[index]) : 0;
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
            lid);

        // REDUCTION COMPLETE ---------------------------
        variance    = fma(-mean, mean, variance);
        invVariance = miopen::rsqrt(variance + static_cast<FpPrecType>(epsilon));

        FpPrecType pvscale = lcl_scale;
        FpPrecType pvbias  = lcl_bias;

        // TODO: Maybe unused
        FpPrecType xhat[max_read];

        if constexpr(mio_config::layout_nhwc || rem == 0)
        {
            /*__attribute__((opencl_unroll_hint(2)))*/
            // TODO: Loop start / and / stride must be constexprs to unroll
            for(unsigned int k = lid; k < less; k += mio_bn_config::launch_dim.grp0)
            {
                nidx   = k / mio_bn_config::hw;
                hwidx  = k - (nidx * mio_bn_config::hw);
                index  = nidx * mio_bn_config::chw + chwid + hwidx;
                bn_out = fma(pvscale, (in[index] - mean) * invVariance, pvbias);
                ActivationFunction<FpPrecType>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                               *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                               cast<FpPrecType>(gamma),
                                               cast<FpPrecType>(beta),
                                               cast<FpPrecType>(alpha));
                out[index] = cast<FpPrecType>(act_out);
            }
        }
        else
        {
            /*__attribute__((opencl_unroll_hint(2)))*/
            // TODO: Loop start / and / stride must be constexprs to unroll
            for(unsigned int k = max_read * lid; k < lessout; k += chunk)
            {
                for(unsigned int j = 0; j < max_read; j++)
                {
                    unsigned int l = k + j;
                    nidx           = l / mio_bn_config::hw;
                    hwidx          = l - (nidx * mio_bn_config::hw);
                    index          = nidx * mio_bn_config::chw + chwid + hwidx;
                    xhat[j]        = (cast<FpPrecType>(in[index]) - mean) * invVariance;
                }

                // Synchronization is not required for correctness but enhances performance.
                //
                // Loop is memory bound as it iterates across all the batches in the tensor,
                // and has memory access strides of CHW size once all the elements in a single
                // sample have been processed, which may be large.
                //
                // `__syncthreads()` helps to coalesce memory accesses as each work-item accesses
                // adjacent elements to its neighbours on the same loop iteration, leading to
                // contiguous memory access across all the waves in a workgroup. By keeping all the
                // waves on the same loop iteration it prevents waves on different loop iterations
                // from stalling as they wait for memory.
                //
                // This can be seen by profiling the kernel with rocprofv3 and comparing the
                // `TCP_PENDING_STALL_CYCLES_sum` counter and also looking at a thread trace in
                // compute viewer and seeing the impact on occupancy.
                __syncthreads();
                for(unsigned int j = 0; j < max_read; j++)
                {
                    unsigned int l = k + j;
                    nidx           = l / mio_bn_config::hw;
                    hwidx          = l - (nidx * mio_bn_config::hw);
                    index          = nidx * mio_bn_config::chw + chwid + hwidx;
                    bn_out         = fma(pvscale, xhat[j], pvbias);
                    ActivationFunction<FpPrecType, 1>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                                      *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                                      cast<FpPrecType>(gamma),
                                                      cast<FpPrecType>(beta),
                                                      cast<FpPrecType>(alpha));
                    out[index] = cast<FpPrecType>(act_out);
                }
            }
        }

        if constexpr(remout > 0u)
        {
            unsigned int remkeyout = (max_read * lid) + lessout;
            for(unsigned int j = 0; j < max_read; j++)
            {
                unsigned int l = remkeyout + j;
                nidx           = l / mio_bn_config::hw;
                hwidx          = l - (nidx * mio_bn_config::hw);
                index          = nidx * mio_bn_config::chw + chwid + hwidx;
                FpPrecType xin = (index < mio_bn_config::nchw) ? cast<FpPrecType>(in[index]) : 0;
                xhat[j]        = (xin - mean) * invVariance;
            }

            __syncthreads();

            for(unsigned int j = 0; j < max_read; j++)
            {
                unsigned int l = remkeyout + j;
                nidx           = l / mio_bn_config::hw;
                hwidx          = l - (nidx * mio_bn_config::hw);
                index          = nidx * mio_bn_config::chw + chwid + hwidx;
                if(index < mio_bn_config::nchw)
                {
                    bn_out = fma(pvscale, xhat[j], pvbias);
                    ActivationFunction<FpPrecType, 1>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                                      *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                                      cast<FpPrecType>(gamma),
                                                      cast<FpPrecType>(beta),
                                                      cast<FpPrecType>(alpha));
                    out[index] = cast<FpPrecType>(act_out);
                }
            }
        }
    }
};

#elif(MIO_BN_VARIANT == 3)

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl<3, FpType, FpPrecType, FpAccumType>
{
    static constexpr bool NormPerN = mio_bn_config::n < mio_bn_config::max_n;

    // This variant implies the image is greater than a wavefront, but smaller than 257
    constexpr __forceinline__ __device__ void operator()(FpPrecType& mean,
                                                         FpPrecType& variance,
                                                         FpPrecType& invVariance,
                                                         float INHW,
                                                         const FpType alpha,
                                                         const FpType beta,
                                                         const FpType gamma,
                                                         double epsilon,
                                                         const FpType* __restrict in,
                                                         FpType* __restrict out,
                                                         const FpPrecType* __restrict bias,
                                                         const FpPrecType* __restrict scale)
    {
        mean        = 0;
        variance    = 0;
        invVariance = 0;

        unsigned int lid   = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int cidx  = grpid * mio_bn_config::hw;

        // Unused if (NormPerN == false)
        FpPrecType minibatch[mio_bn_config::n];

        __shared__ FpPrecType lcl_bias;
        __shared__ FpPrecType lcl_scale;
        if(lid == 0)
        {
            lcl_scale = scale[grpid];
            lcl_bias  = bias[grpid];
        }

        __syncthreads();

        if(lid < mio_bn_config::hw)
        {
            static_unroll_count<unsigned int, 0, mio_bn_config::n, 1, 2>{[&](unsigned int n) {
                unsigned int index = n * mio_bn_config::chw + cidx + lid;
                auto xin           = miopen::cast<FpPrecType>(in[index]);
                if constexpr(NormPerN)
                {
                    minibatch[n] = xin;
                }
                mean += xin;
                variance = fma(xin, xin, variance);
            }};
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
            lid);

        variance    = fma(-mean, mean, variance);
        invVariance = miopen::rsqrt(variance + static_cast<FpPrecType>(epsilon));

        if(lid < mio_bn_config::hw)
        {
            FpPrecType pvscale = lcl_scale;
            FpPrecType pvbias  = lcl_bias;
            FpPrecType bn_out, act_out;

            for(unsigned int n = 0; n < mio_bn_config::n; n++)
            { // apply normalization
                unsigned int index = n * mio_bn_config::chw + cidx + lid;
                FpPrecType inhat   = [&]() {
                    if constexpr(NormPerN)
                    {
                        return (minibatch[n] - mean) * invVariance;
                    }
                    else
                    {
                        return (cast<FpPrecType>(in[index]) - mean) * invVariance;
                    }
                }();

                bn_out = fma(pvscale, inhat, pvbias);
                ActivationFunction<FpPrecType, 1>(*reinterpret_cast<FpPrecType(*)[1]>(&act_out),
                                                  *reinterpret_cast<FpPrecType(*)[1]>(&bn_out),
                                                  miopen::cast<FpPrecType>(gamma),
                                                  miopen::cast<FpPrecType>(beta),
                                                  miopen::cast<FpPrecType>(alpha));
                out[index] = miopen::cast<FpPrecType>(act_out);
            }
        }
    }
};

#endif

} // namespace batchnorm
} // namespace miopen

/// C interfaces

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormActivFwdTrainSpatial(
        float INHW,
        const typename mio_bn_config::fp_type alpha,
        const typename mio_bn_config::fp_type beta,
        const typename mio_bn_config::fp_type gamma,
        double epsilon,
#if(MIO_RUNNING_RESULT == 1)
        double expAvgFactor,
#endif
        const typename mio_bn_config::fp_type* __restrict in,
        typename mio_bn_config::fp_type* __restrict out,
        const typename mio_bn_config::fp_prec_type* __restrict bias,
        const typename mio_bn_config::fp_prec_type* __restrict scale
#if(MIO_RUNNING_RESULT == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict runningMean,
        typename mio_bn_config::fp_prec_type* __restrict runningVariance
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict savedInvVariance,
        typename mio_bn_config::fp_prec_type* __restrict savedMean
#endif
    )
{
    using fp_type         = typename mio_bn_config::fp_type;
    using fp_prec_type    = typename mio_bn_config::fp_prec_type;
    using fp_accum_type   = typename mio_bn_config::fp_accum_type;
    using fp_accum_c_type = typename mio_bn_config::fp_accum_c_type;
    using fp_prec_c_type  = typename mio_bn_config::fp_prec_c_type;

    using ActivFwdTrainSpatialImpl =
        miopen::batchnorm::MIOpenBatchNormActivFwdTrainSpatialHIPImpl<mio_bn_config::variant,
                                                                      fp_type,
                                                                      fp_prec_type,
                                                                      fp_accum_type>;

    unsigned int grpid = blockIdx.x;
    unsigned int lid   = threadIdx.x;
    fp_prec_type mean, variance, invVariance;

    ActivFwdTrainSpatialImpl{}(
        mean, variance, invVariance, INHW, alpha, beta, gamma, epsilon, in, out, bias, scale);

    if(lid == 0)
    {
#if(MIO_RUNNING_RESULT == 1)
        using StashUpdater = miopen::batchnorm::StashUpdater<fp_prec_c_type>;
        StashUpdater updater(miopen::cast<fp_prec_c_type>(mean),
                             miopen::cast<fp_prec_c_type>(variance),
                             miopen::cast<fp_prec_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_prec_c_type, fp_prec_c_type, StashUpdater>(
            runningMean, runningVariance, runningMean, runningVariance, updater, grpid);
#endif
#if(MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_prec_c_type, fp_prec_c_type>(
            savedMean, savedInvVariance, mean, invVariance, grpid);
#endif
    }
}
