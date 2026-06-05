// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "float_types.h"
#include "vector_types.hpp"
#include "batchnorm_functions.hpp"
#include "bnorm_spatial_activation_functions.hpp"
#include "reduction_functions.hpp"
#include "static_unroll.hpp"
#include "miopen_type_traits.hpp"

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;

using fp_type         = typename mio_bn_config::fp_type;
using fp_c_type       = typename mio_bn_config::fp_c_type;
using fp_prec_type    = typename mio_bn_config::fp_prec_type;
using fp_accum_type   = typename mio_bn_config::fp_accum_type;
using fp_accum_c_type = typename mio_bn_config::fp_accum_c_type;
using fp_prec_c_type  = typename mio_bn_config::fp_prec_c_type;
using fp_ls_type      = typename mio_bn_config::fp_ls_type;
using fp_prec_ls_type = typename mio_bn_config::fp_prec_ls_type;

#define SHARED_MEMORY_SCALE 64 // wave size?

template <typename T>
__forceinline__ __device__ __host__ auto toPrecLsType(T val)
{
    return miopen::cast<fp_prec_ls_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toPrecCType(T val)
{
    return miopen::cast<fp_prec_c_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toLsType(T val)
{
    return miopen::cast<fp_ls_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toAccumCType(T val)
{
    return miopen::cast<fp_accum_c_type>(val);
}

template <typename FpPrecVecType,
          typename FpPrecType = miopen::mapped_vector_info<FpPrecVecType>::UnderlyingType>
__forceinline__ __device__ __host__ auto batchBwdNormalization(const FpPrecVecType value,
                                                               const FpPrecVecType xhat,
                                                               const FpPrecType dbias,
                                                               const FpPrecType dscale,
                                                               const FpPrecType pscale,
                                                               const FpPrecType invVariance,
                                                               const unsigned int nhw,
                                                               const FpPrecType inhw)
{
    FpPrecVecType tmp1 =
        miopen::fma(miopen::cast<FpPrecVecType>(nhw), value, miopen::cast<FpPrecVecType>(-dbias));
    FpPrecVecType tmp2 = -xhat * miopen::cast<FpPrecVecType>(dscale);
    FpPrecType tmp3    = pscale * invVariance * inhw;
    return miopen::cast<FpPrecVecType>(tmp3) * (tmp2 + tmp1);
}

// Specialized version for MIO_BN_VARIANT 2
template <typename FpPrecVecType>
__forceinline__ __device__ __host__ auto batchBwdNormalization(const FpPrecVecType value,
                                                               const FpPrecVecType xhat,
                                                               const FpPrecVecType dbias,
                                                               const FpPrecVecType dscale,
                                                               const FpPrecVecType pscale,
                                                               const FpPrecVecType invVariance,
                                                               const FpPrecVecType nhw,
                                                               const FpPrecVecType inhw)
{
    FpPrecVecType tmp1 = miopen::fma(nhw, value, -dbias);
    FpPrecVecType tmp2 = -xhat * dscale;
    FpPrecVecType tmp3 = pscale * invVariance * inhw;
    return tmp3 * (tmp2 + tmp1);
}

template <typename FpPrecVecType,
          miopen::neuron_op_type NrnOpType,
          typename FpPrecType = miopen::mapped_vector_info<FpPrecVecType>::UnderlyingType>
__forceinline__ __host__ __device__ FpPrecVecType
vectorizedBwdActivationOp(FpPrecVecType const& dy,
                          FpPrecVecType const& xnorm,
                          FpPrecType const& scale,
                          FpPrecType const& bias,
                          FpPrecType const& alpha,
                          FpPrecType const& beta)
{
    auto constexpr size = miopen::mapped_vector_info<FpPrecVecType>::size;
    if constexpr(size == 4)
    {
        FpPrecVecType out;
        out.x = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        out.z = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.z, xnorm.z, scale, bias, alpha, beta);
        out.w = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.w, xnorm.w, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(size == 2)
    {
        FpPrecVecType out;
        out.x = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(size == 1)
    {
        return miopen::batchnorm::bwd_activation_op<FpPrecType, NrnOpType>(
            dy, xnorm, scale, bias, alpha, beta);
    }
    else
    {
        static_assert(false, "Unsupported miopen vector operation.");
    }
}

} // namespace

// Note: Calls with !MIO_BN_USESAVED configurations are not tested with the CI. Apparently there are
// some precision issues with the original CL version as well; it is not clear if this is an
// implementation or design problem. During the HIP port we only verified that these kernels run and
// give valid numerical result, but the precision issues were not addressed.

namespace miopen {
namespace batchnorm {

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormBwdSpatialHIPImpl
{
    static_assert(false, "This variant is not supported.");
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormBwdSpatialHIPImpl<0, FpType, FpPrecType, FpAccumType>
{
    static constexpr unsigned int segtmp1 = mio_bn_config::launch_dim.grp0 / mio_bn_config::hw;
    static constexpr unsigned int segtmp2 = segtmp1 == 0 ? 1 : segtmp1;
    static constexpr unsigned int segtmp  = mio_bn_config::hw * segtmp2;
    static constexpr unsigned int segment =
        segtmp > mio_bn_config::nhw ? mio_bn_config::nhw : segtmp;
    static constexpr unsigned int nloop  = (mio_bn_config::nhw + segment - 1) / segment;
    static constexpr unsigned int segihw = segment / mio_bn_config::hw;
    static_assert(nloop > 0);
    static constexpr unsigned int nloopm = nloop - 1;
    static constexpr unsigned int snhw   = nloopm * segihw;

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict x_in,
                                                         const FpType* __restrict dy_in,
                                                         FpType* __restrict dx_out,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                                                         double epsilon,
#elif(MIO_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType INHW,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
#if(MIO_BN_USESAVED == 0)
        FpPrecType variance = 0;
#endif
        FpPrecType mean        = 0;
        FpPrecType invVariance = 0;
        FpPrecType pscale      = 0;
        FpPrecType pbias       = 0;
        FpAccumType ds         = 0;
        FpAccumType db         = 0;

        FpPrecType batchvalues[nloop];
        FpPrecType dyvalues[nloop];

        __shared__ FpPrecType lbns;
#if(MIOPEN_NRN_OP_ID > 0)
        __shared__ FpPrecType lbnb;
#endif

#if(MIO_BN_USESAVED == 1)
        __shared__ FpPrecType lmean, lvar;
#endif
        unsigned int index  = 0;
        unsigned int lid    = threadIdx.x;
        unsigned int grpid  = blockIdx.x;
        unsigned int chwid  = grpid * mio_bn_config::hw + (lid % mio_bn_config::hw);
        unsigned int lidihw = lid / mio_bn_config::hw;
        unsigned int nid    = 0;

        if(lid == 0)
        {
            lbns = bnScale[grpid];
#if(MIOPEN_NRN_OP_ID > 0)
            lbnb = bnBias[grpid];
#endif
        }

#if(MIO_BN_USESAVED == 1)
        if(lid == 0)
        {
            lmean = savedMean[grpid];
            lvar  = savedInvVariance[grpid];
        }
        __syncthreads();
        mean        = lmean;
        invVariance = lvar;
#else // recalc mean and variance below
      // == RECALC MEAN AND VARIANCE ===========
        if(lid < segment)
        {
            for(unsigned int n = 0; n < nloopm; ++n)
            {
                nid            = n * segihw + lidihw;
                index          = nid * mio_bn_config::chw + chwid;
                batchvalues[n] = cast<FpPrecType>(x_in[index]);
                mean += batchvalues[n];
                variance = fma(batchvalues[n], batchvalues[n], variance);
            }
            nid                 = snhw + lidihw;
            index               = nid * mio_bn_config::chw + chwid;
            batchvalues[nloopm] = (index < mio_bn_config::nchw) ? cast<FpPrecType>(x_in[index]) : 0;
            mean += batchvalues[nloopm];
            variance = fma(batchvalues[nloopm], batchvalues[nloopm], variance);
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
        invVariance = rsqrt(variance + epsilon);

#endif // end -- Recalc mean and variance
       //-------------------------------------------
        pscale = lbns;
#if(MIOPEN_NRN_OP_ID > 0)
        pbias = lbnb;
#endif

        //==== CALC DB and DS =========================================
        if(lid < segment)
        {
            for(unsigned int n = 0; n < nloopm; ++n)
            {
                nid         = n * segihw + lidihw;
                index       = nid * mio_bn_config::chw + chwid;
                dyvalues[n] = cast<FpPrecType>(dy_in[index]);

#if(MIO_BN_USESAVED == 1)
                batchvalues[n] = (cast<FpPrecType>(x_in[index]) - mean) * invVariance;
#else
                batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif
                dyvalues[n] = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                    dyvalues[n], batchvalues[n], pscale, pbias, alpha, beta);
                // batchvalues is now xhat
                db += dyvalues[n];
                ds = fma(batchvalues[n], dyvalues[n], ds);
            }
            nid              = snhw + lidihw;
            index            = nid * mio_bn_config::chw + chwid;
            dyvalues[nloopm] = ((index < mio_bn_config::nchw) ? cast<FpPrecType>(dy_in[index]) : 0);

#if(MIO_BN_USESAVED == 1)
            batchvalues[nloopm] = (index < mio_bn_config::nchw)
                                      ? ((cast<FpPrecType>(x_in[index]) - mean) * invVariance)
                                      : 0;
#else
            batchvalues[nloopm] = (batchvalues[nloopm] - mean) * invVariance;
#endif
            dyvalues[nloopm] = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                dyvalues[nloopm], batchvalues[nloopm], pscale, pbias, alpha, beta);
            // batchvalues is now xhat
            db += dyvalues[nloopm];
            ds = fma(batchvalues[nloopm], dyvalues[nloopm], ds);
        }

        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            FpAccumType(1.0),
            lid);

        if(lid < segment)
        {
            //==== CALC NORM =======================
            FpPrecType value;
            for(unsigned int n = 0; n < nloopm; n++)
            {
                nid           = n * segihw + lidihw;
                index         = nid * mio_bn_config::chw + chwid;
                dx_out[index] = cast<FpType>(batchBwdNormalization(dyvalues[n],
                                                                   batchvalues[n],
                                                                   cast<FpPrecType>(db),
                                                                   cast<FpPrecType>(ds),
                                                                   pscale,
                                                                   invVariance,
                                                                   mio_bn_config::nhw,
                                                                   INHW));
            } // end for
            nid   = snhw + lidihw;
            index = nid * mio_bn_config::chw + chwid;
            if(index < mio_bn_config::nchw)
            {
                dx_out[index] = cast<FpType>(batchBwdNormalization(dyvalues[nloopm],
                                                                   batchvalues[nloopm],
                                                                   cast<FpPrecType>(db),
                                                                   cast<FpPrecType>(ds),
                                                                   pscale,
                                                                   invVariance,
                                                                   mio_bn_config::nhw,
                                                                   INHW));
            }
        }
        if(lid == 0)
        {
            dbias[grpid]  = cast<FpPrecType>(db);
            dscale[grpid] = cast<FpPrecType>(ds);
        }
    }
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormBwdSpatialHIPImpl<1, FpType, FpPrecType, FpAccumType>
{
    static constexpr unsigned int read_size  = mio_config::layout_nhwc ? 1 : 4;
    static constexpr unsigned int write_size = mio_config::layout_nhwc ? 1 : 2;

    using fp_read_vec_type       = typename mapped_vector_type<FpType, read_size>::type;
    using fp_prec_read_vec_type  = typename mapped_vector_type<FpPrecType, read_size>::type;
    using fp_write_vec_type      = typename mapped_vector_type<FpType, write_size>::type;
    using fp_prec_write_vec_type = typename mapped_vector_type<FpPrecType, write_size>::type;

    static constexpr unsigned int rd_blk = 1;
    static constexpr unsigned int grprd  = mio_bn_config::launch_dim.grp0 * rd_blk * read_size;
    static constexpr unsigned int rem4  = mio_bn_config::nhw - (mio_bn_config::nhw / grprd) * grprd;
    static constexpr unsigned int less4 = mio_bn_config::nhw - rem4;
    static constexpr unsigned int rem =
        mio_bn_config::nhw -
        (mio_bn_config::nhw / mio_bn_config::launch_dim.grp0) * mio_bn_config::launch_dim.grp0;
    static constexpr unsigned int less  = mio_bn_config::nhw - rem;
    static constexpr unsigned int chunk = write_size * mio_bn_config::launch_dim.grp0;
    static constexpr unsigned int remout =
        mio_bn_config::nhw - ((mio_bn_config::nhw / chunk) * chunk);
    static constexpr unsigned int lessout = mio_bn_config::nhw - remout;

    __forceinline__ __device__ unsigned int getTensorIndex(unsigned int loopIndex)
    {
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * mio_bn_config::hw;
        unsigned int nidx  = loopIndex / mio_bn_config::hw;
        unsigned int hwidx = loopIndex - (nidx * mio_bn_config::hw);
        return mio_config::layout_nhwc
                   ? nidx * mio_bn_config::chw + hwidx * mio_bn_config::c + grpid
                   : nidx * mio_bn_config::chw + chwid + hwidx;
    }

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict x_in,
                                                         const FpType* __restrict dy_in,
                                                         FpType* __restrict dx_out,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                                                         double epsilon,
#elif(MIO_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType INHW,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType mean        = 0;
        FpPrecType invVariance = 0;
        FpPrecType pscale      = 0;
        FpPrecType pbias       = 0;
        FpAccumType db         = 0;
        FpAccumType ds         = 0;
        FpPrecType xhat        = 0;

        unsigned int lid   = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * mio_bn_config::hw;

        pscale = bnScale[grpid];
#if(MIOPEN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(MIO_BN_USESAVED == 0)
        //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
        FpPrecType variance = 0;
        if constexpr(!mio_config::layout_nhwc && mio_bn_config::hw >= 4096)
        {
            fp_prec_read_vec_type read4;
            for(unsigned int k = lid << 2; k < less4; k += grprd)
            {
                read4 = cast<fp_prec_read_vec_type>(
                    *(reinterpret_cast<const fp_read_vec_type*>(x_in + getTensorIndex(k))));
                miopen::batchnorm::_accumulate(mean, read4);
                miopen::batchnorm::_accumulate_mad(variance, read4, read4);
            }

            if constexpr(rem4 > 0)
            {
                if(lid < rem4)
                {
                    unsigned int index = getTensorIndex(lid + less4);
                    if(index + read_size - 1 < mio_bn_config::nchw)
                    {
                        read4 = cast<fp_prec_read_vec_type>(
                            *(reinterpret_cast<const fp_read_vec_type*>(x_in + index)));
                        miopen::batchnorm::_accumulate(mean, read4);
                        miopen::batchnorm::_accumulate_mad(variance, read4, read4);
                    }
                }
            }
        }
        else
        {
            for(unsigned int k = lid; k < less; k += mio_bn_config::launch_dim.grp0)
            {
                FpPrecType in = cast<FpPrecType>(x_in[getTensorIndex(k)]);
                mean += in;
                variance = fma(in, in, variance);
            }
            if constexpr(rem > 0)
            {
                if(lid < rem)
                {
                    unsigned int index = getTensorIndex(lid + less);
                    FpPrecType in =
                        (index < mio_bn_config::nchw) ? cast<FpPrecType>(x_in[index]) : 0;
                    mean += in;
                    variance = fma(in, in, variance);
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
        variance = fma(-mean, mean, variance);
        if(variance < 0)
        {
            variance = 0;
        }
        invVariance = rsqrt(variance + epsilon);

#else // MIO_BN_USESAVED == 1
        mean        = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#endif

        constexpr unsigned int readUnrollHint =
            mio_bn_config::n > mio_bn_config::loop_unroll_max_n ? 4 : 2;
        static_unroll_count<unsigned int, 0, less4, grprd, readUnrollHint>{[&](unsigned int k) {
            unsigned int l = k + (lid << 2 * (1 - mio_config::layout_nhwc));
            if(l < less4)
            {
                unsigned int index     = getTensorIndex(l);
                fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(x_in + index));
                fp_read_vec_type dyRead =
                    *(reinterpret_cast<const fp_read_vec_type*>(dy_in + index));
                fp_prec_read_vec_type dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                fp_prec_read_vec_type xhat =
                    (cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

                dyvalue = vectorizedBwdActivationOp<fp_prec_read_vec_type, mio_config::neuron_op>(
                    dyvalue, xhat, pscale, pbias, alpha, beta);

                miopen::batchnorm::_accumulate(db, dyvalue);
                miopen::batchnorm::_accumulate_mad(ds, xhat, dyvalue);
            }
        }};

        if constexpr(rem4 > 0)
        {
            unsigned int index = getTensorIndex((lid << 2 * (1 - mio_config::layout_nhwc)) + less4);
            if(index + read_size - 1 < mio_bn_config::nchw)
            {
                fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(x_in + index));
                fp_read_vec_type dyRead =
                    *(reinterpret_cast<const fp_read_vec_type*>(dy_in + index));
                fp_prec_read_vec_type dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                fp_prec_read_vec_type xhat =
                    (cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

                dyvalue = vectorizedBwdActivationOp<fp_prec_read_vec_type, mio_config::neuron_op>(
                    dyvalue, xhat, pscale, pbias, alpha, beta);

                miopen::batchnorm::_accumulate(db, dyvalue);
                miopen::batchnorm::_accumulate_mad(ds, xhat, dyvalue);
            }
        }

        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            cast<FpAccumType>(1.0),
            lid);

        __syncthreads();

        if(lid == 0)
        {
            dbias[grpid]  = cast<FpPrecType>(db);
            dscale[grpid] = cast<FpPrecType>(ds);
        }

        constexpr unsigned int writeUnrollHint =
            mio_bn_config::n > mio_bn_config::loop_unroll_max_n ? 2 : 1;
        static_unroll_count<unsigned int, 0, lessout, chunk, writeUnrollHint>{[&](unsigned int k) {
            // Unrolling the loop requires forcing explicit data vectorization otherwise the
            // compiler will start splitting global loads into smaller chunks resulting in
            // significant slowdown.
            fp_prec_write_vec_type vals;
            unsigned int l     = k + (write_size * lid);
            unsigned int index = getTensorIndex(l);
            if(l < lessout)
            {
                fp_write_vec_type xread =
                    *(reinterpret_cast<const fp_write_vec_type*>(x_in + index));
                fp_write_vec_type dyRead =
                    *(reinterpret_cast<const fp_write_vec_type*>(dy_in + index));
                fp_prec_write_vec_type value1 = cast<fp_prec_write_vec_type>(dyRead);
                fp_prec_write_vec_type xhat1 =
                    (cast<fp_prec_write_vec_type>(xread) - mean) * invVariance;

                value1 = vectorizedBwdActivationOp<fp_prec_write_vec_type, mio_config::neuron_op>(
                    value1, xhat1, pscale, pbias, alpha, beta);

                vals = batchBwdNormalization(value1,
                                             xhat1,
                                             cast<FpPrecType>(db),
                                             cast<FpPrecType>(ds),
                                             pscale,
                                             invVariance,
                                             mio_bn_config::nhw,
                                             INHW);
            }

            // Synchronization is not required for correctness but enhances performance.
            //
            // Loop is memory bound as it iterates across all the batches in the tensor,
            // and has memory access strides of CHW size once all the elements in a single
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
                *reinterpret_cast<fp_write_vec_type*>(dx_out + index) =
                    cast<fp_write_vec_type>(vals);
            }
        }};

        if constexpr(remout > 0)
        {
            unsigned int remkeyout = (write_size * lid) + lessout;
            for(unsigned int j = 0; j < write_size; j++)
            {
                unsigned int index = getTensorIndex(remkeyout + j);
                if(index < mio_bn_config::nchw)
                {
                    FpPrecType value1 = cast<FpPrecType>(dy_in[index]);
                    FpPrecType xhat   = (cast<FpPrecType>(x_in[index]) - mean) * invVariance;

                    value1 = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                        value1, xhat, pscale, pbias, alpha, beta);

                    dx_out[index] = cast<FpType>(batchBwdNormalization(value1,
                                                                       xhat,
                                                                       cast<FpPrecType>(db),
                                                                       cast<FpPrecType>(ds),
                                                                       pscale,
                                                                       invVariance,
                                                                       mio_bn_config::nhw,
                                                                       INHW));
                }
            }
        }
    }
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormBwdSpatialHIPImpl<3, FpType, FpPrecType, FpAccumType>
{

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict x_in,
                                                         const FpType* __restrict dy_in,
                                                         FpType* __restrict dx_out,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                                                         double epsilon,
#elif(MIO_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType INHW,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType mean = 0;
#if(MIO_BN_USESAVED == 0)
        FpPrecType variance = 0;
#endif
        FpPrecType invVariance = 0;
        FpPrecType pscale      = 0;
        FpPrecType pbias       = 0;
        FpPrecType ds          = 0;
        FpPrecType db          = 0;

        // maybe unused if MIO_BN_N >= MIO_BN_MAXN
        FpPrecType batchvalues[mio_bn_config::n];
        FpPrecType dyvalues[mio_bn_config::n];

        unsigned int lid   = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int index;
        unsigned int cidx = grpid * mio_bn_config::hw;

        pscale = bnScale[grpid];
#if(MIOPEN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(MIO_BN_USESAVED == 1)
        mean        = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#else // recalc mean and variance

        if(lid < mio_bn_config::hw)
        {
            for(int n = 0; n < mio_bn_config::n; n++)
            {
                index = n * mio_bn_config::chw + cidx + lid;
                if constexpr(mio_bn_config::n < mio_bn_config::max_n)
                {
                    batchvalues[n] = cast<FpPrecType>(x_in[index]);
                    mean += batchvalues[n];
                    variance = fma(batchvalues[n], batchvalues[n], variance);
                }
                else
                {
                    FpPrecType in = cast<FpPrecType>(x_in[index]);
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }
        else
        {
            mean     = 0;
            variance = 0;
        }

        // REDUCE MEAN AND VARIANCE -----------------------
        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(INHW),
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

        if(lid < mio_bn_config::hw)
        {
            for(unsigned int n = 0; n < mio_bn_config::n; n++)
            {
                index = n * mio_bn_config::chw + cidx + lid;
                if constexpr(mio_bn_config::n < mio_bn_config::max_n)
                {
                    dyvalues[n] = cast<FpPrecType>(dy_in[index]);

#if(MIO_BN_USESAVED == 1)
                    batchvalues[n] = (cast<FpPrecType>(x_in[index]) - mean) * invVariance;
#else
                    batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif // batchvalues is now xhat

                    dyvalues[n] = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                        dyvalues[n], batchvalues[n], pscale, pbias, alpha, beta);

                    db += dyvalues[n];
                    ds = fma(batchvalues[n], dyvalues[n], ds);
                }
                else
                {
                    FpPrecType dyvalue = cast<FpPrecType>(dy_in[index]);
                    FpPrecType xhat    = (cast<FpPrecType>(x_in[index]) - mean) * invVariance;

                    dyvalue = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                        dyvalue, xhat, pscale, pbias, alpha, beta);

                    db += dyvalue;
                    ds = fma(xhat, dyvalue, ds);
                }
            }
        }
        else
        {
            db = 0;
            ds = 0;
        }

        __syncthreads();

        miopen::reduction::reduce2<FpAccumType, mio_bn_config::lds_size>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            cast<FpAccumType>(1.0),
            lid);

        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < mio_bn_config::hw)
        {
            for(unsigned int n = 0; n < mio_bn_config::n; n++)
            {
                index = n * mio_bn_config::chw + cidx + lid;
                FpPrecType dyvalue;
                FpPrecType xhat;
                if constexpr(mio_bn_config::n < mio_bn_config::max_n)
                {
                    dyvalue = dyvalues[n];
                    xhat    = batchvalues[n];
                }
                else
                {
                    dyvalue = cast<FpPrecType>(dy_in[index]);
                    xhat    = (cast<FpPrecType>(x_in[index]) - mean) * invVariance;

                    dyvalue = bwd_activation_op<FpPrecType, mio_config::neuron_op>(
                        dyvalue, xhat, pscale, pbias, alpha, beta);
                }

                dx_out[index] = cast<FpType>(batchBwdNormalization(
                    dyvalue, xhat, db, ds, pscale, invVariance, mio_bn_config::nhw, INHW));
            }
        }
        if(lid == 0)
        {
            dbias[grpid]  = db;
            dscale[grpid] = ds;
        }
    }
};

} // namespace batchnorm
} // namespace miopen

/// C interfaces

#if(MIO_BN_VARIANT != 2)

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormBwdSpatial(const fp_type* __restrict x_in,
                              const fp_type* __restrict dy_in,
                              fp_type* __restrict dx_out,
                              const fp_prec_type* __restrict bnScale,
                              const fp_prec_type* __restrict bnBias,
                              fp_prec_type* __restrict dscale,
                              fp_prec_type* __restrict dbias,
#if(MIO_BN_USESAVED == 0)
                              double epsilon,
#elif(MIO_BN_USESAVED == 1)
                              const fp_prec_type* savedMean,
                              const fp_prec_type* savedInvVariance,
#endif
                              fp_prec_type INHW,
                              fp_prec_type alpha,
                              fp_prec_type beta)
{
    using BwdSpatialHIPImpl =
        miopen::batchnorm::MIOpenBatchNormBwdSpatialHIPImpl<mio_bn_config::variant,
                                                            fp_type,
                                                            fp_prec_type,
                                                            fp_accum_type>;

#if(MIO_BN_USESAVED == 0)
    BwdSpatialHIPImpl{}(
        x_in, dy_in, dx_out, bnScale, bnBias, dscale, dbias, epsilon, INHW, alpha, beta);
#elif(MIO_BN_USESAVED == 1)
    BwdSpatialHIPImpl{}(x_in,
                        dy_in,
                        dx_out,
                        bnScale,
                        bnBias,
                        dscale,
                        dbias,
                        savedMean,
                        savedInvVariance,
                        INHW,
                        alpha,
                        beta);
#endif
}

#else

extern "C" __global__ void
__launch_bounds__(MIO_BN_GRP0_FINAL* MIO_BN_GRP1_FINAL* MIO_BN_GRP2_FINAL)
    MIOpenBatchNormBwdSpatialFinalMeanVariance(fp_type* __restrict meanvarbuff,
                                               fp_prec_type INHW,
                                               double epsilon)
{
    unsigned int xlid    = threadIdx.x;
    unsigned int ylid    = threadIdx.y;
    unsigned int zlid    = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid    = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
    unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

    if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean     = toPrecCType(0);
    fp_prec_c_type invVariance;

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            mean += miopen::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
                0,
                MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                ystride / mio_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            variance += miopen::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
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

    if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                 (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type
            lcl_data[2 * MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL];
        miopen::reduction::lds_reduce2_2d(mean,
                                          variance,
                                          toAccumCType(INHW),
                                          lcl_data,
                                          xgrp_sz,
                                          xlid,
                                          ylid + zlid * ygrp_sz,
                                          ygrp_sz * zgrp_sz);
    }
    else
    {
        constexpr auto grp_final_total = MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL;
        miopen::reduction::reduce2<fp_accum_c_type, grp_final_total>(
            mean, variance, toAccumCType(INHW), ylid + zlid * ygrp_sz);
    }

    variance    = miopen::fma(-mean, mean, variance);
    variance    = miopen::max(variance, toPrecCType(0));
    invVariance = miopen::rsqrt(variance + toPrecCType(epsilon));

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            // Replicate mean and variance for all y groups because stash == dx_out and
            // MIOpenBatchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            miopen::batchnorm::storeToStash(mean,
                                            reinterpret_cast<fp_c_type*>(meanvarbuff),
                                            0,
                                            MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                                            MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                                            ystride / mio_bn_config::vec_size_x,
                                            xgrp_sz,
                                            xgrp_id,
                                            xlid,
                                            xstride);
            miopen::batchnorm::storeToStash(invVariance,
                                            reinterpret_cast<fp_c_type*>(meanvarbuff),
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

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormBwdSpatialMeanVariance(const fp_type* __restrict in,
                                          fp_type* __restrict meanvarbuff)
{

    unsigned int xlid    = threadIdx.x;
    unsigned int ylid    = threadIdx.y;
    unsigned int zlid    = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int ygrp_id = blockIdx.y;
    unsigned int zgrp_id = blockIdx.z;
    unsigned int xgid    = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid    = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid    = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
    unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

    if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean     = toPrecCType(0);

    if(ygid * mio_bn_config::vec_size_y < mio_bn_config::hw && zgid < mio_bn_config::n)
    {
        unsigned int index_base = zgid * MIO_BN_N_ELEMENTS * mio_bn_config::chw +
                                  ygid * ystride * mio_bn_config::vec_size_y +
                                  xgid * xstride * mio_bn_config::vec_size_x;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        {
            unsigned int index    = index_base + n * mio_bn_config::chw;
            fp_prec_ls_type value = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(in + index));

            miopen::batchnorm::_accumulate(mean, value);
            miopen::batchnorm::_accumulate_mad(variance, value, value);
        }
    }

    if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                 (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * mio_bn_config::lds_size];
        miopen::reduction::lds_reduce2_2d(mean,
                                          variance,
                                          toAccumCType(1.0),
                                          lcl_data,
                                          xgrp_sz,
                                          xlid,
                                          ylid + zlid * ygrp_sz,
                                          ygrp_sz * zgrp_sz);
    }
    else
    {
        miopen::reduction::reduce2<fp_accum_c_type, mio_bn_config::lds_size>(
            mean, variance, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        miopen::batchnorm::storeToStash(mean,
                                        reinterpret_cast<fp_c_type*>(meanvarbuff),
                                        0,
                                        zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                        ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                                        ystride / mio_bn_config::vec_size_x,
                                        xgrp_sz,
                                        xgrp_id,
                                        xlid,
                                        xstride);
        miopen::batchnorm::storeToStash(variance,
                                        reinterpret_cast<fp_c_type*>(meanvarbuff),
                                        1,
                                        zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                        ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                                        ystride / mio_bn_config::vec_size_x,
                                        xgrp_sz,
                                        xgrp_id,
                                        xlid,
                                        xstride);
    }
} // end spatial mean kernel

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormBwdSpatialDScaleDBias(const fp_type* __restrict x_in,
                                         const fp_type* __restrict dy_in,
                                         fp_type* __restrict buff,
                                         const fp_prec_type* __restrict bnScale,
                                         const fp_prec_type* __restrict bnBias,
#if MIO_BN_USESAVED == 1
                                         const fp_prec_type* __restrict savedMean,
                                         const fp_prec_type* __restrict savedInvVariance,
#endif
                                         fp_prec_type alpha,
                                         fp_prec_type beta)
{
    unsigned int xlid    = threadIdx.x;
    unsigned int ylid    = threadIdx.y;
    unsigned int zlid    = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int ygrp_id = blockIdx.y;
    unsigned int zgrp_id = blockIdx.z;
    unsigned int xgid    = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid    = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid    = blockDim.z * blockIdx.z + threadIdx.z;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
    unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

    if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias  = toPrecCType(0);
    fp_prec_c_type pscale = toPrecCType(0);
    fp_prec_c_type pbias  = toPrecCType(0);

    __shared__ fp_prec_c_type lmean[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type livar[mio_bn_config::launch_dim.grp0];
#if(MIOPEN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lcl_scale[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type lcl_bias[mio_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if MIO_BN_USESAVED == 0
        lmean[xlid] = miopen::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            0,
            zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
            ystride / mio_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = miopen::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            1,
            zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
            ystride / mio_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
#if(MIOPEN_NRN_OP_ID > 0)
        lcl_scale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
        lcl_bias[xlid]  = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    if(ygid * mio_bn_config::vec_size_y < mio_bn_config::hw && zgid < mio_bn_config::n)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
#if(MIOPEN_NRN_OP_ID > 0)
        pscale = lcl_scale[xlid];
        pbias  = lcl_bias[xlid];
#endif

        unsigned int index_base = (zgid * MIO_BN_N_ELEMENTS) * mio_bn_config::chw +
                                  ygid * ystride * mio_bn_config::vec_size_y +
                                  xgid * xstride * mio_bn_config::vec_size_x;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        {
            unsigned int index = index_base + n * mio_bn_config::chw;
            fp_prec_ls_type value1 =
                toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dy_in + index));
            fp_prec_ls_type value2 =
                toPrecLsType(*reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat = (value2 - mean) * invVar;
            // apply activation function on dy
            value1 = miopen::batchnorm::bwd_activation_op<fp_prec_ls_type, mio_config::neuron_op>(
                value1,
                xhat,
                toPrecLsType(pscale),
                toPrecLsType(pbias),
                toPrecLsType(alpha),
                toPrecLsType(beta));

            miopen::batchnorm::_accumulate(dbias, value1);
            miopen::batchnorm::_accumulate_mad(dscale, xhat, value1);
        }
    }

    if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                 (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * mio_bn_config::lds_size];
        miopen::reduction::lds_reduce2_2d(dscale,
                                          dbias,
                                          toAccumCType(1.0),
                                          lcl_data,
                                          xgrp_sz,
                                          xlid,
                                          ylid + zlid * ygrp_sz,
                                          ygrp_sz * zgrp_sz);
    }
    else
    {
        miopen::reduction::reduce2<fp_accum_c_type, mio_bn_config::lds_size>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        const unsigned int stash_index = MIO_BN_USESAVED == 1 ? 0 : 2;
        miopen::batchnorm::storeToStash(dscale,
                                        reinterpret_cast<fp_c_type*>(buff),
                                        stash_index,
                                        zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                        ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                                        ystride / mio_bn_config::vec_size_x,
                                        xgrp_sz,
                                        xgrp_id,
                                        xlid,
                                        xstride);
        miopen::batchnorm::storeToStash(dbias,
                                        reinterpret_cast<fp_c_type*>(buff),
                                        stash_index + 1,
                                        zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
                                        ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
                                        ystride / mio_bn_config::vec_size_x,
                                        xgrp_sz,
                                        xgrp_id,
                                        xlid,
                                        xstride);
    }
}

extern "C" __global__ void
__launch_bounds__(MIO_BN_GRP0_FINAL* MIO_BN_GRP1_FINAL* MIO_BN_GRP2_FINAL)
    MIOpenBatchNormBwdSpatialFinalDScaleDBias(const fp_type* __restrict buff,
                                              fp_prec_type* __restrict delta_scale,
                                              fp_prec_type* __restrict delta_bias)
{
    unsigned int xlid    = threadIdx.x;
    unsigned int ylid    = threadIdx.y;
    unsigned int zlid    = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid    = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    constexpr unsigned int xstride     = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
    constexpr unsigned int ystride     = mio_config::layout_nhwc ? mio_bn_config::c : 1;
    constexpr unsigned int stash_index = MIO_BN_USESAVED == 1 ? 0 : 2;

    if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
    {
        return;
    }

    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias  = toPrecCType(0);

    for(unsigned int zoffset = zlid; zoffset < MIO_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < MIO_BN_NGRPS; yoffset += ygrp_sz)
        {
            dscale += miopen::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index,
                MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                ystride / mio_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            dbias += miopen::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index + 1,
                MIO_BN_GRP2 * zoffset * MIO_BN_N_ELEMENTS,
                MIO_BN_GRP1 * yoffset * mio_bn_config::vec_size_y,
                ystride / mio_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    if constexpr(!mio_bn_config::use_amdgcn || mio_bn_config::launch_dim.grp0 > 1 ||
                 (mio_bn_config::lds_gcn_size == 1) || mio_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type
            lcl_data[2 * MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL];
        miopen::reduction::lds_reduce2_2d(dscale,
                                          dbias,
                                          toAccumCType(1.0),
                                          lcl_data,
                                          xgrp_sz,
                                          xlid,
                                          ylid + zlid * ygrp_sz,
                                          ygrp_sz * zgrp_sz);
    }
    else
    {
        constexpr auto grp_final_total = MIO_BN_GRP0_FINAL * MIO_BN_GRP1_FINAL * MIO_BN_GRP2_FINAL;
        miopen::reduction::reduce2<fp_accum_c_type, grp_final_total>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<fp_prec_c_type*>(delta_scale)[xgid] = dscale;
        reinterpret_cast<fp_prec_c_type*>(delta_bias)[xgid]  = dbias;
    }
}

extern "C" __global__ void __launch_bounds__(
    mio_bn_config::launch_dim.grp0* mio_bn_config::launch_dim.grp1* mio_bn_config::launch_dim.grp2)
    MIOpenBatchNormBwdSpatialDX(const fp_type* __restrict x_in,
                                const fp_type* __restrict dy_in,
                                fp_type* __restrict dx_out,
                                const fp_prec_type* __restrict bnScale,
                                const fp_prec_type* __restrict bnBias,
                                const fp_prec_type* __restrict delta_scale,
                                const fp_prec_type* __restrict delta_bias,
#if MIO_BN_USESAVED == 1
                                const fp_prec_type* __restrict savedMean,
                                const fp_prec_type* __restrict savedInvVariance,
#endif
                                fp_prec_type INHW,
                                fp_prec_type alpha,
                                fp_prec_type beta)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;

    constexpr unsigned int xstride = mio_config::layout_nhwc ? 1 : mio_bn_config::hw;
    constexpr unsigned int ystride = mio_config::layout_nhwc ? mio_bn_config::c : 1;

    if(xgid * mio_bn_config::vec_size_x >= mio_bn_config::c)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type pscale, dscale, dbias;
    fp_prec_c_type pbias = toPrecCType(0);

    __shared__ fp_prec_c_type lscale[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type ldscale[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type ldbias[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type lmean[mio_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type livar[mio_bn_config::launch_dim.grp0];
#if(MIOPEN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lbias[mio_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if MIO_BN_USESAVED == 0
        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_sz = blockDim.x;
        unsigned int ygrp_sz = blockDim.y;
        unsigned int zgrp_sz = blockDim.z;

        lmean[xlid] = miopen::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dx_out),
            0,
            zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
            ystride / mio_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = miopen::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dx_out),
            1,
            zgrp_sz * zgrp_id * MIO_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * mio_bn_config::vec_size_y,
            ystride / mio_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
        lscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
#if(MIOPEN_NRN_OP_ID > 0)
        lbias[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
        ldscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(delta_scale)[xgid];
        ldbias[xlid]  = reinterpret_cast<const fp_prec_c_type*>(delta_bias)[xgid];
    }

    __syncthreads();

    if(ygid * mio_bn_config::vec_size_y < mio_bn_config::hw && zgid < mio_bn_config::n)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
        pscale = lscale[xlid];
#if(MIOPEN_NRN_OP_ID > 0)
        pbias = lbias[xlid];
#endif
        dscale = ldscale[xlid];
        dbias  = ldbias[xlid];

        unsigned int index_base = (zgid * MIO_BN_N_ELEMENTS) * mio_bn_config::chw +
                                  ygid * ystride * mio_bn_config::vec_size_y +
                                  xgid * xstride * mio_bn_config::vec_size_x;
        for(unsigned int n = 0; n < MIO_BN_N_ELEMENTS; n++)
        { // apply normalization
            unsigned int index   = index_base + n * mio_bn_config::chw;
            fp_prec_ls_type x_i  = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat = (x_i - mean) * invVar; // recalculating this again...
            fp_prec_ls_type value1 =
                toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dy_in + index));
            value1 = miopen::batchnorm::bwd_activation_op<fp_prec_ls_type, mio_config::neuron_op>(
                value1,
                xhat,
                toPrecLsType(pscale),
                toPrecLsType(pbias),
                toPrecLsType(alpha),
                toPrecLsType(beta));

            *reinterpret_cast<fp_ls_type*>(dx_out + index) =
                toLsType(batchBwdNormalization(value1,
                                               xhat,
                                               toPrecLsType(dbias),
                                               toPrecLsType(dscale),
                                               toPrecLsType(pscale),
                                               toPrecLsType(invVar),
                                               toPrecLsType(mio_bn_config::nhw),
                                               toPrecLsType(INHW)));
        }
    }
}

#endif
