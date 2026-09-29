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

using fp_type = typename hip_plugin_bn_config::fp_type;
using fp_c_type = typename hip_plugin_bn_config::fp_c_type;
using fp_prec_type = typename hip_plugin_bn_config::fp_prec_type;
using fp_accum_type = typename hip_plugin_bn_config::fp_accum_type;
using fp_accum_c_type = typename hip_plugin_bn_config::fp_accum_c_type;
using fp_prec_c_type = typename hip_plugin_bn_config::fp_prec_c_type;
using fp_ls_type = typename hip_plugin_bn_config::fp_ls_type;
using fp_prec_ls_type = typename hip_plugin_bn_config::fp_prec_ls_type;

#define SHARED_MEMORY_SCALE 64 // wave size?

template <typename T>
__forceinline__ __device__ __host__ auto toPrecLsType(T val)
{
    return hip_kernel_provider::cast<fp_prec_ls_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toPrecCType(T val)
{
    return hip_kernel_provider::cast<fp_prec_c_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toLsType(T val)
{
    return hip_kernel_provider::cast<fp_ls_type>(val);
}

template <typename T>
__forceinline__ __device__ __host__ auto toAccumCType(T val)
{
    return hip_kernel_provider::cast<fp_accum_c_type>(val);
}

template <typename FpPrecVecType,
          typename FpPrecType
          = typename hip_kernel_provider::mapped_vector_info<FpPrecVecType>::UnderlyingType>
__forceinline__ __device__ __host__ auto batchBwdNormalization(const FpPrecVecType value,
                                                               const FpPrecVecType xhat,
                                                               const FpPrecType dbias,
                                                               const FpPrecType dscale,
                                                               const FpPrecType pscale,
                                                               const FpPrecType invVariance,
                                                               const unsigned int nhw,
                                                               const FpPrecType inhw)
{
    FpPrecVecType tmp1 = hip_kernel_provider::fma(hip_kernel_provider::cast<FpPrecVecType>(nhw),
                                                  value,
                                                  hip_kernel_provider::cast<FpPrecVecType>(-dbias));
    FpPrecVecType tmp2 = -xhat * hip_kernel_provider::cast<FpPrecVecType>(dscale);
    FpPrecType tmp3 = pscale * invVariance * inhw;
    return hip_kernel_provider::cast<FpPrecVecType>(tmp3) * (tmp2 + tmp1);
}

// Specialized version for HIP_PLUGIN_BN_VARIANT 2
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
    FpPrecVecType tmp1 = hip_kernel_provider::fma(nhw, value, -dbias);
    FpPrecVecType tmp2 = -xhat * dscale;
    FpPrecVecType tmp3 = pscale * invVariance * inhw;
    return tmp3 * (tmp2 + tmp1);
}

template <typename FpPrecVecType,
          hip_kernel_provider::NeuronOpType NrnOpType,
          typename FpPrecType
          = typename hip_kernel_provider::mapped_vector_info<FpPrecVecType>::UnderlyingType>
__forceinline__ __host__ __device__ FpPrecVecType
    vectorizedBwdActivationOp(FpPrecVecType const& dy,
                              FpPrecVecType const& xnorm,
                              FpPrecType const& scale,
                              FpPrecType const& bias,
                              FpPrecType const& alpha,
                              FpPrecType const& beta)
{
    auto constexpr SIZE = hip_kernel_provider::mapped_vector_info<FpPrecVecType>::size;
    if constexpr(SIZE == 4)
    {
        FpPrecVecType out;
        out.x = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        out.z = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.z, xnorm.z, scale, bias, alpha, beta);
        out.w = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.w, xnorm.w, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(SIZE == 2)
    {
        FpPrecVecType out;
        out.x = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(SIZE == 1)
    {
        return hip_kernel_provider::batchnorm::bwdActivationOp<FpPrecType, NrnOpType>(
            dy, xnorm, scale, bias, alpha, beta);
    }
    else
    {
        static_assert(false, "Unsupported miopen vector operation.");
    }
}

} // namespace

namespace hip_kernel_provider
{
namespace batchnorm
{

template <int BnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormBwdSpatialImpl
{
    static_assert(false, "This variant is not supported.");
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormBwdSpatialImpl<0, FpType, FpPrecType, FpAccumType>
{
    static constexpr unsigned int SEGTMP1
        = hip_plugin_bn_config::LAUNCH_DIM.GRP0 / hip_plugin_bn_config::HW;
    static constexpr unsigned int SEGTMP2 = SEGTMP1 == 0 ? 1 : SEGTMP1;
    static constexpr unsigned int SEGTMP = hip_plugin_bn_config::HW * SEGTMP2;
    static constexpr unsigned int SEGMENT
        = SEGTMP > hip_plugin_bn_config::NHW ? hip_plugin_bn_config::NHW : SEGTMP;
    static constexpr unsigned int NLOOP = (hip_plugin_bn_config::NHW + SEGMENT - 1) / SEGMENT;
    static constexpr unsigned int SEGIHW = SEGMENT / hip_plugin_bn_config::HW;
    static_assert(NLOOP > 0);
    static constexpr unsigned int NLOOPM = NLOOP - 1;
    static constexpr unsigned int SNHW = NLOOPM * SEGIHW;

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict xIn,
                                                         const FpType* __restrict dyIn,
                                                         FpType* __restrict dxOut,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                                         double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType inhw,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
#if(HIP_PLUGIN_BN_USESAVED == 0)
        FpPrecType variance = 0;
#endif
        FpPrecType mean = 0;
        FpPrecType invVariance = 0;
        FpPrecType pscale = 0;
        FpPrecType pbias = 0;
        FpAccumType ds = 0;
        FpAccumType db = 0;

        FpPrecType batchvalues[NLOOP];
        FpPrecType dyvalues[NLOOP];

        __shared__ FpPrecType s_lbns;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        __shared__ FpPrecType lbnb;
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        __shared__ FpPrecType s_lmean, s_lvar;
#endif
        unsigned int index = 0;
        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * hip_plugin_bn_config::HW + (lid % hip_plugin_bn_config::HW);
        unsigned int lidihw = lid / hip_plugin_bn_config::HW;
        unsigned int nid = 0;

        if(lid == 0)
        {
            s_lbns = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
            lbnb = bnBias[grpid];
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
        if(lid < segment)
        {
            for(unsigned int n = 0; n < nloopm; ++n)
            {
                nid = n * segihw + lidihw;
                index = nid * hip_plugin_bn_config::chw + chwid;
                batchvalues[n] = cast<FpPrecType>(x_in[index]);
                mean += batchvalues[n];
                variance = fma(batchvalues[n], batchvalues[n], variance);
            }
            nid = snhw + lidihw;
            index = nid * hip_plugin_bn_config::chw + chwid;
            batchvalues[nloopm]
                = (index < hip_plugin_bn_config::nchw) ? cast<FpPrecType>(x_in[index]) : 0;
            mean += batchvalues[nloopm];
            variance = fma(batchvalues[nloopm], batchvalues[nloopm], variance);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::lds_size>(
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

#endif // end -- Recalc mean and variance \
    //-------------------------------------------
        pscale = s_lbns;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = lbnb;
#endif

        //==== CALC DB and DS =========================================
        if(lid < SEGMENT)
        {
            for(unsigned int n = 0; n < NLOOPM; ++n)
            {
                nid = n * SEGIHW + lidihw;
                index = nid * hip_plugin_bn_config::CHW + chwid;
                dyvalues[n] = cast<FpPrecType>(dyIn[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                batchvalues[n] = (cast<FpPrecType>(xIn[index]) - mean) * invVariance;
#else
                batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif
                dyvalues[n] = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
                    dyvalues[n], batchvalues[n], pscale, pbias, alpha, beta);
                // batchvalues is now xhat
                db += dyvalues[n];
                ds = fma(batchvalues[n], dyvalues[n], ds);
            }
            nid = SNHW + lidihw;
            index = nid * hip_plugin_bn_config::CHW + chwid;
            dyvalues[NLOOPM]
                = ((index < hip_plugin_bn_config::NCHW) ? cast<FpPrecType>(dyIn[index]) : 0);

#if(HIP_PLUGIN_BN_USESAVED == 1)
            batchvalues[NLOOPM] = (index < hip_plugin_bn_config::NCHW)
                                      ? ((cast<FpPrecType>(xIn[index]) - mean) * invVariance)
                                      : 0;
#else
            batchvalues[nloopm] = (batchvalues[nloopm] - mean) * invVariance;
#endif
            dyvalues[NLOOPM] = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
                dyvalues[NLOOPM], batchvalues[NLOOPM], pscale, pbias, alpha, beta);
            // batchvalues is now xhat
            db += dyvalues[NLOOPM];
            ds = fma(batchvalues[NLOOPM], dyvalues[NLOOPM], ds);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            FpAccumType(1.0),
            lid);

        if(lid < SEGMENT)
        {
            //==== CALC NORM =======================
            FpPrecType value;
            for(unsigned int n = 0; n < NLOOPM; n++)
            {
                nid = n * SEGIHW + lidihw;
                index = nid * hip_plugin_bn_config::CHW + chwid;
                dxOut[index] = cast<FpType>(batchBwdNormalization(dyvalues[n],
                                                                   batchvalues[n],
                                                                   cast<FpPrecType>(db),
                                                                   cast<FpPrecType>(ds),
                                                                   pscale,
                                                                   invVariance,
                                                                   hip_plugin_bn_config::NHW,
                                                                   inhw));
            } // end for
            nid = SNHW + lidihw;
            index = nid * hip_plugin_bn_config::CHW + chwid;
            if(index < hip_plugin_bn_config::NCHW)
            {
                dxOut[index] = cast<FpType>(batchBwdNormalization(dyvalues[NLOOPM],
                                                                   batchvalues[NLOOPM],
                                                                   cast<FpPrecType>(db),
                                                                   cast<FpPrecType>(ds),
                                                                   pscale,
                                                                   invVariance,
                                                                   hip_plugin_bn_config::NHW,
                                                                   inhw));
            }
        }
        if(lid == 0)
        {
            dbias[grpid] = cast<FpPrecType>(db);
            dscale[grpid] = cast<FpPrecType>(ds);
        }
    }
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormBwdSpatialImpl<1, FpType, FpPrecType, FpAccumType>
{
    static constexpr unsigned int READ_SIZE = hip_plugin_config::LAYOUT_NHWC ? 1 : 4;
    static constexpr unsigned int WRITE_SIZE = hip_plugin_config::LAYOUT_NHWC ? 1 : 2;

    using fp_read_vec_type = typename mapped_vector_type<FpType, READ_SIZE>::type;
    using fp_prec_read_vec_type = typename mapped_vector_type<FpPrecType, READ_SIZE>::type;
    using fp_write_vec_type = typename mapped_vector_type<FpType, WRITE_SIZE>::type;
    using fp_prec_write_vec_type = typename mapped_vector_type<FpPrecType, WRITE_SIZE>::type;

    static constexpr unsigned int RD_BLK = 1;
    static constexpr unsigned int GRPRD
        = hip_plugin_bn_config::LAUNCH_DIM.GRP0 * RD_BLK * READ_SIZE;
    static constexpr unsigned int REM4
        = hip_plugin_bn_config::NHW - (hip_plugin_bn_config::NHW / GRPRD) * GRPRD;
    static constexpr unsigned int LESS4 = hip_plugin_bn_config::NHW - REM4;
    static constexpr unsigned int REM
        = hip_plugin_bn_config::NHW
          - (hip_plugin_bn_config::NHW / hip_plugin_bn_config::LAUNCH_DIM.GRP0)
                * hip_plugin_bn_config::LAUNCH_DIM.GRP0;
    static constexpr unsigned int LESS = hip_plugin_bn_config::NHW - REM;
    static constexpr unsigned int CHUNK = WRITE_SIZE * hip_plugin_bn_config::LAUNCH_DIM.GRP0;
    static constexpr unsigned int REMOUT
        = hip_plugin_bn_config::NHW - ((hip_plugin_bn_config::NHW / CHUNK) * CHUNK);
    static constexpr unsigned int LESSOUT = hip_plugin_bn_config::NHW - REMOUT;

    __forceinline__ __device__ unsigned int getTensorIndex(unsigned int loopIndex)
    {
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * hip_plugin_bn_config::HW;
        unsigned int nidx = loopIndex / hip_plugin_bn_config::HW;
        unsigned int hwidx = loopIndex - (nidx * hip_plugin_bn_config::HW);
        return hip_plugin_config::LAYOUT_NHWC
                   ? nidx * hip_plugin_bn_config::CHW + hwidx * hip_plugin_bn_config::C + grpid
                   : nidx * hip_plugin_bn_config::CHW + chwid + hwidx;
    }

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict xIn,
                                                         const FpType* __restrict dyIn,
                                                         FpType* __restrict dxOut,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                                         double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType inhw,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType mean = 0;
        FpPrecType invVariance = 0;
        FpPrecType pscale = 0;
        FpPrecType pbias = 0;
        FpAccumType db = 0;
        FpAccumType ds = 0;
        FpPrecType xhat = 0;

        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int chwid = grpid * hip_plugin_bn_config::HW;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 0)
        //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
        FpPrecType variance = 0;
        if constexpr(!hip_plugin_config::layout_nhwc && hip_plugin_bn_config::hw >= 4096)
        {
            fp_prec_read_vec_type read4;
            for(unsigned int k = lid << 2; k < less4; k += grprd)
            {
                read4 = cast<fp_prec_read_vec_type>(
                    *(reinterpret_cast<const fp_read_vec_type*>(x_in + getTensorIndex(k))));
                hip_kernel_provider::batchnorm::accumulate(mean, read4);
                hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
            }

            if constexpr(rem4 > 0)
            {
                if(lid < rem4)
                {
                    unsigned int index = getTensorIndex((lid << 2) + less4);
                    if(index + read_size - 1 < hip_plugin_bn_config::nchw)
                    {
                        read4 = cast<fp_prec_read_vec_type>(
                            *(reinterpret_cast<const fp_read_vec_type*>(x_in + index)));
                        hip_kernel_provider::batchnorm::accumulate(mean, read4);
                        hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                    }
                }
            }
        }
        else
        {
            for(unsigned int k = lid; k < less; k += hip_plugin_bn_config::launch_dim.grp0)
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
                    FpPrecType in
                        = (index < hip_plugin_bn_config::nchw) ? cast<FpPrecType>(x_in[index]) : 0;
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::lds_size>(
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

#else // HIP_PLUGIN_BN_USESAVED == 1
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#endif

        constexpr unsigned int READ_UNROLL_HINT
            = hip_plugin_bn_config::N > hip_plugin_bn_config::LOOP_UNROLL_MAX_N ? 4 : 2;
        StaticUnrollCount<unsigned int, 0, LESS4, GRPRD, READ_UNROLL_HINT>{[&](unsigned int k) {
            unsigned int l = k + (lid << 2 * (1 - hip_plugin_config::LAYOUT_NHWC));
            if(l < LESS4)
            {
                unsigned int index = getTensorIndex(l);
                fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(xIn + index));
                fp_read_vec_type dyRead
                    = *(reinterpret_cast<const fp_read_vec_type*>(dyIn + index));
                fp_prec_read_vec_type dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                fp_prec_read_vec_type xhat
                    = (cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

                dyvalue = vectorizedBwdActivationOp<fp_prec_read_vec_type,
                                                    hip_plugin_config::NEURON_OP>(
                    dyvalue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::accumulate(db, dyvalue);
                hip_kernel_provider::batchnorm::accumulateMad(ds, xhat, dyvalue);
            }
        }};

        if constexpr(REM4 > 0)
        {
            unsigned int index
                = getTensorIndex((lid << 2 * (1 - hip_plugin_config::LAYOUT_NHWC)) + LESS4);
            if(index + READ_SIZE - 1 < hip_plugin_bn_config::NCHW)
            {
                fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(xIn + index));
                fp_read_vec_type dyRead
                    = *(reinterpret_cast<const fp_read_vec_type*>(dyIn + index));
                fp_prec_read_vec_type dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                fp_prec_read_vec_type xhat
                    = (cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

                dyvalue = vectorizedBwdActivationOp<fp_prec_read_vec_type,
                                                    hip_plugin_config::NEURON_OP>(
                    dyvalue, xhat, pscale, pbias, alpha, beta);

                hip_kernel_provider::batchnorm::accumulate(db, dyvalue);
                hip_kernel_provider::batchnorm::accumulateMad(ds, xhat, dyvalue);
            }
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            cast<FpAccumType>(1.0),
            lid);

        __syncthreads();

        if(lid == 0)
        {
            dbias[grpid] = cast<FpPrecType>(db);
            dscale[grpid] = cast<FpPrecType>(ds);
        }

        constexpr unsigned int WRITE_UNROLL_HINT
            = hip_plugin_bn_config::N > hip_plugin_bn_config::LOOP_UNROLL_MAX_N ? 2 : 1;
        StaticUnrollCount<unsigned int, 0, LESSOUT, CHUNK, WRITE_UNROLL_HINT>{[&](unsigned int k) {
            // Unrolling the loop requires forcing explicit data vectorization otherwise the
            // compiler will start splitting global loads into smaller chunks resulting in
            // significant slowdown.
            fp_prec_write_vec_type vals;
            unsigned int l = k + (WRITE_SIZE * lid);
            unsigned int index = getTensorIndex(l);
            if(l < LESSOUT)
            {
                fp_write_vec_type xread
                    = *(reinterpret_cast<const fp_write_vec_type*>(xIn + index));
                fp_write_vec_type dyRead
                    = *(reinterpret_cast<const fp_write_vec_type*>(dyIn + index));
                fp_prec_write_vec_type value1 = cast<fp_prec_write_vec_type>(dyRead);
                fp_prec_write_vec_type xhat1
                    = (cast<fp_prec_write_vec_type>(xread) - mean) * invVariance;

                value1 = vectorizedBwdActivationOp<fp_prec_write_vec_type,
                                                   hip_plugin_config::NEURON_OP>(
                    value1, xhat1, pscale, pbias, alpha, beta);

                vals = batchBwdNormalization(value1,
                                             xhat1,
                                             cast<FpPrecType>(db),
                                             cast<FpPrecType>(ds),
                                             pscale,
                                             invVariance,
                                             hip_plugin_bn_config::NHW,
                                             inhw);
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

            if(l < LESSOUT)
            {
                *reinterpret_cast<fp_write_vec_type*>(dxOut + index)
                    = cast<fp_write_vec_type>(vals);
            }
        }};

        if constexpr(REMOUT > 0)
        {
            unsigned int remkeyout = (WRITE_SIZE * lid) + LESSOUT;
            for(unsigned int j = 0; j < WRITE_SIZE; j++)
            {
                unsigned int index = getTensorIndex(remkeyout + j);
                if(index < hip_plugin_bn_config::NCHW)
                {
                    FpPrecType value1 = cast<FpPrecType>(dyIn[index]);
                    FpPrecType xhat = (cast<FpPrecType>(xIn[index]) - mean) * invVariance;

                    value1 = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
                        value1, xhat, pscale, pbias, alpha, beta);

                    dxOut[index] = cast<FpType>(batchBwdNormalization(value1,
                                                                       xhat,
                                                                       cast<FpPrecType>(db),
                                                                       cast<FpPrecType>(ds),
                                                                       pscale,
                                                                       invVariance,
                                                                       hip_plugin_bn_config::NHW,
                                                                       inhw));
                }
            }
        }
    }
};

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct BatchNormBwdSpatialImpl<3, FpType, FpPrecType, FpAccumType>
{

    constexpr __forceinline__ __device__ void operator()(const FpType* __restrict xIn,
                                                         const FpType* __restrict dyIn,
                                                         FpType* __restrict dxOut,
                                                         const FpPrecType* __restrict bnScale,
                                                         const FpPrecType* __restrict bnBias,
                                                         FpPrecType* __restrict dscale,
                                                         FpPrecType* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                                                         double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                                                         const FpPrecType* savedMean,
                                                         const FpPrecType* savedInvVariance,
#endif
                                                         FpPrecType inhw,
                                                         FpPrecType alpha,
                                                         FpPrecType beta)
    {
        FpPrecType mean = 0;
#if(HIP_PLUGIN_BN_USESAVED == 0)
        FpPrecType variance = 0;
#endif
        FpPrecType invVariance = 0;
        FpPrecType pscale = 0;
        FpPrecType pbias = 0;
        FpPrecType ds = 0;
        FpPrecType db = 0;

        // Unused if hip_plugin_bn_config::n >= hip_plugin_bn_config::max_n
        FpPrecType batchvalues[hip_plugin_bn_config::N];
        FpPrecType dyvalues[hip_plugin_bn_config::N];

        unsigned int lid = threadIdx.x;
        unsigned int grpid = blockIdx.x;
        unsigned int index;
        unsigned int cidx = grpid * hip_plugin_bn_config::HW;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#else // recalc mean and variance

        if(lid < hip_plugin_bn_config::hw)
        {
            for(int n = 0; n < hip_plugin_bn_config::n; n++)
            {
                index = n * hip_plugin_bn_config::chw + cidx + lid;
                if constexpr(hip_plugin_bn_config::n < hip_plugin_bn_config::max_n)
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
            mean = 0;
            variance = 0;
        }

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::lds_size>(
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

        if(lid < hip_plugin_bn_config::HW)
        {
            for(unsigned int n = 0; n < hip_plugin_bn_config::N; n++)
            {
                index = n * hip_plugin_bn_config::CHW + cidx + lid;
                if constexpr(hip_plugin_bn_config::N < hip_plugin_bn_config::MAX_N)
                {
                    dyvalues[n] = cast<FpPrecType>(dyIn[index]);

#if(HIP_PLUGIN_BN_USESAVED == 1)
                    batchvalues[n] = (cast<FpPrecType>(xIn[index]) - mean) * invVariance;
#else
                    batchvalues[n] = (batchvalues[n] - mean) * invVariance;
#endif // batchvalues is now xhat

                    dyvalues[n] = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
                        dyvalues[n], batchvalues[n], pscale, pbias, alpha, beta);

                    db += dyvalues[n];
                    ds = fma(batchvalues[n], dyvalues[n], ds);
                }
                else
                {
                    FpPrecType dyvalue = cast<FpPrecType>(dyIn[index]);
                    FpPrecType xhat = (cast<FpPrecType>(xIn[index]) - mean) * invVariance;

                    dyvalue = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
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

        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(ds),
            reinterpret_cast<FpAccumType&>(db),
            cast<FpAccumType>(1.0),
            lid);

        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < hip_plugin_bn_config::HW)
        {
            for(unsigned int n = 0; n < hip_plugin_bn_config::N; n++)
            {
                index = n * hip_plugin_bn_config::CHW + cidx + lid;
                FpPrecType dyvalue;
                FpPrecType xhat;
                if constexpr(hip_plugin_bn_config::N < hip_plugin_bn_config::MAX_N)
                {
                    dyvalue = dyvalues[n];
                    xhat = batchvalues[n];
                }
                else
                {
                    dyvalue = cast<FpPrecType>(dyIn[index]);
                    xhat = (cast<FpPrecType>(xIn[index]) - mean) * invVariance;

                    dyvalue = bwdActivationOp<FpPrecType, hip_plugin_config::NEURON_OP>(
                        dyvalue, xhat, pscale, pbias, alpha, beta);
                }

                dxOut[index] = cast<FpType>(batchBwdNormalization(
                    dyvalue, xhat, db, ds, pscale, invVariance, hip_plugin_bn_config::NHW, inhw));
            }
        }
        if(lid == 0)
        {
            dbias[grpid] = db;
            dscale[grpid] = ds;
        }
    }
};

} // namespace batchnorm
} // namespace hip_kernel_provider

/// C interfaces

#if(HIP_PLUGIN_BN_VARIANT != 2)

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::launch_dim.grp0* hip_plugin_bn_config::launch_dim
                          .grp1* hip_plugin_bn_config::launch_dim.grp2)
        batchNormBwdSpatial(const fp_type* __restrict xIn,
                            const fp_type* __restrict dyIn,
                            fp_type* __restrict dxOut,
                            const fp_prec_type* __restrict bnScale,
                            const fp_prec_type* __restrict bnBias,
                            fp_prec_type* __restrict dscale,
                            fp_prec_type* __restrict dbias,
#if(HIP_PLUGIN_BN_USESAVED == 0)
                            double epsilon,
#elif(HIP_PLUGIN_BN_USESAVED == 1)
                            const fp_prec_type* savedMean,
                            const fp_prec_type* savedInvVariance,
#endif
                            fp_prec_type inhw,
                            fp_prec_type alpha,
                            fp_prec_type beta)
{
    using BwdSpatialHIPImpl
        = hip_kernel_provider::batchnorm::BatchNormBwdSpatialImpl<hip_plugin_bn_config::VARIANT,
                                                                  fp_type,
                                                                  fp_prec_type,
                                                                  fp_accum_type>;

#if(HIP_PLUGIN_BN_USESAVED == 0)
    BwdSpatialHIPImpl{}(
        x_in, dy_in, dx_out, bnScale, bnBias, dscale, dbias, epsilon, INHW, alpha, beta);
#elif(HIP_PLUGIN_BN_USESAVED == 1)
    BwdSpatialHIPImpl{}(xIn,
                        dyIn,
                        dxOut,
                        bnScale,
                        bnBias,
                        dscale,
                        dbias,
                        savedMean,
                        savedInvVariance,
                        inhw,
                        alpha,
                        beta);
#endif
}

#else

extern "C" __global__ void
    __launch_bounds__(HIP_PLUGIN_BN_GRP0_FINAL* HIP_PLUGIN_BN_GRP1_FINAL* HIP_PLUGIN_BN_GRP2_FINAL)
        BatchNormBwdSpatialFinalMeanVariance(fp_type* __restrict meanvarbuff,
                                             fp_prec_type INHW,
                                             double epsilon)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = hip_plugin_config::layout_nhwc ? 1 : hip_plugin_bn_config::hw;
    unsigned int ystride = hip_plugin_config::layout_nhwc ? hip_plugin_bn_config::c : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::c)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean = toPrecCType(0);
    fp_prec_c_type invVariance;

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += ygrp_sz)
        {
            mean += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
                0,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            variance += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
                1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL
                                            * HIP_PLUGIN_BN_GRP2_FINAL];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(mean,
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
        constexpr auto grp_final_total
            = HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type, grp_final_total>(
            mean, variance, toAccumCType(INHW), ylid + zlid * ygrp_sz);
    }

    variance = hip_kernel_provider::fma(-mean, mean, variance);
    variance = hip_kernel_provider::max(variance, toPrecCType(0));
    invVariance = hip_kernel_provider::rsqrt(variance + toPrecCType(epsilon));

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += ygrp_sz)
        {
            // Replicate mean and variance for all y groups because stash == dx_out and
            // BatchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            hip_kernel_provider::batchnorm::storeToStash(
                mean,
                reinterpret_cast<fp_c_type*>(meanvarbuff),
                0,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            hip_kernel_provider::batchnorm::storeToStash(
                invVariance,
                reinterpret_cast<fp_c_type*>(meanvarbuff),
                1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
    }
}

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::launch_dim.grp0* hip_plugin_bn_config::launch_dim
                          .grp1* hip_plugin_bn_config::launch_dim.grp2)
        BatchNormBwdSpatialMeanVariance(const fp_type* __restrict in,
                                        fp_type* __restrict meanvarbuff)
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
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = hip_plugin_config::layout_nhwc ? 1 : hip_plugin_bn_config::hw;
    unsigned int ystride = hip_plugin_config::layout_nhwc ? hip_plugin_bn_config::c : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::c)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean = toPrecCType(0);

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::hw
       && zgid < hip_plugin_bn_config::n)
    {
        unsigned int index_base = zgid * HIP_PLUGIN_BN_N_ELEMENTS * hip_plugin_bn_config::chw
                                  + ygid * ystride * hip_plugin_bn_config::vec_size_y
                                  + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        {
            unsigned int index = index_base + n * hip_plugin_bn_config::chw;
            fp_prec_ls_type value = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(in + index));

            hip_kernel_provider::batchnorm::accumulate(mean, value);
            hip_kernel_provider::batchnorm::accumulateMad(variance, value, value);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * hip_plugin_bn_config::lds_size];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(mean,
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
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type,
                                                           hip_plugin_bn_config::lds_size>(
            mean, variance, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        hip_kernel_provider::batchnorm::storeToStash(mean,
                                                     reinterpret_cast<fp_c_type*>(meanvarbuff),
                                                     0,
                                                     zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                                                     ygrp_sz * ygrp_id
                                                         * hip_plugin_bn_config::vec_size_y,
                                                     ystride / hip_plugin_bn_config::vec_size_x,
                                                     xgrp_sz,
                                                     xgrp_id,
                                                     xlid,
                                                     xstride);
        hip_kernel_provider::batchnorm::storeToStash(variance,
                                                     reinterpret_cast<fp_c_type*>(meanvarbuff),
                                                     1,
                                                     zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                                                     ygrp_sz * ygrp_id
                                                         * hip_plugin_bn_config::vec_size_y,
                                                     ystride / hip_plugin_bn_config::vec_size_x,
                                                     xgrp_sz,
                                                     xgrp_id,
                                                     xlid,
                                                     xstride);
    }
} // end spatial mean kernel

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::launch_dim.grp0* hip_plugin_bn_config::launch_dim
                          .grp1* hip_plugin_bn_config::launch_dim.grp2)
        BatchNormBwdSpatialDScaleDBias(const fp_type* __restrict x_in,
                                       const fp_type* __restrict dy_in,
                                       fp_type* __restrict buff,
                                       const fp_prec_type* __restrict bnScale,
                                       const fp_prec_type* __restrict bnBias,
#if HIP_PLUGIN_BN_USESAVED == 1
                                       const fp_prec_type* __restrict savedMean,
                                       const fp_prec_type* __restrict savedInvVariance,
#endif
                                       fp_prec_type alpha,
                                       fp_prec_type beta)
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
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    unsigned int xstride = hip_plugin_config::layout_nhwc ? 1 : hip_plugin_bn_config::hw;
    unsigned int ystride = hip_plugin_config::layout_nhwc ? hip_plugin_bn_config::c : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::c)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias = toPrecCType(0);
    fp_prec_c_type pscale = toPrecCType(0);
    fp_prec_c_type pbias = toPrecCType(0);

    __shared__ fp_prec_c_type lmean[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type livar[hip_plugin_bn_config::launch_dim.grp0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lcl_scale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type lcl_bias[hip_plugin_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            0,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            1,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        lcl_scale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
        lcl_bias[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::hw
       && zgid < hip_plugin_bn_config::n)
    {
        mean = lmean[xlid];
        invVar = livar[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pscale = lcl_scale[xlid];
        pbias = lcl_bias[xlid];
#endif

        unsigned int index_base = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * hip_plugin_bn_config::chw
                                  + ygid * ystride * hip_plugin_bn_config::vec_size_y
                                  + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        {
            unsigned int index = index_base + n * hip_plugin_bn_config::chw;
            fp_prec_ls_type value1
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dy_in + index));
            fp_prec_ls_type value2
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat = (value2 - mean) * invVar;
            // apply activation function on dy
            value1
                = hip_kernel_provider::batchnorm::bwdActivationOp<fp_prec_ls_type,
                                                                    hip_plugin_config::neuron_op>(
                    value1,
                    xhat,
                    toPrecLsType(pscale),
                    toPrecLsType(pbias),
                    toPrecLsType(alpha),
                    toPrecLsType(beta));

            hip_kernel_provider::batchnorm::accumulate(dbias, value1);
            hip_kernel_provider::batchnorm::accumulateMad(dscale, xhat, value1);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * hip_plugin_bn_config::lds_size];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(dscale,
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
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type,
                                                           hip_plugin_bn_config::lds_size>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        const unsigned int stash_index = HIP_PLUGIN_BN_USESAVED == 1 ? 0 : 2;
        hip_kernel_provider::batchnorm::storeToStash(dscale,
                                                     reinterpret_cast<fp_c_type*>(buff),
                                                     stash_index,
                                                     zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                                                     ygrp_sz * ygrp_id
                                                         * hip_plugin_bn_config::vec_size_y,
                                                     ystride / hip_plugin_bn_config::vec_size_x,
                                                     xgrp_sz,
                                                     xgrp_id,
                                                     xlid,
                                                     xstride);
        hip_kernel_provider::batchnorm::storeToStash(dbias,
                                                     reinterpret_cast<fp_c_type*>(buff),
                                                     stash_index + 1,
                                                     zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
                                                     ygrp_sz * ygrp_id
                                                         * hip_plugin_bn_config::vec_size_y,
                                                     ystride / hip_plugin_bn_config::vec_size_x,
                                                     xgrp_sz,
                                                     xgrp_id,
                                                     xlid,
                                                     xstride);
    }
}

extern "C" __global__ void
    __launch_bounds__(HIP_PLUGIN_BN_GRP0_FINAL* HIP_PLUGIN_BN_GRP1_FINAL* HIP_PLUGIN_BN_GRP2_FINAL)
        BatchNormBwdSpatialFinalDScaleDBias(const fp_type* __restrict buff,
                                            fp_prec_type* __restrict delta_scale,
                                            fp_prec_type* __restrict delta_bias)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    constexpr unsigned int xstride = hip_plugin_config::layout_nhwc ? 1 : hip_plugin_bn_config::hw;
    constexpr unsigned int ystride = hip_plugin_config::layout_nhwc ? hip_plugin_bn_config::c : 1;
    constexpr unsigned int stash_index = HIP_PLUGIN_BN_USESAVED == 1 ? 0 : 2;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::c)
    {
        return;
    }

    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias = toPrecCType(0);

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += ygrp_sz)
        {
            dscale += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            dbias += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index + 1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::vec_size_y,
                ystride / hip_plugin_bn_config::vec_size_x,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    if constexpr(!hip_plugin_bn_config::use_amdgcn || hip_plugin_bn_config::launch_dim.grp0 > 1
                 || (hip_plugin_bn_config::lds_gcn_size == 1)
                 || hip_plugin_bn_config::vec_size_x > 1)
    {
        __shared__ fp_accum_c_type lcl_data[2 * HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL
                                            * HIP_PLUGIN_BN_GRP2_FINAL];
        hip_kernel_provider::batchnorm::reduction::lds_reduce2_2d(dscale,
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
        constexpr auto grp_final_total
            = HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type, grp_final_total>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * ygrp_sz);
    }

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<fp_prec_c_type*>(delta_scale)[xgid] = dscale;
        reinterpret_cast<fp_prec_c_type*>(delta_bias)[xgid] = dbias;
    }
}

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::launch_dim.grp0* hip_plugin_bn_config::launch_dim
                          .grp1* hip_plugin_bn_config::launch_dim.grp2)
        BatchNormBwdSpatialDX(const fp_type* __restrict x_in,
                              const fp_type* __restrict dy_in,
                              fp_type* __restrict dx_out,
                              const fp_prec_type* __restrict bnScale,
                              const fp_prec_type* __restrict bnBias,
                              const fp_prec_type* __restrict delta_scale,
                              const fp_prec_type* __restrict delta_bias,
#if HIP_PLUGIN_BN_USESAVED == 1
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

    constexpr unsigned int xstride = hip_plugin_config::layout_nhwc ? 1 : hip_plugin_bn_config::hw;
    constexpr unsigned int ystride = hip_plugin_config::layout_nhwc ? hip_plugin_bn_config::c : 1;

    if(xgid * hip_plugin_bn_config::vec_size_x >= hip_plugin_bn_config::c)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type pscale, dscale, dbias;
    fp_prec_c_type pbias = toPrecCType(0);

    __shared__ fp_prec_c_type lscale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type ldscale[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type ldbias[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type lmean[hip_plugin_bn_config::launch_dim.grp0];
    __shared__ fp_prec_c_type livar[hip_plugin_bn_config::launch_dim.grp0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lbias[hip_plugin_bn_config::launch_dim.grp0];
#endif

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        unsigned int xgrp_id = blockIdx.x;
        unsigned int ygrp_id = blockIdx.y;
        unsigned int zgrp_id = blockIdx.z;

        unsigned int xgrp_sz = blockDim.x;
        unsigned int ygrp_sz = blockDim.y;
        unsigned int zgrp_sz = blockDim.z;

        lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dx_out),
            0,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dx_out),
            1,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * hip_plugin_bn_config::vec_size_y,
            ystride / hip_plugin_bn_config::vec_size_x,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
#else
        lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
        lscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        lbias[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
        ldscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(delta_scale)[xgid];
        ldbias[xlid] = reinterpret_cast<const fp_prec_c_type*>(delta_bias)[xgid];
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::vec_size_y < hip_plugin_bn_config::hw
       && zgid < hip_plugin_bn_config::n)
    {
        mean = lmean[xlid];
        invVar = livar[xlid];
        pscale = lscale[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = lbias[xlid];
#endif
        dscale = ldscale[xlid];
        dbias = ldbias[xlid];

        unsigned int index_base = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * hip_plugin_bn_config::chw
                                  + ygid * ystride * hip_plugin_bn_config::vec_size_y
                                  + xgid * xstride * hip_plugin_bn_config::vec_size_x;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        { // apply normalization
            unsigned int index = index_base + n * hip_plugin_bn_config::chw;
            fp_prec_ls_type x_i = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat = (x_i - mean) * invVar; // recalculating this again...
            fp_prec_ls_type value1
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dy_in + index));
            value1
                = hip_kernel_provider::batchnorm::bwdActivationOp<fp_prec_ls_type,
                                                                    hip_plugin_config::neuron_op>(
                    value1,
                    xhat,
                    toPrecLsType(pscale),
                    toPrecLsType(pbias),
                    toPrecLsType(alpha),
                    toPrecLsType(beta));

            *reinterpret_cast<fp_ls_type*>(dx_out + index)
                = toLsType(batchBwdNormalization(value1,
                                                 xhat,
                                                 toPrecLsType(dbias),
                                                 toPrecLsType(dscale),
                                                 toPrecLsType(pscale),
                                                 toPrecLsType(invVar),
                                                 toPrecLsType(hip_plugin_bn_config::nhw),
                                                 toPrecLsType(INHW)));
        }
    }
}

#endif
