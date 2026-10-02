// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "BatchnormFunctions.hpp"
#include "ReductionFunctions.hpp"
#include "StaticUnroll.hpp"
#include "VectorTypes.hpp"

using hip_plugin_config = hip_kernel_provider::config;
using hip_plugin_bn_config = hip_kernel_provider::batchnorm::config;

// Load the configs to this file
namespace /*anonymous*/
{

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
          = typename hip_kernel_provider::MappedVectorInfo<FpPrecVecType>::UnderlyingType>
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
          = typename hip_kernel_provider::MappedVectorInfo<FpPrecVecType>::UnderlyingType>
__forceinline__ __host__ __device__ FpPrecVecType
    vectorizedBwdActivationOp(FpPrecVecType const& dy,
                              FpPrecVecType const& xnorm,
                              FpPrecType const& scale,
                              FpPrecType const& bias,
                              FpPrecType const& alpha,
                              FpPrecType const& beta)
{
    auto constexpr SIZE = hip_kernel_provider::MappedVectorInfo<FpPrecVecType>::SIZE;
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
    //NOLINTBEGIN(readability-static-accessed-through-instance)
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
    //NOLINTEND(readability-static-accessed-through-instance)

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

        FpPrecType batchvalues[NLOOP]; // NOLINT(modernize-avoid-c-arrays)
        FpPrecType dyvalues[NLOOP]; // NOLINT(modernize-avoid-c-arrays)

        __shared__ FpPrecType s_lbns;
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        __shared__ FpPrecType s_lbnb;
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        __shared__ FpPrecType s_lmean;
        __shared__ FpPrecType s_lvar;
#endif
        unsigned int index = 0;
        unsigned int lid = threadIdx.x; //NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; //NOLINT(readability-static-accessed-through-instance)
        const unsigned int chwid
            = grpid * hip_plugin_bn_config::HW + (lid % hip_plugin_bn_config::HW);
        const unsigned int lidihw = lid / hip_plugin_bn_config::HW;
        unsigned int nid = 0;

        if(lid == 0)
        {
            s_lbns = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
            s_lbnb = bnBias[grpid];
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
            for(unsigned int n = 0; n < NLOOPM; ++n)
            {
                nid = n * SEGIHW + lidihw;
                index = nid * hip_plugin_bn_config::CHW + chwid;
                batchvalues[n] = cast<FpPrecType>(xIn[index]);
                mean += batchvalues[n];
                variance = fma(batchvalues[n], batchvalues[n], variance);
            }
            nid = SNHW + lidihw;
            index = nid * hip_plugin_bn_config::CHW + chwid;
            batchvalues[NLOOPM]
                = (index < hip_plugin_bn_config::NCHW) ? cast<FpPrecType>(xIn[index]) : 0;
            mean += batchvalues[NLOOPM];
            variance = fma(batchvalues[NLOOPM], batchvalues[NLOOPM], variance);
        }

        __syncthreads();

        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(inhw),
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
        pbias = s_lbnb;
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
            batchvalues[NLOOPM] = (batchvalues[NLOOPM] - mean) * invVariance;
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
            ds, db, FpAccumType(1.0), lid);

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

    using fp_read_vec_type = typename MappedVectorType<FpType, READ_SIZE>::type;
    using fp_prec_read_vec_type = typename MappedVectorType<FpPrecType, READ_SIZE>::type;
    using fp_write_vec_type = typename MappedVectorType<FpType, WRITE_SIZE>::type;
    using fp_prec_write_vec_type = typename MappedVectorType<FpPrecType, WRITE_SIZE>::type;

    //NOLINTBEGIN(readability-static-accessed-through-instance)
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
    //NOLINTEND(readability-static-accessed-through-instance)

    __forceinline__ __device__ unsigned int getTensorIndex(unsigned int loopIndex)
    {
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int chwid = grpid * hip_plugin_bn_config::HW;
        const unsigned int nidx = loopIndex / hip_plugin_bn_config::HW;
        const unsigned int hwidx = loopIndex - (nidx * hip_plugin_bn_config::HW);
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

        const unsigned int lid
            = threadIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int grpid
            = blockIdx.x; // NOLINT(readability-static-accessed-through-instance)
        const unsigned int chwid = grpid * hip_plugin_bn_config::HW;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 0)
        //==== CALC MEAN and VARIANCE ONCE AGAIN =======================
        FpPrecType variance = 0;
        if constexpr(!hip_plugin_config::LAYOUT_NHWC && hip_plugin_bn_config::HW >= 4096)
        {
            fp_prec_read_vec_type read4;
            for(unsigned int k = lid << 2; k < LESS4; k += GRPRD)
            {
                read4 = cast<fp_prec_read_vec_type>(
                    *(reinterpret_cast<const fp_read_vec_type*>(xIn + getTensorIndex(k))));
                hip_kernel_provider::batchnorm::accumulate(mean, read4);
                hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
            }

            if constexpr(REM4 > 0)
            {
                if(lid < REM4)
                {
                    unsigned int index = getTensorIndex((lid << 2) + LESS4);
                    if(index + READ_SIZE - 1 < hip_plugin_bn_config::NCHW)
                    {
                        read4 = cast<fp_prec_read_vec_type>(
                            *(reinterpret_cast<const fp_read_vec_type*>(xIn + index)));
                        hip_kernel_provider::batchnorm::accumulate(mean, read4);
                        hip_kernel_provider::batchnorm::accumulateMad(variance, read4, read4);
                    }
                }
            }
        }
        else
        {
            for(unsigned int k = lid; k < LESS; k += hip_plugin_bn_config::LAUNCH_DIM.GRP0)
            {
                FpPrecType in = cast<FpPrecType>(xIn[getTensorIndex(k)]);
                mean += in;
                variance = fma(in, in, variance);
            }
            if constexpr(REM > 0)
            {
                if(lid < REM)
                {
                    unsigned int index = getTensorIndex(lid + LESS);
                    FpPrecType in
                        = (index < hip_plugin_bn_config::NCHW) ? cast<FpPrecType>(xIn[index]) : 0;
                    mean += in;
                    variance = fma(in, in, variance);
                }
            }
        }

        __syncthreads();

        // REDUCE MEAN AND VARIANCE -----------------------
        hip_kernel_provider::batchnorm::reduction::reduce2<FpAccumType,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(inhw),
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
            const unsigned int l = k + (lid << 2 * (1 - hip_plugin_config::LAYOUT_NHWC));
            if(l < LESS4)
            {
                const unsigned int index = getTensorIndex(l);
                const fp_read_vec_type xread
                    = *(reinterpret_cast<const fp_read_vec_type*>(xIn + index));
                const fp_read_vec_type dyRead
                    = *(reinterpret_cast<const fp_read_vec_type*>(dyIn + index));
                auto dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                const fp_prec_read_vec_type xhat
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
            const unsigned int index
                = getTensorIndex((lid << 2 * (1 - int(hip_plugin_config::LAYOUT_NHWC))) + LESS4);
            if(index + READ_SIZE - 1 < hip_plugin_bn_config::NCHW)
            {
                const fp_read_vec_type xread
                    = *(reinterpret_cast<const fp_read_vec_type*>(xIn + index));
                const fp_read_vec_type dyRead
                    = *(reinterpret_cast<const fp_read_vec_type*>(dyIn + index));
                auto dyvalue = cast<fp_prec_read_vec_type>(dyRead);
                const fp_prec_read_vec_type xhat
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
            ds, db, cast<FpAccumType>(1.0), lid);

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
            const unsigned int l = k + (WRITE_SIZE * lid);
            const unsigned int index = getTensorIndex(l);
            if(l < LESSOUT)
            {
                const fp_write_vec_type xread
                    = *(reinterpret_cast<const fp_write_vec_type*>(xIn + index));
                const fp_write_vec_type dyRead
                    = *(reinterpret_cast<const fp_write_vec_type*>(dyIn + index));
                auto value1 = cast<fp_prec_write_vec_type>(dyRead);
                const fp_prec_write_vec_type xhat1
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
            const unsigned int remkeyout = (WRITE_SIZE * lid) + LESSOUT;
            for(unsigned int j = 0; j < WRITE_SIZE; j++)
            {
                const unsigned int index = getTensorIndex(remkeyout + j);
                if(index < hip_plugin_bn_config::NCHW)
                {
                    auto value1 = cast<FpPrecType>(dyIn[index]);
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

        // Unused if hip_plugin_bn_config::N >= hip_plugin_bn_config::MAX_N
        // NOLINTBEGIN(modernize-avoid-c-arrays)
        FpPrecType batchvalues[hip_plugin_bn_config::N];
        FpPrecType dyvalues[hip_plugin_bn_config::N];
        // NOLINTEND(modernize-avoid-c-arrays)

        unsigned int lid = threadIdx.x; //NOLINT(readability-static-accessed-through-instance)
        unsigned int grpid = blockIdx.x; //NOLINT(readability-static-accessed-through-instance)
        unsigned int index;
        const unsigned int cidx = grpid * hip_plugin_bn_config::HW;

        pscale = bnScale[grpid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = bnBias[grpid];
#endif

#if(HIP_PLUGIN_BN_USESAVED == 1)
        mean = savedMean[grpid];
        invVariance = savedInvVariance[grpid];
#else // recalc mean and variance

        if(lid < hip_plugin_bn_config::HW)
        {
            for(int n = 0; n < hip_plugin_bn_config::N; n++)
            {
                index = n * hip_plugin_bn_config::CHW + cidx + lid;
                if constexpr(hip_plugin_bn_config::N < hip_plugin_bn_config::MAX_N)
                {
                    batchvalues[n] = cast<FpPrecType>(xIn[index]);
                    mean += batchvalues[n];
                    variance = fma(batchvalues[n], batchvalues[n], variance);
                }
                else
                {
                    FpPrecType in = cast<FpPrecType>(xIn[index]);
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
                                                           hip_plugin_bn_config::LDS_SIZE>(
            reinterpret_cast<FpAccumType&>(mean),
            reinterpret_cast<FpAccumType&>(variance),
            static_cast<FpAccumType>(inhw),
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
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
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
        xIn, dyIn, dxOut, bnScale, bnBias, dscale, dbias, epsilon, inhw, alpha, beta);
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
        batchNormBwdSpatialFinalMeanVariance(fp_type* __restrict meanvarbuff,
                                             fp_prec_type inhw,
                                             double epsilon)
{
    //NOLINTBEGIN(readability-static-accessed-through-instance)
    const unsigned int xlid = threadIdx.x;
    const unsigned int ylid = threadIdx.y;
    const unsigned int zlid = threadIdx.z;
    const unsigned int xGroupId = blockIdx.x;
    const unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int xGroupSize = blockDim.x;
    const unsigned int yGroupSize = blockDim.y;
    const unsigned int zGroupSize = blockDim.z;
    //NOLINTEND(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
    constexpr unsigned int YSTRIDE = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean = toPrecCType(0);
    fp_prec_c_type invVariance;

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zGroupSize)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += yGroupSize)
        {
            mean += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
                0,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
            variance += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(meanvarbuff),
                1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
        }
    }

    //NOLINTNEXTLINE(readability-static-accessed-through-instance)
    if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ fp_accum_c_type
            s_lcl_data[2 * HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL
                       * HIP_PLUGIN_BN_GRP2_FINAL];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(mean,
                                                                variance,
                                                                toAccumCType(inhw),
                                                                s_lcl_data,
                                                                xGroupSize,
                                                                xlid,
                                                                ylid + zlid * yGroupSize,
                                                                yGroupSize * zGroupSize);
    }
    else
    {
        constexpr auto GRP_FINAL_TOTAL
            = HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type, GRP_FINAL_TOTAL>(
            mean, variance, toAccumCType(inhw), ylid + zlid * yGroupSize);
    }

    variance = hip_kernel_provider::fma(-mean, mean, variance);
    variance = hip_kernel_provider::max(variance, toPrecCType(0));
    invVariance = hip_kernel_provider::rsqrt(variance + toPrecCType(epsilon));

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zGroupSize)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += yGroupSize)
        {
            // Replicate mean and variance for all y groups because stash == dxOut and
            // batchNormBwdSpatialDX will read them and rewrite the buffer entirely.
            hip_kernel_provider::batchnorm::storeToStash(
                mean,
                reinterpret_cast<fp_c_type*>(meanvarbuff),
                0,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
            hip_kernel_provider::batchnorm::storeToStash(
                invVariance,
                reinterpret_cast<fp_c_type*>(meanvarbuff),
                1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
        }
    }
}

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormBwdSpatialMeanVariance(const fp_type* __restrict in,
                                        fp_type* __restrict meanvarbuff)
{
    //NOLINTBEGIN(readability-static-accessed-through-instance)
    const unsigned int xlid = threadIdx.x;
    const unsigned int ylid = threadIdx.y;
    const unsigned int zlid = threadIdx.z;
    const unsigned int xGroupId = blockIdx.x;
    const unsigned int yGroupId = blockIdx.y;
    const unsigned int zGroupId = blockIdx.z;
    const unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;
    const unsigned int xGroupSize = blockDim.x;
    const unsigned int yGroupSize = blockDim.y;
    const unsigned int zGroupSize = blockDim.z;
    //NOLINTEND(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
    constexpr unsigned int YSTRIDE = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
    {
        return;
    }

    fp_prec_c_type variance = toPrecCType(0);
    fp_prec_c_type mean = toPrecCType(0);

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::HW
       && zgid < hip_plugin_bn_config::N)
    {
        const unsigned int indexBase = zgid * HIP_PLUGIN_BN_N_ELEMENTS * hip_plugin_bn_config::CHW
                                       + ygid * YSTRIDE * hip_plugin_bn_config::VEC_SIZE_Y
                                       + xgid * XSTRIDE * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        {
            const unsigned int index = indexBase + n * hip_plugin_bn_config::CHW;
            const fp_prec_ls_type value
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(in + index));

            hip_kernel_provider::batchnorm::accumulate(mean, value);
            hip_kernel_provider::batchnorm::accumulateMad(variance, value, value);
        }
    }

    //NOLINTNEXTLINE(readability-static-accessed-through-instance)
    if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ fp_accum_c_type s_lcl_data[2 * hip_plugin_bn_config::LDS_SIZE];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(mean,
                                                                variance,
                                                                toAccumCType(1.0),
                                                                s_lcl_data,
                                                                xGroupSize,
                                                                xlid,
                                                                ylid + zlid * yGroupSize,
                                                                yGroupSize * zGroupSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            mean, variance, toAccumCType(1.0), ylid + zlid * yGroupSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        hip_kernel_provider::batchnorm::storeToStash(
            mean,
            reinterpret_cast<fp_c_type*>(meanvarbuff),
            0,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
        hip_kernel_provider::batchnorm::storeToStash(
            variance,
            reinterpret_cast<fp_c_type*>(meanvarbuff),
            1,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
    }
} // end spatial mean kernel

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormBwdSpatialDScaleDBias(const fp_type* __restrict xIn,
                                        const fp_type* __restrict dyIn,
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
    //NOLINTBEGIN(readability-static-accessed-through-instance)
    const unsigned int xlid = threadIdx.x;
    const unsigned int ylid = threadIdx.y;
    const unsigned int zlid = threadIdx.z;
    const unsigned int xGroupId = blockIdx.x;
    const unsigned int yGroupId = blockIdx.y;
    const unsigned int zGroupId = blockIdx.z;
    const unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;
    const unsigned int xGroupSize = blockDim.x;
    const unsigned int yGroupSize = blockDim.y;
    const unsigned int zGroupSize = blockDim.z;
    //NOLINTEND(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
    constexpr unsigned int YSTRIDE = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
    {
        return;
    }

    fp_prec_c_type mean;
    fp_prec_c_type invVar;
    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias = toPrecCType(0);
    fp_prec_c_type pscale = toPrecCType(0);
    fp_prec_c_type pbias = toPrecCType(0);

    //NOLINTBEGIN(readability-static-accessed-through-instance)
    __shared__ fp_prec_c_type s_lmean[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_livar[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type s_lcl_scale[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_lcl_bias[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
#endif
    //NOLINTEND(readability-static-accessed-through-instance)

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        s_lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            0,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
        s_livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(buff),
            1,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
#else
        s_lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        s_livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        s_lcl_scale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
        s_lcl_bias[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::HW
       && zgid < hip_plugin_bn_config::N)
    {
        mean = s_lmean[xlid];
        invVar = s_livar[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pscale = s_lcl_scale[xlid];
        pbias = s_lcl_bias[xlid];
#endif

        const unsigned int indexBase = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * hip_plugin_bn_config::CHW
                                       + ygid * YSTRIDE * hip_plugin_bn_config::VEC_SIZE_Y
                                       + xgid * XSTRIDE * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        {
            const unsigned int index = indexBase + n * hip_plugin_bn_config::CHW;
            fp_prec_ls_type value1
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dyIn + index));
            const fp_prec_ls_type value2
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(xIn + index));
            const fp_prec_ls_type xhat = (value2 - mean) * invVar;
            // apply activation function on dy
            value1 = hip_kernel_provider::batchnorm::bwdActivationOp<fp_prec_ls_type,
                                                                     hip_plugin_config::NEURON_OP>(
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

    //NOLINTNEXTLINE(readability-static-accessed-through-instance)
    if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ fp_accum_c_type s_lcl_data[2 * hip_plugin_bn_config::LDS_SIZE];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(dscale,
                                                                dbias,
                                                                toAccumCType(1.0),
                                                                s_lcl_data,
                                                                xGroupSize,
                                                                xlid,
                                                                ylid + zlid * yGroupSize,
                                                                yGroupSize * zGroupSize);
    }
    else
    {
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type,
                                                           hip_plugin_bn_config::LDS_SIZE>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * yGroupSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        constexpr unsigned int STASH_INDEX = HIP_PLUGIN_BN_USESAVED == 1 ? 0 : 2;
        hip_kernel_provider::batchnorm::storeToStash(
            dscale,
            reinterpret_cast<fp_c_type*>(buff),
            STASH_INDEX,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
        hip_kernel_provider::batchnorm::storeToStash(
            dbias,
            reinterpret_cast<fp_c_type*>(buff),
            STASH_INDEX + 1,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
    }
}

extern "C" __global__ void
    __launch_bounds__(HIP_PLUGIN_BN_GRP0_FINAL* HIP_PLUGIN_BN_GRP1_FINAL* HIP_PLUGIN_BN_GRP2_FINAL)
        batchNormBwdSpatialFinalDScaleDBias(const fp_type* __restrict buff,
                                             fp_prec_type* __restrict deltaSale,
                                             fp_prec_type* __restrict deltaBias)
{
    //NOLINTBEGIN(readability-static-accessed-through-instance)
    const unsigned int xlid = threadIdx.x;
    const unsigned int ylid = threadIdx.y;
    const unsigned int zlid = threadIdx.z;
    const unsigned int xGroupId = blockIdx.x;
    const unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int xGroupSize = blockDim.x;
    const unsigned int yGroupSize = blockDim.y;
    const unsigned int zGroupSize = blockDim.z;
    //NOLINTEND(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
    constexpr unsigned int YSTRIDE = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;
    constexpr unsigned int STASH_INDEX = HIP_PLUGIN_BN_USESAVED == 1 ? 0 : 2;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
    {
        return;
    }

    fp_prec_c_type dscale = toPrecCType(0);
    fp_prec_c_type dbias = toPrecCType(0);

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zGroupSize)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += yGroupSize)
        {
            dscale += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                STASH_INDEX,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
            dbias += hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                STASH_INDEX + 1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * hip_plugin_bn_config::VEC_SIZE_Y,
                YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
                xGroupSize,
                xGroupId,
                xlid,
                XSTRIDE);
        }
    }

    //NOLINTNEXTLINE(readability-static-accessed-through-instance)
    if constexpr(!hip_plugin_bn_config::USE_AMDGCN || hip_plugin_bn_config::LAUNCH_DIM.GRP0 > 1
                 || (hip_plugin_bn_config::LDS_GCN_SIZE == 1)
                 || hip_plugin_bn_config::VEC_SIZE_X > 1)
    {
        __shared__ fp_accum_c_type
            s_lcl_data[2 * HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL
                       * HIP_PLUGIN_BN_GRP2_FINAL];
        hip_kernel_provider::batchnorm::reduction::ldsReduce22d(dscale,
                                                                dbias,
                                                                toAccumCType(1.0),
                                                                s_lcl_data,
                                                                xGroupSize,
                                                                xlid,
                                                                ylid + zlid * yGroupSize,
                                                                yGroupSize * zGroupSize);
    }
    else
    {
        constexpr auto GRP_FINAL_TOTAL
            = HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;
        hip_kernel_provider::batchnorm::reduction::reduce2<fp_accum_c_type, GRP_FINAL_TOTAL>(
            dscale, dbias, toAccumCType(1.0), ylid + zlid * yGroupSize);
    }

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<fp_prec_c_type*>(deltaSale)[xgid] = dscale;
        reinterpret_cast<fp_prec_c_type*>(deltaBias)[xgid] = dbias;
    }
}

extern "C" __global__ void
    __launch_bounds__(hip_plugin_bn_config::LAUNCH_DIM.GRP0* hip_plugin_bn_config::LAUNCH_DIM
                          .GRP1* hip_plugin_bn_config::LAUNCH_DIM.GRP2)
        batchNormBwdSpatialDX(const fp_type* __restrict xIn,
                              const fp_type* __restrict dyIn,
                              fp_type* __restrict dxOut,
                              const fp_prec_type* __restrict bnScale,
                              const fp_prec_type* __restrict bnBias,
                              const fp_prec_type* __restrict deltaSale,
                              const fp_prec_type* __restrict deltaBias,
#if HIP_PLUGIN_BN_USESAVED == 1
                              const fp_prec_type* __restrict savedMean,
                              const fp_prec_type* __restrict savedInvVariance,
#endif
                              fp_prec_type inhw,
                              fp_prec_type alpha,
                              fp_prec_type beta)
{
    //NOLINTBEGIN(readability-static-accessed-through-instance)
    const unsigned int xlid = threadIdx.x;
    const unsigned int ylid = threadIdx.y;
    const unsigned int zlid = threadIdx.z;
    const unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;
    //NOLINTEND(readability-static-accessed-through-instance)

    constexpr unsigned int XSTRIDE = hip_plugin_config::LAYOUT_NHWC ? 1 : hip_plugin_bn_config::HW;
    constexpr unsigned int YSTRIDE = hip_plugin_config::LAYOUT_NHWC ? hip_plugin_bn_config::C : 1;

    if(xgid * hip_plugin_bn_config::VEC_SIZE_X >= hip_plugin_bn_config::C)
    {
        return;
    }

    fp_prec_c_type mean;
    fp_prec_c_type invVar;
    fp_prec_c_type pscale;
    fp_prec_c_type dscale;
    fp_prec_c_type dbias;
    fp_prec_c_type pbias = toPrecCType(0);

    //NOLINTBEGIN(readability-static-accessed-through-instance)
    __shared__ fp_prec_c_type s_lscale[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_ldscale[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_ldbias[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_lmean[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
    __shared__ fp_prec_c_type s_livar[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type s_lbias[hip_plugin_bn_config::LAUNCH_DIM.GRP0];
#endif
    //NOLINTEND(readability-static-accessed-through-instance)

    if(ylid == 0 && zlid == 0)
    {
#if HIP_PLUGIN_BN_USESAVED == 0
        //NOLINTBEGIN(readability-static-accessed-through-instance)
        unsigned int xGroupId = blockIdx.x;
        unsigned int yGroupId = blockIdx.y;
        unsigned int zGroupId = blockIdx.z;

        unsigned int xGroupSize = blockDim.x;
        unsigned int yGroupSize = blockDim.y;
        unsigned int zGroupSize = blockDim.z;
        //NOLINTEND(readability-static-accessed-through-instance)

        s_lmean[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dxOut),
            0,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
        s_livar[xlid] = hip_kernel_provider::batchnorm::loadFromStash<fp_prec_c_type>(
            reinterpret_cast<const fp_c_type*>(dxOut),
            1,
            zGroupSize * zGroupId * HIP_PLUGIN_BN_N_ELEMENTS,
            yGroupSize * yGroupId * hip_plugin_bn_config::VEC_SIZE_Y,
            YSTRIDE / hip_plugin_bn_config::VEC_SIZE_X,
            xGroupSize,
            xGroupId,
            xlid,
            XSTRIDE);
#else
        s_lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        s_livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#endif
        s_lscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        s_lbias[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
        s_ldscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(deltaSale)[xgid];
        s_ldbias[xlid] = reinterpret_cast<const fp_prec_c_type*>(deltaBias)[xgid];
    }

    __syncthreads();

    if(ygid * hip_plugin_bn_config::VEC_SIZE_Y < hip_plugin_bn_config::HW
       && zgid < hip_plugin_bn_config::N)
    {
        mean = s_lmean[xlid];
        invVar = s_livar[xlid];
        pscale = s_lscale[xlid];
#if(HIP_PLUGIN_BN_NRN_OP_ID > 0)
        pbias = s_lbias[xlid];
#endif
        dscale = s_ldscale[xlid];
        dbias = s_ldbias[xlid];

        const unsigned int indexBase = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * hip_plugin_bn_config::CHW
                                       + ygid * YSTRIDE * hip_plugin_bn_config::VEC_SIZE_Y
                                       + xgid * XSTRIDE * hip_plugin_bn_config::VEC_SIZE_X;
        for(unsigned int n = 0; n < HIP_PLUGIN_BN_N_ELEMENTS; n++)
        { // apply normalization
            const unsigned int index = indexBase + n * hip_plugin_bn_config::CHW;
            const fp_prec_ls_type xI
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(xIn + index));
            const fp_prec_ls_type xhat = (xI - mean) * invVar; // recalculating this again...
            fp_prec_ls_type value1
                = toPrecLsType(*reinterpret_cast<const fp_ls_type*>(dyIn + index));
            value1 = hip_kernel_provider::batchnorm::bwdActivationOp<fp_prec_ls_type,
                                                                     hip_plugin_config::NEURON_OP>(
                value1,
                xhat,
                toPrecLsType(pscale),
                toPrecLsType(pbias),
                toPrecLsType(alpha),
                toPrecLsType(beta));

            *reinterpret_cast<fp_ls_type*>(dxOut + index)
                = toLsType(batchBwdNormalization(value1,
                                                 xhat,
                                                 toPrecLsType(dbias),
                                                 toPrecLsType(dscale),
                                                 toPrecLsType(pscale),
                                                 toPrecLsType(invVar),
                                                 toPrecLsType(hip_plugin_bn_config::NHW),
                                                 toPrecLsType(inhw)));
        }
    }
}

#endif
