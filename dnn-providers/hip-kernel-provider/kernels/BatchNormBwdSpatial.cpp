// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "BatchnormActivation.hpp"
#include "FloatTypes.h"
#include "ReductionFunctions.hpp"
#include "StaticUnroll.hpp"
#include "VectorTypes.hpp"

constexpr unsigned int blockSize = HIP_PLUGIN_BN_GRP0;
constexpr unsigned int ldsSize   = HIP_PLUGIN_BN_LDS_SIZE;

// ============================================================================
// Variant 0: Threads mapped to (n_segment, hw), pre-load into arrays.
//            NCHW only, good for small HW.
// ============================================================================
#if(HIP_PLUGIN_BN_VARIANT == 0)

constexpr unsigned int segtmp1 = blockSize / HIP_PLUGIN_BN_HW;
constexpr unsigned int segtmp2 = segtmp1 == 0 ? 1 : segtmp1;
constexpr unsigned int segtmp  = HIP_PLUGIN_BN_HW * segtmp2;
constexpr unsigned int segment =
    segtmp > HIP_PLUGIN_BN_NHW ? HIP_PLUGIN_BN_NHW : segtmp;
constexpr unsigned int nloop  = (HIP_PLUGIN_BN_NHW + segment - 1) / segment;
static_assert(nloop > 0);
constexpr unsigned int nloopm = nloop - 1;
constexpr unsigned int segihw = segment / HIP_PLUGIN_BN_HW;
constexpr unsigned int snhw   = nloopm * segihw;

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormBwdSpatialSaved(const FLOAT* __restrict x_in,
                             const FLOAT* __restrict dy_in,
                             FLOAT* __restrict dx_out,
                             const FLOAT_ACCUM* __restrict bnScale,
                             const FLOAT_ACCUM* __restrict bnBias,
                             FLOAT_ACCUM* __restrict dscale,
                             FLOAT_ACCUM* __restrict dbias,
                             const FLOAT_ACCUM* __restrict savedMean,
                             const FLOAT_ACCUM* __restrict savedInvVariance,
                             FLOAT_ACCUM INHW,
                             FLOAT_ACCUM actAlpha,
                             FLOAT_ACCUM actBeta)
{
    FLOAT_ACCUM mean        = 0;
    FLOAT_ACCUM invVariance = 0;
    FLOAT_ACCUM pscale      = 0;
    FLOAT_ACCUM pbias       = 0;
    FLOAT_ACCUM ds          = 0;
    FLOAT_ACCUM db          = 0;

    FLOAT_ACCUM batchvalues[nloop];
    FLOAT_ACCUM dyvalues[nloop];

    __shared__ FLOAT_ACCUM lbns;
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    __shared__ FLOAT_ACCUM lbnb;
#endif
    __shared__ FLOAT_ACCUM lmean, lvar;

    unsigned int lid    = threadIdx.x;
    unsigned int grpid  = blockIdx.x;
    unsigned int chwid  = grpid * HIP_PLUGIN_BN_HW + (lid % HIP_PLUGIN_BN_HW);
    unsigned int lidihw = lid / HIP_PLUGIN_BN_HW;

    if(lid == 0)
    {
        lbns = bnScale[grpid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
        lbnb = bnBias[grpid];
#endif
        lmean = savedMean[grpid];
        lvar  = savedInvVariance[grpid];
    }
    __syncthreads();
    mean        = lmean;
    invVariance = lvar;
    pscale      = lbns;
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    pbias = lbnb;
#endif

    if(lid < segment)
    {
        for(unsigned int loop_n = 0; loop_n < nloopm; ++loop_n)
        {
            unsigned int nid   = loop_n * segihw + lidihw;
            unsigned int index = nid * HIP_PLUGIN_BN_CHW + chwid;
            dyvalues[loop_n]   = CVT_FLOAT2ACCUM(dy_in[index]);
            batchvalues[loop_n] = (CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance;

            dyvalues[loop_n] = hip_kernel_plugin::batchnorm::applyActivationGradient<
                FLOAT_ACCUM,
                hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
                dyvalues[loop_n], batchvalues[loop_n], pscale, pbias, actAlpha, actBeta);

            db += dyvalues[loop_n];
            ds += batchvalues[loop_n] * dyvalues[loop_n];
        }
        unsigned int nid    = snhw + lidihw;
        unsigned int index  = nid * HIP_PLUGIN_BN_CHW + chwid;
        dyvalues[nloopm]    = (index < HIP_PLUGIN_BN_NCHW) ? CVT_FLOAT2ACCUM(dy_in[index]) : 0;
        batchvalues[nloopm] = (index < HIP_PLUGIN_BN_NCHW)
                                  ? ((CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance)
                                  : 0;

        dyvalues[nloopm] = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dyvalues[nloopm], batchvalues[nloopm], pscale, pbias, actAlpha, actBeta);

        db += dyvalues[nloopm];
        ds += batchvalues[nloopm] * dyvalues[nloopm];
    }

    __syncthreads();

    __shared__ FLOAT_ACCUM lcl_ds[ldsSize];
    __shared__ FLOAT_ACCUM lcl_db[ldsSize];
    hip_kernel_plugin::reduction::lds_reduce2<FLOAT_ACCUM, ldsSize>(
        ds, db, FLOAT_ACCUM(1.0), lcl_ds, lcl_db, lid);

    if(lid < segment)
    {
        for(unsigned int loop_n = 0; loop_n < nloopm; loop_n++)
        {
            unsigned int nid   = loop_n * segihw + lidihw;
            unsigned int index = nid * HIP_PLUGIN_BN_CHW + chwid;
            FLOAT_ACCUM tmp1 = static_cast<FLOAT_ACCUM>(HIP_PLUGIN_BN_NHW) * dyvalues[loop_n] - db;
            FLOAT_ACCUM tmp2 = -batchvalues[loop_n] * ds;
            FLOAT_ACCUM tmp3 = pscale * invVariance * INHW;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
        }
        unsigned int nid   = snhw + lidihw;
        unsigned int index = nid * HIP_PLUGIN_BN_CHW + chwid;
        if(index < HIP_PLUGIN_BN_NCHW)
        {
            FLOAT_ACCUM tmp1 = static_cast<FLOAT_ACCUM>(HIP_PLUGIN_BN_NHW) * dyvalues[nloopm] - db;
            FLOAT_ACCUM tmp2 = -batchvalues[nloopm] * ds;
            FLOAT_ACCUM tmp3 = pscale * invVariance * INHW;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }
}

// ============================================================================
// Variant 1: Threads loop over NHW with vectorized reads.
//            Supports both NCHW and NHWC.
// ============================================================================
#elif(HIP_PLUGIN_BN_VARIANT == 1)

constexpr unsigned int readSize  = HIP_PLUGIN_LAYOUT_NHWC ? 1 : 4;
constexpr unsigned int writeSize = HIP_PLUGIN_LAYOUT_NHWC ? 1 : 2;

using fp_read_vec_type      = typename hip_kernel_plugin::mapped_vector_type<FLOAT, readSize>::type;
using fp_prec_read_vec_type = typename hip_kernel_plugin::mapped_vector_type<FLOAT_ACCUM, readSize>::type;
using fp_write_vec_type     = typename hip_kernel_plugin::mapped_vector_type<FLOAT, writeSize>::type;
using fp_prec_write_vec_type =
    typename hip_kernel_plugin::mapped_vector_type<FLOAT_ACCUM, writeSize>::type;

constexpr unsigned int grprd   = blockSize * 1 * readSize;
constexpr unsigned int rem4    = HIP_PLUGIN_BN_NHW - (HIP_PLUGIN_BN_NHW / grprd) * grprd;
constexpr unsigned int less4   = HIP_PLUGIN_BN_NHW - rem4;
constexpr unsigned int rem     = HIP_PLUGIN_BN_NHW - (HIP_PLUGIN_BN_NHW / blockSize) * blockSize;
constexpr unsigned int less    = HIP_PLUGIN_BN_NHW - rem;
constexpr unsigned int chunk   = writeSize * blockSize;
constexpr unsigned int remout  = HIP_PLUGIN_BN_NHW - ((HIP_PLUGIN_BN_NHW / chunk) * chunk);
constexpr unsigned int lessout = HIP_PLUGIN_BN_NHW - remout;

__device__ __forceinline__ unsigned int getTensorIndex_v1(unsigned int loopIndex)
{
    unsigned int grpid = blockIdx.x;
    unsigned int chwid = grpid * HIP_PLUGIN_BN_HW;
    unsigned int nidx  = loopIndex / HIP_PLUGIN_BN_HW;
    unsigned int hwidx = loopIndex - (nidx * HIP_PLUGIN_BN_HW);
    if constexpr(HIP_PLUGIN_LAYOUT_NHWC)
    {
        return nidx * HIP_PLUGIN_BN_CHW + hwidx * HIP_PLUGIN_BN_C + grpid;
    }
    else
    {
        return nidx * HIP_PLUGIN_BN_CHW + chwid + hwidx;
    }
}

template <typename FpPrecVecT>
__device__ __forceinline__ FpPrecVecT
vectorizedBwdActivation(FpPrecVecT const& dy,
                        FpPrecVecT const& xnorm,
                        FLOAT_ACCUM scale,
                        FLOAT_ACCUM bias,
                        FLOAT_ACCUM alpha,
                        FLOAT_ACCUM beta)
{
    constexpr auto vecSize = hip_kernel_plugin::mapped_vector_info<FpPrecVecT>::size;
    if constexpr(vecSize == 4)
    {
        FpPrecVecT out;
        out.x = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        out.z = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.z, xnorm.z, scale, bias, alpha, beta);
        out.w = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.w, xnorm.w, scale, bias, alpha, beta);
        return out;
    }
    else if constexpr(vecSize == 2)
    {
        FpPrecVecT out;
        out.x = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.x, xnorm.x, scale, bias, alpha, beta);
        out.y = hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy.y, xnorm.y, scale, bias, alpha, beta);
        return out;
    }
    else
    {
        return hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy, xnorm, scale, bias, alpha, beta);
    }
}

template <typename FpPrecVecT>
__device__ __forceinline__ void accumulateVec(FLOAT_ACCUM& acc, FpPrecVecT const& val)
{
    constexpr auto vecSize = hip_kernel_plugin::mapped_vector_info<FpPrecVecT>::size;
    if constexpr(vecSize == 4)
    {
        acc += val.x;
        acc += val.y;
        acc += val.z;
        acc += val.w;
    }
    else if constexpr(vecSize == 2)
    {
        acc += val.x;
        acc += val.y;
    }
    else
    {
        acc += val;
    }
}

template <typename FpPrecVecT>
__device__ __forceinline__ void accumulateMadVec(FLOAT_ACCUM& acc,
                                                 FpPrecVecT const& a,
                                                 FpPrecVecT const& b)
{
    constexpr auto vecSize = hip_kernel_plugin::mapped_vector_info<FpPrecVecT>::size;
    if constexpr(vecSize == 4)
    {
        acc = fma(a.x, b.x, acc);
        acc = fma(a.y, b.y, acc);
        acc = fma(a.z, b.z, acc);
        acc = fma(a.w, b.w, acc);
    }
    else if constexpr(vecSize == 2)
    {
        acc = fma(a.x, b.x, acc);
        acc = fma(a.y, b.y, acc);
    }
    else
    {
        acc = fma(a, b, acc);
    }
}

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormBwdSpatialSaved(const FLOAT* __restrict x_in,
                             const FLOAT* __restrict dy_in,
                             FLOAT* __restrict dx_out,
                             const FLOAT_ACCUM* __restrict bnScale,
                             const FLOAT_ACCUM* __restrict bnBias,
                             FLOAT_ACCUM* __restrict dscale,
                             FLOAT_ACCUM* __restrict dbias,
                             const FLOAT_ACCUM* __restrict savedMean,
                             const FLOAT_ACCUM* __restrict savedInvVariance,
                             FLOAT_ACCUM INHW,
                             FLOAT_ACCUM actAlpha,
                             FLOAT_ACCUM actBeta)
{
    FLOAT_ACCUM mean        = 0;
    FLOAT_ACCUM invVariance = 0;
    FLOAT_ACCUM pscale      = 0;
    FLOAT_ACCUM pbias       = 0;
    FLOAT_ACCUM db          = 0;
    FLOAT_ACCUM ds          = 0;

    unsigned int lid   = threadIdx.x;
    unsigned int grpid = blockIdx.x;

    pscale = bnScale[grpid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    pbias = bnBias[grpid];
#endif
    mean        = savedMean[grpid];
    invVariance = savedInvVariance[grpid];

    constexpr unsigned int readUnrollHint =
        HIP_PLUGIN_BN_N > 768 ? 4 : 2;
    hip_kernel_plugin::static_unroll_count<
        unsigned int, 0, less4, grprd, readUnrollHint>{[&](unsigned int k) {
        unsigned int l = k + (lid << 2 * (1 - HIP_PLUGIN_LAYOUT_NHWC));
        if(l < less4)
        {
            unsigned int index     = getTensorIndex_v1(l);
            fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(x_in + index));
            fp_read_vec_type dyRead =
                *(reinterpret_cast<const fp_read_vec_type*>(dy_in + index));
            fp_prec_read_vec_type dyvalue = hip_kernel_plugin::cast<fp_prec_read_vec_type>(dyRead);
            fp_prec_read_vec_type xhat =
                (hip_kernel_plugin::cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

            dyvalue = vectorizedBwdActivation(dyvalue, xhat, pscale, pbias, actAlpha, actBeta);

            accumulateVec(db, dyvalue);
            accumulateMadVec(ds, xhat, dyvalue);
        }
    }};

    if constexpr(rem4 > 0)
    {
        unsigned int index = getTensorIndex_v1((lid << 2 * (1 - HIP_PLUGIN_LAYOUT_NHWC)) + less4);
        if(index + readSize - 1 < HIP_PLUGIN_BN_NCHW)
        {
            fp_read_vec_type xread = *(reinterpret_cast<const fp_read_vec_type*>(x_in + index));
            fp_read_vec_type dyRead =
                *(reinterpret_cast<const fp_read_vec_type*>(dy_in + index));
            fp_prec_read_vec_type dyvalue = hip_kernel_plugin::cast<fp_prec_read_vec_type>(dyRead);
            fp_prec_read_vec_type xhat =
                (hip_kernel_plugin::cast<fp_prec_read_vec_type>(xread) - mean) * invVariance;

            dyvalue = vectorizedBwdActivation(dyvalue, xhat, pscale, pbias, actAlpha, actBeta);

            accumulateVec(db, dyvalue);
            accumulateMadVec(ds, xhat, dyvalue);
        }
    }

    __syncthreads();

    __shared__ FLOAT_ACCUM lcl_ds[ldsSize];
    __shared__ FLOAT_ACCUM lcl_db[ldsSize];
    hip_kernel_plugin::reduction::lds_reduce2<FLOAT_ACCUM, ldsSize>(
        ds, db, FLOAT_ACCUM(1.0), lcl_ds, lcl_db, lid);

    __syncthreads();

    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }

    constexpr unsigned int writeUnrollHint =
        HIP_PLUGIN_BN_N > 768 ? 2 : 1;
    hip_kernel_plugin::static_unroll_count<
        unsigned int, 0, lessout, chunk, writeUnrollHint>{[&](unsigned int k) {
        fp_prec_write_vec_type vals;
        unsigned int l     = k + (writeSize * lid);
        unsigned int index = getTensorIndex_v1(l);
        if(l < lessout)
        {
            fp_write_vec_type xread =
                *(reinterpret_cast<const fp_write_vec_type*>(x_in + index));
            fp_write_vec_type dyRead =
                *(reinterpret_cast<const fp_write_vec_type*>(dy_in + index));
            fp_prec_write_vec_type value1 = hip_kernel_plugin::cast<fp_prec_write_vec_type>(dyRead);
            fp_prec_write_vec_type xhat1 =
                (hip_kernel_plugin::cast<fp_prec_write_vec_type>(xread) - mean) * invVariance;

            value1 = vectorizedBwdActivation(value1, xhat1, pscale, pbias, actAlpha, actBeta);

            fp_prec_write_vec_type tmp1 =
                hip_kernel_plugin::cast<fp_prec_write_vec_type>(static_cast<FLOAT_ACCUM>(HIP_PLUGIN_BN_NHW)) * value1
                + hip_kernel_plugin::cast<fp_prec_write_vec_type>(-db);
            fp_prec_write_vec_type tmp2 =
                -xhat1 * hip_kernel_plugin::cast<fp_prec_write_vec_type>(ds);
            FLOAT_ACCUM tmp3 = pscale * invVariance * INHW;
            vals = hip_kernel_plugin::cast<fp_prec_write_vec_type>(tmp3) * (tmp2 + tmp1);
        }

        __syncthreads();

        if(l < lessout)
        {
            *reinterpret_cast<fp_write_vec_type*>(dx_out + index) =
                hip_kernel_plugin::cast<fp_write_vec_type>(vals);
        }
    }};

    if constexpr(remout > 0)
    {
        unsigned int remkeyout = (writeSize * lid) + lessout;
        for(unsigned int j = 0; j < writeSize; j++)
        {
            unsigned int index = getTensorIndex_v1(remkeyout + j);
            if(index < HIP_PLUGIN_BN_NCHW)
            {
                FLOAT_ACCUM value1 = CVT_FLOAT2ACCUM(dy_in[index]);
                FLOAT_ACCUM xhat   = (CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance;

                value1 = hip_kernel_plugin::batchnorm::applyActivationGradient<
                    FLOAT_ACCUM,
                    hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
                    value1, xhat, pscale, pbias, actAlpha, actBeta);

                FLOAT_ACCUM tmp1 = static_cast<FLOAT_ACCUM>(HIP_PLUGIN_BN_NHW) * value1 - db;
                FLOAT_ACCUM tmp2 = -xhat * ds;
                FLOAT_ACCUM tmp3 = pscale * invVariance * INHW;
                dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
        }
    }
}

// ============================================================================
// Variant 3: Threads map to HW positions, loop over N.
//            NCHW only, good for moderate HW with small batch sizes.
// ============================================================================
#elif(HIP_PLUGIN_BN_VARIANT == 3)

extern "C" __global__ void __launch_bounds__(blockSize)
    BatchNormBwdSpatialSaved(const FLOAT* __restrict x_in,
                             const FLOAT* __restrict dy_in,
                             FLOAT* __restrict dx_out,
                             const FLOAT_ACCUM* __restrict bnScale,
                             const FLOAT_ACCUM* __restrict bnBias,
                             FLOAT_ACCUM* __restrict dscale,
                             FLOAT_ACCUM* __restrict dbias,
                             const FLOAT_ACCUM* __restrict savedMean,
                             const FLOAT_ACCUM* __restrict savedInvVariance,
                             FLOAT_ACCUM INHW,
                             FLOAT_ACCUM actAlpha,
                             FLOAT_ACCUM actBeta)
{
    FLOAT_ACCUM mean        = 0;
    FLOAT_ACCUM invVariance = 0;
    FLOAT_ACCUM pscale      = 0;
    FLOAT_ACCUM pbias       = 0;
    FLOAT_ACCUM ds          = 0;
    FLOAT_ACCUM db          = 0;

    FLOAT_ACCUM batchvalues[HIP_PLUGIN_BN_N];
    FLOAT_ACCUM dyvalues[HIP_PLUGIN_BN_N];

    unsigned int lid   = threadIdx.x;
    unsigned int grpid = blockIdx.x;
    unsigned int cidx  = grpid * HIP_PLUGIN_BN_HW;

    pscale = bnScale[grpid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    pbias = bnBias[grpid];
#endif
    mean        = savedMean[grpid];
    invVariance = savedInvVariance[grpid];

    if(lid < HIP_PLUGIN_BN_HW)
    {
        for(unsigned int batch_n = 0; batch_n < HIP_PLUGIN_BN_N; batch_n++)
        {
            unsigned int index = batch_n * HIP_PLUGIN_BN_CHW + cidx + lid;
            if constexpr(HIP_PLUGIN_BN_N < HIP_PLUGIN_BN_MAXN)
            {
                dyvalues[batch_n]    = CVT_FLOAT2ACCUM(dy_in[index]);
                batchvalues[batch_n] = (CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance;

                dyvalues[batch_n] = hip_kernel_plugin::batchnorm::applyActivationGradient<
                    FLOAT_ACCUM,
                    hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
                    dyvalues[batch_n], batchvalues[batch_n], pscale, pbias, actAlpha, actBeta);

                db += dyvalues[batch_n];
                ds = fma(batchvalues[batch_n], dyvalues[batch_n], ds);
            }
            else
            {
                FLOAT_ACCUM dyvalue = CVT_FLOAT2ACCUM(dy_in[index]);
                FLOAT_ACCUM xhat    = (CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance;

                dyvalue = hip_kernel_plugin::batchnorm::applyActivationGradient<
                    FLOAT_ACCUM,
                    hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
                    dyvalue, xhat, pscale, pbias, actAlpha, actBeta);

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

    __shared__ FLOAT_ACCUM lcl_ds[ldsSize];
    __shared__ FLOAT_ACCUM lcl_db[ldsSize];
    hip_kernel_plugin::reduction::lds_reduce2<FLOAT_ACCUM, ldsSize>(
        ds, db, FLOAT_ACCUM(1.0), lcl_ds, lcl_db, lid);

    __syncthreads();

    if(lid < HIP_PLUGIN_BN_HW)
    {
        for(unsigned int batch_n = 0; batch_n < HIP_PLUGIN_BN_N; batch_n++)
        {
            unsigned int index = batch_n * HIP_PLUGIN_BN_CHW + cidx + lid;
            FLOAT_ACCUM dyvalue;
            FLOAT_ACCUM xhat;
            if constexpr(HIP_PLUGIN_BN_N < HIP_PLUGIN_BN_MAXN)
            {
                dyvalue = dyvalues[batch_n];
                xhat    = batchvalues[batch_n];
            }
            else
            {
                dyvalue = CVT_FLOAT2ACCUM(dy_in[index]);
                xhat    = (CVT_FLOAT2ACCUM(x_in[index]) - mean) * invVariance;

                dyvalue = hip_kernel_plugin::batchnorm::applyActivationGradient<
                    FLOAT_ACCUM,
                    hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
                    dyvalue, xhat, pscale, pbias, actAlpha, actBeta);
            }

            FLOAT_ACCUM tmp1 = static_cast<FLOAT_ACCUM>(HIP_PLUGIN_BN_NHW) * dyvalue - db;
            FLOAT_ACCUM tmp2 = -xhat * ds;
            FLOAT_ACCUM tmp3 = pscale * invVariance * INHW;
            dx_out[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
        }
    }
    if(lid == 0)
    {
        dbias[grpid]  = db;
        dscale[grpid] = ds;
    }
}

#endif
