// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Variant 2 "spatial multiple" backward batch normalization.
// Uses saved mean/invVariance (USESAVED=1 path only).
// 3-kernel pipeline: DScaleDBias -> FinalDScaleDBias -> DX
// dx_out is reused as stash workspace between kernels.

#include "BatchnormActivation.hpp"
#include "BatchnormStash.hpp"
#include "FloatTypes.h"
#include "ReductionFunctions.hpp"
#include "VectorTypes.hpp"

constexpr unsigned int xlocalsize = HIP_PLUGIN_BN_GRP0;
constexpr unsigned int ylocalsize = HIP_PLUGIN_BN_GRP1;
constexpr unsigned int zlocalsize = HIP_PLUGIN_BN_GRP2;
constexpr unsigned int blockTotal = xlocalsize * ylocalsize * zlocalsize;
constexpr unsigned int ldsSize    = HIP_PLUGIN_BN_LDS_SIZE;

constexpr unsigned int xstride = HIP_PLUGIN_LAYOUT_NHWC ? 1 : HIP_PLUGIN_BN_HW;
constexpr unsigned int ystride = HIP_PLUGIN_LAYOUT_NHWC ? HIP_PLUGIN_BN_C : 1;
constexpr unsigned int vecSizeX = HIP_PLUGIN_BN_VEC_SIZE;
constexpr unsigned int vecSizeY = 1;

using fp_ls_type      = typename hip_kernel_plugin::mapped_vector_type<FLOAT, vecSizeX>::type;
using fp_prec_ls_type = typename hip_kernel_plugin::mapped_vector_type<FLOAT_ACCUM, vecSizeX>::type;
using fp_prec_c_type  = FLOAT_ACCUM;
using fp_c_type       = FLOAT;

__device__ __forceinline__ void accumulateVal(fp_prec_c_type& a, fp_prec_ls_type const& b)
{
    if constexpr(vecSizeX == 1)
    {
        a += static_cast<fp_prec_c_type>(b);
    }
    else if constexpr(vecSizeX == 2)
    {
        a += static_cast<fp_prec_c_type>(b.x);
        a += static_cast<fp_prec_c_type>(b.y);
    }
    else if constexpr(vecSizeX == 4)
    {
        a += static_cast<fp_prec_c_type>(b.x);
        a += static_cast<fp_prec_c_type>(b.y);
        a += static_cast<fp_prec_c_type>(b.z);
        a += static_cast<fp_prec_c_type>(b.w);
    }
    else if constexpr(vecSizeX == 8)
    {
        a += static_cast<fp_prec_c_type>(b.s0);
        a += static_cast<fp_prec_c_type>(b.s1);
        a += static_cast<fp_prec_c_type>(b.s2);
        a += static_cast<fp_prec_c_type>(b.s3);
        a += static_cast<fp_prec_c_type>(b.s4);
        a += static_cast<fp_prec_c_type>(b.s5);
        a += static_cast<fp_prec_c_type>(b.s6);
        a += static_cast<fp_prec_c_type>(b.s7);
    }
}

__device__ __forceinline__ void accumulateMadVal(fp_prec_c_type& a,
                                                 fp_prec_ls_type const& b,
                                                 fp_prec_ls_type const& c)
{
    if constexpr(vecSizeX == 1)
    {
        a = fma(static_cast<fp_prec_c_type>(b), static_cast<fp_prec_c_type>(c), a);
    }
    else if constexpr(vecSizeX == 2)
    {
        a = fma(static_cast<fp_prec_c_type>(b.x), static_cast<fp_prec_c_type>(c.x), a);
        a = fma(static_cast<fp_prec_c_type>(b.y), static_cast<fp_prec_c_type>(c.y), a);
    }
    else if constexpr(vecSizeX == 4)
    {
        a = fma(static_cast<fp_prec_c_type>(b.x), static_cast<fp_prec_c_type>(c.x), a);
        a = fma(static_cast<fp_prec_c_type>(b.y), static_cast<fp_prec_c_type>(c.y), a);
        a = fma(static_cast<fp_prec_c_type>(b.z), static_cast<fp_prec_c_type>(c.z), a);
        a = fma(static_cast<fp_prec_c_type>(b.w), static_cast<fp_prec_c_type>(c.w), a);
    }
    else if constexpr(vecSizeX == 8)
    {
        a = fma(static_cast<fp_prec_c_type>(b.s0), static_cast<fp_prec_c_type>(c.s0), a);
        a = fma(static_cast<fp_prec_c_type>(b.s1), static_cast<fp_prec_c_type>(c.s1), a);
        a = fma(static_cast<fp_prec_c_type>(b.s2), static_cast<fp_prec_c_type>(c.s2), a);
        a = fma(static_cast<fp_prec_c_type>(b.s3), static_cast<fp_prec_c_type>(c.s3), a);
        a = fma(static_cast<fp_prec_c_type>(b.s4), static_cast<fp_prec_c_type>(c.s4), a);
        a = fma(static_cast<fp_prec_c_type>(b.s5), static_cast<fp_prec_c_type>(c.s5), a);
        a = fma(static_cast<fp_prec_c_type>(b.s6), static_cast<fp_prec_c_type>(c.s6), a);
        a = fma(static_cast<fp_prec_c_type>(b.s7), static_cast<fp_prec_c_type>(c.s7), a);
    }
}

template <typename T, typename U>
__device__ __forceinline__ T toPrecLsType(U val)
{
    return hip_kernel_plugin::cast<T>(val);
}

template <typename T, typename U>
__device__ __forceinline__ T toLsType(U val)
{
    return hip_kernel_plugin::cast<T>(val);
}

template <typename FpPrecVecType>
__device__ __forceinline__ auto batchBwdNormalization(const FpPrecVecType value,
                                                      const FpPrecVecType xhat,
                                                      const FLOAT_ACCUM db,
                                                      const FLOAT_ACCUM ds,
                                                      const FLOAT_ACCUM pscale,
                                                      const FLOAT_ACCUM invVar,
                                                      const unsigned int nhw,
                                                      const FLOAT_ACCUM inhw)
{
    FpPrecVecType tmp1 = hip_kernel_plugin::cast<FpPrecVecType>(
                             static_cast<FLOAT_ACCUM>(nhw)) *
                             value +
                         hip_kernel_plugin::cast<FpPrecVecType>(-db);
    FpPrecVecType tmp2 = -xhat * hip_kernel_plugin::cast<FpPrecVecType>(ds);
    FLOAT_ACCUM tmp3   = pscale * invVar * inhw;
    return hip_kernel_plugin::cast<FpPrecVecType>(tmp3) * (tmp2 + tmp1);
}

template <typename FpPrecVecType>
__device__ __forceinline__ FpPrecVecType bwdActivationVec(FpPrecVecType const& dy,
                                                          FpPrecVecType const& xnorm,
                                                          FLOAT_ACCUM scale,
                                                          FLOAT_ACCUM bias,
                                                          FLOAT_ACCUM alpha,
                                                          FLOAT_ACCUM beta)
{
    if constexpr(vecSizeX == 1)
    {
        return hip_kernel_plugin::batchnorm::applyActivationGradient<
            FLOAT_ACCUM,
            hip_kernel_plugin::batchnorm::ActivationMode{HIP_PLUGIN_NRN_OP_ID}>(
            dy, xnorm, scale, bias, alpha, beta);
    }
    else if constexpr(vecSizeX == 2)
    {
        FpPrecVecType out;
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
    else if constexpr(vecSizeX == 4)
    {
        FpPrecVecType out;
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
    else
    {
        return dy;
    }
}

// ============================================================================
// Kernel 1: DScaleDBias
// Computes partial dscale/dbias per workgroup, stashes to dx_out buffer.
// ============================================================================
extern "C" __global__ void __launch_bounds__(blockTotal)
    BatchNormBwdSpatialDScaleDBias(const FLOAT* __restrict x_in,
                                   const FLOAT* __restrict dy_in,
                                   FLOAT* __restrict buff,
                                   const FLOAT_ACCUM* __restrict bnScale,
                                   const FLOAT_ACCUM* __restrict bnBias,
                                   const FLOAT_ACCUM* __restrict savedMean,
                                   const FLOAT_ACCUM* __restrict savedInvVariance,
                                   FLOAT_ACCUM actAlpha,
                                   FLOAT_ACCUM actBeta)
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

    if(xgid * vecSizeX >= HIP_PLUGIN_BN_C)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type dscale_partial = static_cast<fp_prec_c_type>(0);
    fp_prec_c_type dbias_partial  = static_cast<fp_prec_c_type>(0);
    fp_prec_c_type pscale         = static_cast<fp_prec_c_type>(0);
    fp_prec_c_type pbias          = static_cast<fp_prec_c_type>(0);

    __shared__ fp_prec_c_type lmean[xlocalsize];
    __shared__ fp_prec_c_type livar[xlocalsize];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lcl_scale[xlocalsize];
    __shared__ fp_prec_c_type lcl_bias[xlocalsize];
#endif

    if(ylid == 0 && zlid == 0)
    {
        lmean[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid] = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
        lcl_scale[xlid] = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
        lcl_bias[xlid]  = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
    }

    __syncthreads();

    constexpr unsigned int chw = HIP_PLUGIN_BN_C * HIP_PLUGIN_BN_HW / vecSizeX;

    if(ygid * vecSizeY < HIP_PLUGIN_BN_HW && zgid < HIP_PLUGIN_BN_N)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
        pscale = lcl_scale[xlid];
        pbias  = lcl_bias[xlid];
#endif

        unsigned int index_base = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * chw +
                                  ygid * (ystride / vecSizeX) * vecSizeY +
                                  xgid * (xstride);
        for(unsigned int nn = 0; nn < HIP_PLUGIN_BN_N_ELEMENTS; nn++)
        {
            unsigned int index = index_base + nn * chw;
            fp_prec_ls_type value1 =
                hip_kernel_plugin::cast<fp_prec_ls_type>(
                    *reinterpret_cast<const fp_ls_type*>(dy_in + index));
            fp_prec_ls_type value2 =
                hip_kernel_plugin::cast<fp_prec_ls_type>(
                    *reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat = (value2 - mean) * invVar;

            value1 = bwdActivationVec(
                value1, xhat, pscale, pbias, actAlpha, actBeta);

            accumulateVal(dbias_partial, value1);
            accumulateMadVal(dscale_partial, xhat, value1);
        }
    }

    __shared__ fp_prec_c_type lcl_data[2 * ldsSize];
    hip_kernel_plugin::reduction::lds_reduce2_2d<fp_prec_c_type, FLOAT_ACCUM, 2 * ldsSize>(
        dscale_partial,
        dbias_partial,
        static_cast<FLOAT_ACCUM>(1.0),
        lcl_data,
        xgrp_sz,
        xlid,
        ylid + zlid * ygrp_sz,
        ygrp_sz * zgrp_sz);

    if(ylid == 0 && zlid == 0)
    {
        constexpr unsigned int stash_index = 0;
        hip_kernel_plugin::batchnorm::storeToStash<fp_prec_c_type>(
            dscale_partial,
            reinterpret_cast<fp_c_type*>(buff),
            stash_index,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * vecSizeY,
            ystride / vecSizeX,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
        hip_kernel_plugin::batchnorm::storeToStash<fp_prec_c_type>(
            dbias_partial,
            reinterpret_cast<fp_c_type*>(buff),
            stash_index + 1,
            zgrp_sz * zgrp_id * HIP_PLUGIN_BN_N_ELEMENTS,
            ygrp_sz * ygrp_id * vecSizeY,
            ystride / vecSizeX,
            xgrp_sz,
            xgrp_id,
            xlid,
            xstride);
    }
}

// ============================================================================
// Kernel 2: FinalDScaleDBias
// Reduces partial dscale/dbias from stash into final dscale/dbias outputs.
// ============================================================================
extern "C" __global__ void __launch_bounds__(HIP_PLUGIN_BN_GRP0_FINAL* HIP_PLUGIN_BN_GRP1_FINAL*
                                             HIP_PLUGIN_BN_GRP2_FINAL)
    BatchNormBwdSpatialFinalDScaleDBias(const FLOAT* __restrict buff,
                                        FLOAT_ACCUM* __restrict delta_scale,
                                        FLOAT_ACCUM* __restrict delta_bias)
{
    unsigned int xlid    = threadIdx.x;
    unsigned int ylid    = threadIdx.y;
    unsigned int zlid    = threadIdx.z;
    unsigned int xgrp_id = blockIdx.x;
    unsigned int xgid    = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int xgrp_sz = blockDim.x;
    unsigned int ygrp_sz = blockDim.y;
    unsigned int zgrp_sz = blockDim.z;

    constexpr unsigned int stash_index = 0;

    if(xgid * vecSizeX >= HIP_PLUGIN_BN_C)
    {
        return;
    }

    fp_prec_c_type dscale_acc = static_cast<fp_prec_c_type>(0);
    fp_prec_c_type dbias_acc  = static_cast<fp_prec_c_type>(0);

    for(unsigned int zoffset = zlid; zoffset < HIP_PLUGIN_BN_NGRPS2; zoffset += zgrp_sz)
    {
        for(unsigned int yoffset = ylid; yoffset < HIP_PLUGIN_BN_NGRPS; yoffset += ygrp_sz)
        {
            dscale_acc += hip_kernel_plugin::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * vecSizeY,
                ystride / vecSizeX,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
            dbias_acc += hip_kernel_plugin::batchnorm::loadFromStash<fp_prec_c_type>(
                reinterpret_cast<const fp_c_type*>(buff),
                stash_index + 1,
                HIP_PLUGIN_BN_GRP2 * zoffset * HIP_PLUGIN_BN_N_ELEMENTS,
                HIP_PLUGIN_BN_GRP1 * yoffset * vecSizeY,
                ystride / vecSizeX,
                xgrp_sz,
                xgrp_id,
                xlid,
                xstride);
        }
    }

    constexpr unsigned int finalLdsSize =
        2 * HIP_PLUGIN_BN_GRP0_FINAL * HIP_PLUGIN_BN_GRP1_FINAL * HIP_PLUGIN_BN_GRP2_FINAL;
    __shared__ fp_prec_c_type lcl_data[finalLdsSize];
    hip_kernel_plugin::reduction::lds_reduce2_2d<fp_prec_c_type, FLOAT_ACCUM, finalLdsSize>(
        dscale_acc,
        dbias_acc,
        static_cast<FLOAT_ACCUM>(1.0),
        lcl_data,
        xgrp_sz,
        xlid,
        ylid + zlid * ygrp_sz,
        ygrp_sz * zgrp_sz);

    if(ylid == 0 && zlid == 0)
    {
        reinterpret_cast<fp_prec_c_type*>(delta_scale)[xgid] = dscale_acc;
        reinterpret_cast<fp_prec_c_type*>(delta_bias)[xgid]  = dbias_acc;
    }
}

// ============================================================================
// Kernel 3: DX
// Computes final dx using dscale, dbias, savedMean, savedInvVariance.
// ============================================================================
extern "C" __global__ void __launch_bounds__(blockTotal)
    BatchNormBwdSpatialDX(const FLOAT* __restrict x_in,
                          const FLOAT* __restrict dy_in,
                          FLOAT* __restrict dx_out,
                          const FLOAT_ACCUM* __restrict bnScale,
                          const FLOAT_ACCUM* __restrict bnBias,
                          const FLOAT_ACCUM* __restrict delta_scale,
                          const FLOAT_ACCUM* __restrict delta_bias,
                          const FLOAT_ACCUM* __restrict savedMean,
                          const FLOAT_ACCUM* __restrict savedInvVariance,
                          FLOAT_ACCUM INHW,
                          FLOAT_ACCUM actAlpha,
                          FLOAT_ACCUM actBeta)
{
    unsigned int xlid = threadIdx.x;
    unsigned int ylid = threadIdx.y;
    unsigned int zlid = threadIdx.z;
    unsigned int xgid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int ygid = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int zgid = blockDim.z * blockIdx.z + threadIdx.z;

    if(xgid * vecSizeX >= HIP_PLUGIN_BN_C)
    {
        return;
    }

    fp_prec_c_type mean, invVar;
    fp_prec_c_type pscale, ds, db;
    fp_prec_c_type pbias = static_cast<fp_prec_c_type>(0);

    __shared__ fp_prec_c_type lscale[xlocalsize];
    __shared__ fp_prec_c_type ldscale[xlocalsize];
    __shared__ fp_prec_c_type ldbias[xlocalsize];
    __shared__ fp_prec_c_type lmean[xlocalsize];
    __shared__ fp_prec_c_type livar[xlocalsize];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
    __shared__ fp_prec_c_type lbias[xlocalsize];
#endif

    if(ylid == 0 && zlid == 0)
    {
        lmean[xlid]   = reinterpret_cast<const fp_prec_c_type*>(savedMean)[xgid];
        livar[xlid]   = reinterpret_cast<const fp_prec_c_type*>(savedInvVariance)[xgid];
        lscale[xlid]  = reinterpret_cast<const fp_prec_c_type*>(bnScale)[xgid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
        lbias[xlid]   = reinterpret_cast<const fp_prec_c_type*>(bnBias)[xgid];
#endif
        ldscale[xlid] = reinterpret_cast<const fp_prec_c_type*>(delta_scale)[xgid];
        ldbias[xlid]  = reinterpret_cast<const fp_prec_c_type*>(delta_bias)[xgid];
    }

    __syncthreads();

    constexpr unsigned int chw = HIP_PLUGIN_BN_C * HIP_PLUGIN_BN_HW / vecSizeX;

    if(ygid * vecSizeY < HIP_PLUGIN_BN_HW && zgid < HIP_PLUGIN_BN_N)
    {
        mean   = lmean[xlid];
        invVar = livar[xlid];
        pscale = lscale[xlid];
#if(HIP_PLUGIN_NRN_OP_ID > 0)
        pbias = lbias[xlid];
#endif
        ds = ldscale[xlid];
        db = ldbias[xlid];

        unsigned int index_base = (zgid * HIP_PLUGIN_BN_N_ELEMENTS) * chw +
                                  ygid * (ystride / vecSizeX) * vecSizeY +
                                  xgid * (xstride);
        for(unsigned int nn = 0; nn < HIP_PLUGIN_BN_N_ELEMENTS; nn++)
        {
            unsigned int index    = index_base + nn * chw;
            fp_prec_ls_type x_i   = hip_kernel_plugin::cast<fp_prec_ls_type>(
                *reinterpret_cast<const fp_ls_type*>(x_in + index));
            fp_prec_ls_type xhat  = (x_i - mean) * invVar;
            fp_prec_ls_type value1 = hip_kernel_plugin::cast<fp_prec_ls_type>(
                *reinterpret_cast<const fp_ls_type*>(dy_in + index));

            value1 = bwdActivationVec(
                value1, xhat, pscale, pbias, actAlpha, actBeta);

            *reinterpret_cast<fp_ls_type*>(dx_out + index) =
                hip_kernel_plugin::cast<fp_ls_type>(
                    batchBwdNormalization(
                        value1,
                        xhat,
                        toPrecLsType<fp_prec_ls_type>(db),
                        toPrecLsType<fp_prec_ls_type>(ds),
                        toPrecLsType<fp_prec_ls_type>(pscale),
                        toPrecLsType<fp_prec_ls_type>(invVar),
                        HIP_PLUGIN_BN_NHW,
                        toPrecLsType<fp_prec_ls_type>(INHW)));
        }
    }
}
