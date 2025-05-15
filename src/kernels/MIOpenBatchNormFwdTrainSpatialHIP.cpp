/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"
#include "reduction_functions.hpp"

#ifndef MIO_LAYOUT_NHWC
#define MIO_LAYOUT_NHWC 0
#endif

static_assert(MIO_LAYOUT_NHWC == 0 || MIO_LAYOUT_NHWC == 1, "MIO_LAYOUT_NHWC must be 0 or 1");

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl
{
};

// This is the instance for MIO_BN_VARIANT == 1
template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl<1, FpType, FpPrecType, FpAccumType>
{
    // Configs
    static constexpr int MIO_MAX_READ =
        MIO_LAYOUT_NHWC == 1 ? MIO_MAX_READ : (MIO_BN_HW >= 4096 ? 3 : 2);
    static constexpr int RD_BLK = 1;
    static constexpr int GRPRD =
        MIO_LAYOUT_NHWC == 1 ? (MIO_BN_GRP0 * RD_BLK) : (MIO_BN_GRP0 * RD_BLK * 4);

    static constexpr int MIO_BN_REM4   = (MIO_BN_NHW - ((MIO_BN_NHW / GRPRD) * GRPRD));
    static constexpr int MIO_BN_LESS4  = (MIO_BN_NHW - MIO_BN_REM4);
    static constexpr int MIO_BN_CHUNK4 = (MIO_MAX_READ * GRPRD);
    static constexpr int MIO_BN_REMOUT4 =
        (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK4) * MIO_BN_CHUNK4));
    static constexpr int MIO_BN_LESSOUT4 = (MIO_BN_NHW - MIO_BN_REMOUT4);
    static constexpr int MIO_BN_REM   = (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0));
    static constexpr int MIO_BN_LESS  = (MIO_BN_NHW - MIO_BN_REM);
    static constexpr int MIO_BN_CHUNK = (MIO_MAX_READ * MIO_BN_GRP0);
    static constexpr int MIO_BN_REMOUT =
        (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK) * MIO_BN_CHUNK));
    static constexpr int MIO_BN_LESSOUT = (MIO_BN_NHW - MIO_BN_REMOUT);

    // Kernel
    __forceinline__ __device__ void operator()(const FpType* __restrict in,
                                               FpType* __restrict out,
                                               const FpPrecType* __restrict scale,
                                               const FpPrecType* __restrict bias,
                                               FpPrecType INHW,
                                               double epsilon,
                                               FpPrecType& mean,
                                               FpPrecType& variance,
                                               FpPrecType& invVariance)
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
#if !MIO_LAYOUT_NHWC
        const unsigned int chwid = grpid * MIO_BN_HW;
#endif
        unsigned int nidx  = 0;
        unsigned int hwidx = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!MIO_LAYOUT_NHWC && MIO_BN_HW >= 4096)
        {
            using fp_type4 = typename mapped_vector_type<FpType, 4>::type;
            fp_type4 read4;
#pragma unroll
            for(unsigned int k = lid << 2; k < static_cast<decltype(k)>(MIO_BN_LESS4); k += GRPRD)
            {
                nidx  = k / MIO_BN_HW;
                hwidx = k - (nidx * MIO_BN_HW);
                index = nidx * MIO_BN_CHW + chwid + hwidx;
                read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                mean += fp_to_fpprec<FpPrecType>(read4.x);
                mean += fp_to_fpprec<FpPrecType>(read4.y);
                mean += fp_to_fpprec<FpPrecType>(read4.z);
                mean += fp_to_fpprec<FpPrecType>(read4.w);
                variance = fma(
                    fp_to_fpprec<FpPrecType>(read4.x), fp_to_fpprec<FpPrecType>(read4.x), variance);
                variance = fma(
                    fp_to_fpprec<FpPrecType>(read4.y), fp_to_fpprec<FpPrecType>(read4.y), variance);
                variance = fma(
                    fp_to_fpprec<FpPrecType>(read4.z), fp_to_fpprec<FpPrecType>(read4.z), variance);
                variance = fma(
                    fp_to_fpprec<FpPrecType>(read4.w), fp_to_fpprec<FpPrecType>(read4.w), variance);
            }

            if constexpr(MIO_BN_REM4)
            {
                const unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
                nidx                      = remkey / MIO_BN_HW;
                hwidx                     = remkey - (nidx * MIO_BN_HW);
                index                     = nidx * MIO_BN_CHW + chwid + hwidx;
                // TODO: This is not right, index is unsigned int, MIO_BN_NCHW is int
                // comparing them is not right.
                if(index < (MIO_BN_NCHW - 3))
                {
                    read4 = *(reinterpret_cast<const fp_type4*>(in + index));
                    mean += fp_to_fpprec<FpPrecType>(read4.x);
                    mean += fp_to_fpprec<FpPrecType>(read4.y);
                    mean += fp_to_fpprec<FpPrecType>(read4.z);
                    mean += fp_to_fpprec<FpPrecType>(read4.w);
                    variance = fma(fp_to_fpprec<FpPrecType>(read4.x),
                                   fp_to_fpprec<FpPrecType>(read4.x),
                                   variance);
                    variance = fma(fp_to_fpprec<FpPrecType>(read4.y),
                                   fp_to_fpprec<FpPrecType>(read4.y),
                                   variance);
                    variance = fma(fp_to_fpprec<FpPrecType>(read4.z),
                                   fp_to_fpprec<FpPrecType>(read4.z),
                                   variance);
                    variance = fma(fp_to_fpprec<FpPrecType>(read4.w),
                                   fp_to_fpprec<FpPrecType>(read4.w),
                                   variance);
                }
            }
        }
        else
        {
#pragma unroll
            for(unsigned int k = lid; k < static_cast<decltype(k)>(MIO_BN_LESS); k += MIO_BN_GRP0)
            {
                nidx  = k / MIO_BN_HW;
                hwidx = k - (nidx * MIO_BN_HW);
                if constexpr(MIO_LAYOUT_NHWC)
                {
                    index = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
                }
                else
                {
                    index = nidx * MIO_BN_CHW + chwid + hwidx;
                }
                const auto xin = fp_to_fpprec<FpPrecType>(in[index]);
                mean += xin;
                variance = fma(xin, xin, variance);
            }

            // TODO: not really understand whey MIO_BN_REM is integer instead of boolean value
            if constexpr(MIO_BN_REM)
            {
                // TODO: here I have to cast the lid into int64_t, to avoid compile error
                // could be able to use signed int, if lid is not going exceed the limit
                if(static_cast<int64_t>(lid) < MIO_BN_REM)
                {
                    unsigned int remkey = lid + MIO_BN_LESS;
                    nidx                = remkey / MIO_BN_HW;
                    hwidx               = remkey - (nidx * MIO_BN_HW);
                    if constexpr(MIO_LAYOUT_NHWC)
                    {
                        index = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
                    }
                    else
                    {
                        index = nidx * MIO_BN_CHW + chwid + hwidx;
                    }
                    const auto xin =
                        (index < MIO_BN_NCHW) ? fp_to_fpprec<FpPrecType>(in[index]) : FpPrecType{0};
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto lcl_data_size =
            static_cast<bool>(MIOPEN_USE_AMDGCN) ? MIO_BN_LDSGCN_SIZE : MIO_BN_LDS_SIZE;
        __shared__ FpAccumType lcl_data_x[lcl_data_size];
        __shared__ FpAccumType lcl_data_y[lcl_data_size];

        if constexpr(!MIOPEN_USE_AMDGCN)
        {
            // TODO: I don't know if this is right
            // mean is FpType, could be 16 bit floating point type
            // but we are going to perform lds_reduce2 over FpAccumType&
            // which is hardcoded to float, so &mean which has type FpType&
            // should be case to FpAccumType&, which is unsafe. and the
            // internal data pattern could be inconsistent between these 2
            // types.
            lds_reduce2<FpAccumType, lcl_data_size>(reinterpret_cast<FpAccumType&>(mean),
                                                    reinterpret_cast<FpAccumType&>(variance),
                                                    static_cast<FpAccumType>(INHW),
                                                    lcl_data_x,
                                                    lcl_data_y,
                                                    lid);
        }
        else
        {
            // TODO: this as well as above
            gcn_reduce2<FpAccumType, lcl_data_size>(reinterpret_cast<FpAccumType&>(mean),
                                                    reinterpret_cast<FpAccumType&>(variance),
                                                    static_cast<FpAccumType>(INHW),
                                                    lcl_data_x,
                                                    lcl_data_y,
                                                    lid);
        }

        // REDUCTION COMPLETE ---------------------------
        // TODO: it seems that fma doesn't directly supports half, it supports _Float16
        // here I don't know if I should run mad for FpPrecType or FpType, if I only
        // need to run this for FpPrecType, then we don't need to have this hip_mad struct.
        variance = hip_mad<FpPrecType>{}(-mean, mean, variance);
        if(variance < FpPrecType{0})
        {
            variance = FpPrecType{0};
        }
        // TODO: I don't know if this is correct, the input epsilon is double, but it should be
        // casted to FpType here
        invVariance = rsqrt(variance + static_cast<FpPrecType>(epsilon));
        pvscale     = lcl_scale;
        pvbias      = lcl_bias;

        if constexpr(MIO_LAYOUT_NHWC || MIO_BN_REM == 0)
        {
            constexpr unsigned int k_limit =
                static_cast<bool>(MIO_LAYOUT_NHWC) ? MIO_BN_NHW : MIO_BN_LESS;
#pragma unroll 2
            for(unsigned int k = lid; k < k_limit; k += MIO_BN_GRP0)
            {
                nidx  = k / MIO_BN_HW;
                hwidx = k - (nidx * MIO_BN_HW);
                if constexpr(MIO_LAYOUT_NHWC)
                {
                    index = nidx * MIO_BN_CHW + hwidx * MIO_BN_C + grpid;
                }
                else
                {
                    index = nidx * MIO_BN_CHW + chwid + hwidx;
                }
                out[index] = fpprec_to_fp<FpType>(
                    fma(pvscale,
                        (fp_to_fpprec<FpPrecType>(in[index]) - fp_to_fpprec<FpPrecType>(mean)) *
                            fp_to_fpprec<FpPrecType>(invVariance),
                        pvbias));
            }
        }
        else
        {
            FpPrecType xhat[MIO_MAX_READ];
#pragma unroll 2
            for(unsigned int k = (MIO_MAX_READ * lid); k < MIO_BN_LESSOUT; k += MIO_BN_CHUNK)
            {
#pragma unroll
                for(unsigned int j = 0; j < MIO_MAX_READ; ++j)
                {
                    const unsigned int l = k + j;
                    nidx                 = l / MIO_BN_HW;
                    hwidx                = l - (nidx * MIO_BN_HW);
                    index                = nidx * MIO_BN_CHW + chwid + hwidx;
                    xhat[j] =
                        (fp_to_fpprec<FpPrecType>(in[index]) - fp_to_fpprec<FpPrecType>(mean)) *
                        fp_to_fpprec<FpPrecType>(invVariance);
                }

                __syncthreads();
#pragma unroll
                for(unsigned int j = 0; j < MIO_MAX_READ; ++j)
                {
                    const unsigned int l = k + j;
                    nidx                 = l / MIO_BN_HW;
                    hwidx                = l - (nidx * MIO_BN_HW);
                    index                = nidx * MIO_BN_CHW + chwid + hwidx;
                    out[index]           = fpprec_to_fp<FpType>(fma(pvscale, xhat[j], pvbias));
                }

                if constexpr(MIO_BN_REMOUT)
                {
                    const unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
#pragma unroll
                    for(unsigned int j = 0; j < MIO_MAX_READ; ++j)
                    {
                        unsigned int l = remkeyout + j;
                        nidx           = l / MIO_BN_HW;
                        hwidx          = l - (nidx * MIO_BN_HW);
                        index          = nidx * MIO_BN_CHW + chwid + hwidx;
                        const auto xin = (index < MIO_BN_NCHW) ? fp_to_fpprec<FpPrecType>(in[index])
                                                               : FpPrecType{0};
                        xhat[j]        = (xin - fp_to_fpprec<FpPrecType>(mean)) *
                                  fp_to_fpprec<FpPrecType>(invVariance);
                    }

                    __syncthreads();
#pragma unroll
                    for(unsigned int j = 0; j < MIO_MAX_READ; ++j)
                    {
                        const unsigned int l = remkeyout + j;
                        nidx                 = l / MIO_BN_HW;
                        hwidx                = l - (nidx * MIO_BN_HW);
                        index                = nidx * MIO_BN_CHW + chwid + hwidx;
                        if(index < MIO_BN_NCHW)
                        {
                            out[index] = fpprec_to_fp<FpType>(fma(pvscale, xhat[j], pvbias));
                        }
                    }
                }
            }
        }

        return;
    }
};

#if (MIO_BN_VARIANT != 2)
extern "C" __global__ void __launch_bounds__(MIO_BN_GRP0* MIO_BN_GRP1* MIO_BN_GRP2)
    MIOpenBatchNormFwdTrainSpatialHIP(
        const FP_TYPE* __restrict in,
        FP_TYPE* __restrict out,
        const FP_TYPE_PREC* __restrict scale,
        const FP_TYPE_PREC* __restrict bias,
        FP_TYPE_PREC INHW,
#if (MIO_RUNNING_RESULT == 1)
        [[maybe_unused]] double expAvgFactor,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningMean,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultRunningVariance,
#endif
        double epsilon
#if (MIO_SAVE_MEAN_VARIANCE == 1)
        ,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveMean,
        [[maybe_unused]] FP_TYPE_PREC* __restrict resultSaveInvVariance
#endif
    )
{
    FP_TYPE_PREC mean, variance, invVariance;
    const unsigned int lid                    = threadIdx.x;
    [[maybe_unused]] const unsigned int grpid = blockIdx.x;

    MIOpenBatchNormFwdTrainSpatialHIPImpl<MIO_BN_VARIANT, FP_TYPE, FP_TYPE_PREC, _FLOAT_ACCUM>{}(
        in, out, scale, bias, INHW, epsilon, mean, variance, invVariance);

    if(lid == 0)
    {
        if constexpr(MIO_RUNNING_RESULT == 1)
        {
            running_stash<_FLOAT_ACCUM, _FLOAT_ACCUM_C, _FLOAT_PREC_C>(
                resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
        }

        if constexpr(MIO_SAVE_MEAN_VARIANCE == 1)
        {
            // saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
        }
    }

    return;
}

#endif
