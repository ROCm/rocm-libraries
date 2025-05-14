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

#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"

#ifndef MIO_LAYOUT_NHWC
#define MIO_LAYOUT_NHWC 0
#endif

static_assert(MIO_LAYOUT_NHWC == 0 || MIO_LAYOUT_NHWC == 1, "MIO_LAYOUT_NHWC must be 0 or 1");

#if defined(__AMDGCN__) && !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX120X)
static constexpr bool MIOPEN_USE_AMDGCN = true;
#else
static constexpr bool MIOPEN_USE_AMDGCN = false;
#endif

// there is no half4 implementation in hip so we should implement one
struct half4
{
    half x, y, z, w;

    __host__ __device__ half4() = default;

    __host__ __device__ half4(half a, half b, half c, half d) : x(a), y(b), z(c), w(d) {}

    __host__ __device__ half4& operator=(const half4&) = default;
};

// Hip does have fma which does the same thing as OpenCL mad
template <typename T>
struct HipMad
{
    static_assert(sizeof(T) == sizeof(half) || sizeof(T) == sizeof(float) ||
                      sizeof(T) == sizeof(double),
                  "Input floating point type size is wrong!");
    __forceinline__ __device__ auto operator()(T _1, T _2, T _3)
    {
        if constexpr(sizeof(T) == sizeof(half))
        {
            // TODO: I don't know if this is right, it uses _Float16
            return fma(
                static_cast<_Float16>(_1), static_cast<_Float16>(_2), static_cast<_Float16>(_3));
        }
        else if constexpr(sizeof(T) == sizeof(float))
        {
            return fma(static_cast<float>(_1), static_cast<float>(_2), static_cast<float>(_3));
        }
        else
        {
            return fma(static_cast<double>(_1), static_cast<double>(_2), static_cast<double>(_3));
        }
    }
};

template <int MIoBnVariant>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl
{
};

// This is the instance for MIO_BN_VARIANT == 1
template <>
struct MIOpenBatchNormFwdTrainSpatialHIPImpl<1>
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
    __forceinline__ __device__ void
    operator()([[maybe_unused]] const FP_TYPE* __restrict in,
               [[maybe_unused]] FP_TYPE* __restrict out,
               [[maybe_unused]] const FP_TYPE_PREC* __restrict scale,
               [[maybe_unused]] const FP_TYPE_PREC* __restrict bias,
               [[maybe_unused]] FP_TYPE_PREC INHW,
               [[maybe_unused]] double epsilon)
    {
        [[maybe_unused]] FP_TYPE mean        = 0;
        [[maybe_unused]] FP_TYPE variance    = 0;
        [[maybe_unused]] FP_TYPE invVariance = 0;
        [[maybe_unused]] FP_TYPE pvscale, pvbias;

        [[maybe_unused]] __shared__ FP_TYPE_PREC lcl_bias;
        [[maybe_unused]] __shared__ FP_TYPE_PREC lcl_scale;

        [[maybe_unused]] unsigned int index = 0;
        unsigned int lid                    = threadIdx.x;
        unsigned int grpid                  = blockIdx.x;
#if !MIO_LAYOUT_NHWC
        [[maybe_unused]] unsigned int chwid = grpid * MIO_BN_HW;
#endif
        [[maybe_unused]] unsigned int nidx  = 0;
        [[maybe_unused]] unsigned int hwidx = 0;

        if(lid == 0)
        {
            lcl_scale = *(scale + grpid);
            lcl_bias  = *(bias + grpid);
        }

        __syncthreads();

        if constexpr(!MIO_LAYOUT_NHWC && MIO_BN_HW >= 4096)
        {
            FP_TYPE4 read4;
#pragma unroll
            for(unsigned int k = lid << 2; k < static_cast<decltype(k)>(MIO_BN_LESS4); k += GRPRD)
            {
                nidx  = k / MIO_BN_HW;
                hwidx = k - (nidx * MIO_BN_HW);
                index = nidx * MIO_BN_CHW + chwid + hwidx;
                read4 = *(reinterpret_cast<const FP_TYPE4*>(in + index));
                mean += FLOAT2FLOATPREC(read4.x);
                mean += FLOAT2FLOATPREC(read4.y);
                mean += FLOAT2FLOATPREC(read4.z);
                mean += FLOAT2FLOATPREC(read4.w);
                variance = fma(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
                variance = fma(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
                variance = fma(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
                variance = fma(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
            }

            if constexpr(MIO_BN_REM4)
            {
                unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
                nidx                = remkey / MIO_BN_HW;
                hwidx               = remkey - (nidx * MIO_BN_HW);
                index               = nidx * MIO_BN_CHW + chwid + hwidx;
                // TODO: This is not right, index is unsigned int, MIO_BN_NCHW is int
                // comparing them is not right.
                if(index < (MIO_BN_NCHW - 3))
                {
                    read4 = *(reinterpret_cast<const FP_TYPE4*>(in + index));
                    mean += FLOAT2FLOATPREC(read4.x);
                    mean += FLOAT2FLOATPREC(read4.y);
                    mean += FLOAT2FLOATPREC(read4.z);
                    mean += FLOAT2FLOATPREC(read4.w);
                    variance = fma(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
                    variance = fma(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
                    variance = fma(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
                    variance = fma(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
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
                const auto xin = FLOAT2FLOATPREC(*(in + index));
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
                    const auto xin = (index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(in + index))
                                                           : static_cast<FP_TYPE_PREC>(0);
                    mean += xin;
                    variance = fma(xin, xin, variance);
                }
            }
        }

        __syncthreads();

        constexpr auto lcl_data_size =
            static_cast<bool>(MIOPEN_USE_AMDGCN) ? MIO_BN_LDSGCN_SIZE : MIO_BN_LDS_SIZE;
        [[maybe_unused]] __shared__ _FLOAT_ACCUM lcl_data_x[lcl_data_size];
        [[maybe_unused]] __shared__ _FLOAT_ACCUM lcl_data_y[lcl_data_size];

        if constexpr(!MIOPEN_USE_AMDGCN)
        {
            // TODO: these functions need to be implemented
            // lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
        }
        else
        {
            // gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
        }

        // REDUCTION COMPLETE ---------------------------
        // TODO: it seems that fma doesn't directly supports half, it supports _Float16
        // here I don't know if I should run mad for FP_TYPE_PREC or FP_TYPE, if I only
        // need to run this for FP_TYPE_PREC, then we don't need to have this HipMad struct.
        variance = HipMad<decltype(variance)>{}(-mean, mean, variance);
        if(variance < static_cast<decltype(variance)>(0))
        {
            variance = static_cast<decltype(variance)>(0);
        }
        // TODO: I don't know if this is correct, the input epsilon is double, but it should be
        // casted to FP_TYPE here
        invVariance = rsqrt(variance + static_cast<decltype(variance)>(epsilon));
        pvscale     = lcl_scale;
        pvbias      = lcl_bias;

        if constexpr(MIO_LAYOUT_NHWC || MIO_BN_REM == 0)
        {
            constexpr unsigned int k_limit =
                static_cast<bool>(MIO_LAYOUT_NHWC) ? MIO_BN_NHW : MIO_BN_LESS;
#pragma unroll
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
                out[index] =
                    FLOATPREC2FLOAT(fma(pvscale,
                                        (FLOAT2FLOATPREC(*(in + index)) - FLOAT2FLOATPREC(mean)) *
                                            FLOAT2FLOATPREC(invVariance),
                                        pvbias));
                // TODO: continue to MIOpenBatchNormFwdTrainSpatial.cl:377
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
    MIOpenBatchNormFwdTrainSpatialHIPImpl<MIO_BN_VARIANT>{}(in, out, scale, bias, INHW, epsilon);
#if (MIO_RUNNING_RESULT == 1)
    // TODO: these functions need to be implemented
    // running_stash(
    // resultRunningMean, resultRunningVariance, expAvgFactor, mean, variance, grpid);
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
    // saved_stash(resultSaveMean, resultSaveInvVariance, mean, invVariance, grpid);
#endif
    return;
}

#endif
