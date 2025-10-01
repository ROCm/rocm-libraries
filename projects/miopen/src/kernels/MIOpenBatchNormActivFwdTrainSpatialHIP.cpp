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

#define MIOPEN_USE_AMDGCN 0
#if defined(__AMDGCN__) && !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX120X)
#undef MIOPEN_USE_AMDGCN
#define MIOPEN_USE_AMDGCN 1
#endif

#include "batchnorm_functions.hpp"
#include "activation_functions.hpp"
#include "reduction_functions.hpp"

#if (MIO_BN_VARIANT == 0)

#elif (MIO_BN_VARIANT == 1)

// //===========

// #if(MIO_BN_HW >= 4096)
// #define MIO_MAX_READ 3
// #else
// #define MIO_MAX_READ 2
// #endif
// #define RD_BLK 1
// #define GRPRD (MIO_BN_GRP0 * RD_BLK * 4)
// #define MIO_BN_REM4 (MIO_BN_NHW - ((MIO_BN_NHW / GRPRD) * GRPRD))
// #define MIO_BN_LESS4 (MIO_BN_NHW - MIO_BN_REM4)
// #define MIO_BN_CHUNK4 (MIO_MAX_READ * GRPRD)
// #define MIO_BN_REMOUT4 (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK4) * MIO_BN_CHUNK4))
// #define MIO_BN_LESSOUT4 (MIO_BN_NHW - MIO_BN_REMOUT4)
// #define MIO_BN_REM (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_GRP0) * MIO_BN_GRP0))
// #define MIO_BN_LESS (MIO_BN_NHW - MIO_BN_REM)
// #define MIO_BN_CHUNK (MIO_MAX_READ * MIO_BN_GRP0)
// #define MIO_BN_REMOUT (MIO_BN_NHW - ((MIO_BN_NHW / MIO_BN_CHUNK) * MIO_BN_CHUNK))
// #define MIO_BN_LESSOUT (MIO_BN_NHW - MIO_BN_REMOUT)

// __attribute__((reqd_work_group_size(MIO_BN_GRP0, MIO_BN_GRP1, MIO_BN_GRP2))) __kernel void
// MIOpenBatchNormActivFwdTrainSpatial(

//     float INHW,
//     const _FLOAT alpha,
//     const _FLOAT beta,
//     const _FLOAT gamma,
//     double epsilon,
// #if(MIO_RUNNING_RESULT == 1)
//     double expAvgFactor,
// #endif
//     const __global _FLOAT* __restrict in,
//     __global _FLOAT* __restrict out,
//     __constant _FLOAT_PREC* __restrict bias,
//     __constant _FLOAT_PREC* __restrict scale

// #if(MIO_RUNNING_RESULT == 1)
//     ,
//     __global _FLOAT_PREC* __restrict runningMean,
//     __global _FLOAT_PREC* __restrict runningVariance
// #endif

// #if(MIO_SAVE_MEAN_VARIANCE == 1)
//     ,
//     __global _FLOAT_PREC* __restrict savedInvVariance,
//     __global _FLOAT_PREC* __restrict savedMean
// #endif

// )
// {

//     // SPATIAL

//     _FLOAT_PREC mean        = (_FLOAT_PREC)0.;
//     _FLOAT_PREC variance    = (_FLOAT_PREC)0.;
//     _FLOAT_PREC invVariance = (_FLOAT_PREC)0.;
//     _FLOAT_PREC pvscale, pvbias;
//     _FLOAT_PREC bn_out, act_out;

//     __local _FLOAT_PREC lcl_bias;
//     __local _FLOAT_PREC lcl_scale;

//     int index = 0;
//     int lid   = get_local_id(0);
//     int grpid = get_group_id(0);
//     int chwid = grpid * MIO_BN_HW;
//     int nidx  = 0;
//     int hwidx = 0;

//     if(lid == 0)
//     {
//         lcl_scale = *(scale + grpid);
//         lcl_bias  = *(bias + grpid);
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);

// #if(MIO_BN_HW >= 4096)
//     _FLOAT4 read4;
//     __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid << 2; k < MIO_BN_LESS4;
//                                                k += GRPRD)
//     {
//         nidx  = k / MIO_BN_HW;
//         hwidx = k - (nidx * MIO_BN_HW);
//         index = nidx * MIO_BN_CHW + chwid + hwidx;
//         read4 = *((const global _FLOAT4*)(in + index));
//         mean += FLOAT2FLOATPREC(read4.x);
//         mean += FLOAT2FLOATPREC(read4.y);
//         mean += FLOAT2FLOATPREC(read4.z);
//         mean += FLOAT2FLOATPREC(read4.w);
//         variance = mad(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
//     }

// #if(MIO_BN_REM4)
//     unsigned int remkey = (lid << 2) + MIO_BN_LESS4;
//     nidx                = remkey / MIO_BN_HW;
//     hwidx               = remkey - (nidx * MIO_BN_HW);
//     index               = nidx * MIO_BN_CHW + chwid + hwidx;
//     if(index < MIO_BN_NCHW)
//     {
//         read4 = *((const global _FLOAT4*)(in + index));
//         mean += FLOAT2FLOATPREC(read4.x);
//         mean += FLOAT2FLOATPREC(read4.y);
//         mean += FLOAT2FLOATPREC(read4.z);
//         mean += FLOAT2FLOATPREC(read4.w);
//         variance = mad(FLOAT2FLOATPREC(read4.x), FLOAT2FLOATPREC(read4.x), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.y), FLOAT2FLOATPREC(read4.y), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.z), FLOAT2FLOATPREC(read4.z), variance);
//         variance = mad(FLOAT2FLOATPREC(read4.w), FLOAT2FLOATPREC(read4.w), variance);
//     }

// #endif

// #else
//     __attribute__((opencl_unroll_hint(4))) for(unsigned int k = lid; k < MIO_BN_LESS;
//                                                k += MIO_BN_GRP0)
//     {
//         nidx            = k / MIO_BN_HW;
//         hwidx           = k - (nidx * MIO_BN_HW);
//         index           = nidx * MIO_BN_CHW + chwid + hwidx;
//         _FLOAT_PREC xin = FLOAT2FLOATPREC(*(in + index));
//         mean += xin;
//         variance = mad(xin, xin, variance);
//     }
// #if(MIO_BN_REM)
//     if(lid < MIO_BN_REM)
//     {
//         unsigned int remkey = lid + MIO_BN_LESS;
//         nidx                = remkey / MIO_BN_HW;
//         hwidx               = remkey - (nidx * MIO_BN_HW);
//         index               = nidx * MIO_BN_CHW + chwid + hwidx;
//         _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(in + index)) :
//         (_FLOAT_PREC)0.; mean += xin; variance = mad(xin, xin, variance);
//     }
// #endif
// #endif
//     barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

// // REDUCE MEAN AND VARIANCE -----------------------
// #if !MIOPEN_USE_AMDGCN
//     local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDS_SIZE];
//     local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDS_SIZE];
//     lds_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
// #else
//     local _FLOAT_ACCUM lcl_data_x[MIO_BN_LDSGCN_SIZE];
//     local _FLOAT_ACCUM lcl_data_y[MIO_BN_LDSGCN_SIZE];
//     gcn_reduce2(&mean, &variance, (_FLOAT_ACCUM)INHW, lcl_data_x, lcl_data_y, lid);
// #endif

//     // REDUCTION COMPLETE ---------------------------
//     variance    = mad(-mean, mean, variance);
//     invVariance = rsqrt(variance + epsilon);

//     pvscale = lcl_scale;
//     pvbias  = lcl_bias;

// #if(MIO_BN_REM == 0)
//     __attribute__((opencl_unroll_hint(2))) for(unsigned int k = lid; k < MIO_BN_LESS;
//                                                k += MIO_BN_GRP0)
//     {
//         nidx   = k / MIO_BN_HW;
//         hwidx  = k - (nidx * MIO_BN_HW);
//         index  = nidx * MIO_BN_CHW + chwid + hwidx;
//         bn_out = mad(pvscale, (*(in + index) - mean) * invVariance, pvbias);
//         ActivationFunction(1,
//                            &act_out,
//                            &bn_out,
//                            FLOAT2FLOATPREC(gamma),
//                            FLOAT2FLOATPREC(beta),
//                            FLOAT2FLOATPREC(alpha));
//         out[index] = FLOATPREC2FLOAT(act_out);

//     } // end for
// #else
//     _FLOAT_PREC xhat[MIO_MAX_READ];
//     __attribute__((opencl_unroll_hint(2))) for(unsigned int k = (MIO_MAX_READ * lid);
//                                                k < MIO_BN_LESSOUT;
//                                                k += MIO_BN_CHUNK)
//     {
//         for(unsigned int j = 0; j < MIO_MAX_READ; j++)
//         {
//             unsigned int l = k + j;
//             nidx           = l / MIO_BN_HW;
//             hwidx          = l - (nidx * MIO_BN_HW);
//             index          = nidx * MIO_BN_CHW + chwid + hwidx;
//             xhat[j]        = (FLOAT2FLOATPREC(*(in + index)) - mean) * invVariance;
//         }
//         barrier(CLK_GLOBAL_MEM_FENCE);
//         for(unsigned int j = 0; j < MIO_MAX_READ; j++)
//         {
//             unsigned int l = k + j;
//             nidx           = l / MIO_BN_HW;
//             hwidx          = l - (nidx * MIO_BN_HW);
//             index          = nidx * MIO_BN_CHW + chwid + hwidx;
//             bn_out         = mad(pvscale, xhat[j], pvbias);
//             ActivationFunction(1,
//                                &act_out,
//                                &bn_out,
//                                FLOAT2FLOATPREC(gamma),
//                                FLOAT2FLOATPREC(beta),
//                                FLOAT2FLOATPREC(alpha));
//             out[index] = FLOATPREC2FLOAT(act_out);
//         }
//     } // end for

// #if(MIO_BN_REMOUT)
//     unsigned int remkeyout = (MIO_MAX_READ * lid) + MIO_BN_LESSOUT;
//     for(unsigned int j = 0; j < MIO_MAX_READ; j++)
//     {
//         unsigned int l  = remkeyout + j;
//         nidx            = l / MIO_BN_HW;
//         hwidx           = l - (nidx * MIO_BN_HW);
//         index           = nidx * MIO_BN_CHW + chwid + hwidx;
//         _FLOAT_PREC xin = (index < MIO_BN_NCHW) ? FLOAT2FLOATPREC(*(in + index)) :
//         (_FLOAT_PREC)0.; xhat[j]         = (xin - mean) * invVariance;
//     }
//     barrier(CLK_GLOBAL_MEM_FENCE);
//     for(unsigned int j = 0; j < MIO_MAX_READ; j++)
//     {
//         unsigned int l = remkeyout + j;
//         nidx           = l / MIO_BN_HW;
//         hwidx          = l - (nidx * MIO_BN_HW);
//         index          = nidx * MIO_BN_CHW + chwid + hwidx;
//         if(index < MIO_BN_NCHW)
//         {
//             bn_out = mad(pvscale, xhat[j], pvbias);
//             ActivationFunction(1,
//                                &act_out,
//                                &bn_out,
//                                FLOAT2FLOATPREC(gamma),
//                                FLOAT2FLOATPREC(beta),
//                                FLOAT2FLOATPREC(alpha));
//             out[index] = FLOATPREC2FLOAT(act_out);
//         }
//     }
// #endif
// #endif

//     if(lid == 0)
//     {
// #if(MIO_RUNNING_RESULT == 1)
//         running_stash(runningMean, runningVariance, expAvgFactor, mean, variance, grpid);
// #endif

// #if(MIO_SAVE_MEAN_VARIANCE == 1)
//         saved_stash(savedMean, savedInvVariance, mean, invVariance, grpid);
// #endif
//     }

// } // end spatial norm

#elif (MIO_BN_VARIANT == 2)
// MULTI-KERNEL reduction for > 33M elements

// TODO static assert not imlemented!

#elif (MIO_BN_VARIANT == 3)

// ported ...

#endif

// Load the configs to this file
namespace /*anonymous*/ {
using mio_config    = miopen::config;
using mio_bn_config = miopen::batchnorm::config;
} // namespace

namespace miopen {
namespace batchnorm {

template <int MIoBnVariant, typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl
{
    static_assert(false, "This variant is not supported.");
};

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

        FpPrecType pvscale = 0;
        FpPrecType pvbias  = 0;
        FpPrecType batchvalues[nloop];

        unsigned int index  = 0;
        unsigned int lid    = threadIdx.x;
        unsigned int grpid  = blockIdx.x;
        unsigned int chwid  = grpid * mio_bn_config::hw + (lid % mio_bn_config::hw);
        unsigned int lidihw = lid / mio_bn_config::hw;
        unsigned int nid    = 0;
        FpPrecType bn_out   = 0.;
        FpPrecType act_out  = 0.;

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
            // #if(MIOPEN_USE_FP16 == 0)
            //         __attribute__((opencl_unroll_hint(2)))
            // #endif
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

        constexpr auto lcl_data_size =
            mio_bn_config::use_amdgnc ? mio_bn_config::lds_gcn_size : mio_bn_config::lds_size;
        __shared__ FpAccumType lcl_data_x[lcl_data_size];
        __shared__ FpAccumType lcl_data_y[lcl_data_size];
        if constexpr(mio_bn_config::use_amdgnc)
        {
            miopen::reduction::gcn_reduce2<FpAccumType, lcl_data_size>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(INHW),
                lcl_data_x,
                lcl_data_y,
                lid);
        }
        else
        {
            miopen::reduction::lds_reduce2<FpAccumType, lcl_data_size>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(INHW),
                lcl_data_x,
                lcl_data_y,
                lid);
        }

        variance    = fma(-mean, mean, variance);
        invVariance = rsqrt(variance + cast<FpPrecType>(epsilon));
        pvscale     = lcl_scale;
        pvbias      = lcl_bias;

        if(lid < segment)
        {
            FpPrecType inhat = 0;

            // Apply normalization
            for(unsigned int n = 0; n < nloopm; n++)
            {
                inhat  = (batchvalues[n] - mean) * invVariance;
                nid    = n * segihw + lidihw;
                index  = nid * mio_bn_config::chw + chwid;
                bn_out = fma(pvscale, inhat, pvbias);
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
                bn_out = fma(pvscale, inhat, pvbias);
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

template <typename FpType, typename FpPrecType, typename FpAccumType>
struct MIOpenBatchNormActivFwdTrainSpatialHIPImpl<1, FpType, FpPrecType, FpAccumType>
{
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
    {}
};

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
        FpPrecType minibatch[NormPerN ? mio_bn_config::n : 1];

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
                auto xin           = miopen::batchnorm::cast<FpPrecType>(in[index]);
                if constexpr(NormPerN)
                {
                    minibatch[n] = xin;
                }
                mean += xin;
                variance = fma(xin, xin, variance);
            }};
        }

        __syncthreads();

        if constexpr(mio_config::use_amdgnc)
        {
            __shared__ FpAccumType lcl_data_x2[MIO_BN_LDSGCN_SIZE];
            __shared__ FpAccumType lcl_data_y2[MIO_BN_LDSGCN_SIZE];
            miopen::reduction::gcn_reduce2<FpAccumType, MIO_BN_LDSGCN_SIZE>(
                reinterpret_cast<FpAccumType&>(mean),
                reinterpret_cast<FpAccumType&>(variance),
                static_cast<FpAccumType>(INHW),
                lcl_data_x2,
                lcl_data_y2,
                lid);
        }
        else
        {
#if MIOPEN_USE_FP16 == 1
            //     local float lcl_data[MIO_BN_LDS_SIZE];
            //     lcl_data[lid] = (float)mean;
            //     barrier(CLK_LOCAL_MEM_FENCE);
            //     for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
            //     {
            //         if(lid < red)
            //             lcl_data[lid] += lcl_data[lid + red];
            //         barrier(CLK_LOCAL_MEM_FENCE);
            //     }
            //     float temp_mean = (float)mean;
            //     regLDSreduce(&temp_mean, lcl_data, lid, (float)INHW);
            //     mean = (_FLOAT_PREC)temp_mean;
            //     barrier(CLK_LOCAL_MEM_FENCE);
            //     lcl_data[lid] = (float)variance;
            //     barrier(CLK_LOCAL_MEM_FENCE);

            //     for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
            //     {
            //         if(lid < red)
            //             lcl_data[lid] += lcl_data[lid + red];
            //         barrier(CLK_LOCAL_MEM_FENCE);
            //     }
            //     float temp_variance = (float)variance;
            //     regLDSreduce(&temp_variance, lcl_data, lid, (float)INHW);
            //     variance = (_FLOAT_PREC)temp_variance;
#else
            FpPrecType lcl_data[MIO_BN_LDS_SIZE];

            // Reduce mean
            lcl_data[lid] = mean;
            __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
            {
                if(lid < red)
                {
                    lcl_data[lid] += lcl_data[lid + red];
                }
                __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);
            }
            regLDSreduce(&mean, lcl_data, lid, static_cast<FpPrecType>(INHW));
            __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);

            // Reduce variance
            lcl_data[lid] = variance;
            __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);
            for(unsigned int red = (MIO_BN_GRP0 >> 1); red > 256; red >>= 1)
            {
                if(lid < red)
                    lcl_data[lid] += lcl_data[lid + red];
                __syncthreads(); // barrier(CLK_LOCAL_MEM_FENCE);
            }
            regLDSreduce(&variance, lcl_data, lid, static_cast<FpPrecType>(INHW));
#endif
        }

        __syncthreads();

        variance    = fma(-mean, mean, variance);
        invVariance = rsqrt(variance + FpPrecType(epsilon));

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
                                                  miopen::batchnorm::cast<FpPrecType>(gamma),
                                                  miopen::batchnorm::cast<FpPrecType>(beta),
                                                  miopen::batchnorm::cast<FpPrecType>(alpha));
                out[index] = miopen::batchnorm::cast<FpPrecType>(act_out);
            } // end for
        } // end if
    }
};

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
#if (MIO_RUNNING_RESULT == 1)
        double expAvgFactor,
#endif
        const typename mio_bn_config::fp_type* __restrict in,
        typename mio_bn_config::fp_type* __restrict out,
        const typename mio_bn_config::fp_prec_type* __restrict bias,
        const typename mio_bn_config::fp_prec_type* __restrict scale
#if (MIO_RUNNING_RESULT == 1)
        ,
        typename mio_bn_config::fp_prec_type* __restrict runningMean,
        typename mio_bn_config::fp_prec_type* __restrict runningVariance
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
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
#if (MIO_RUNNING_RESULT == 1)
        using StashUpdater = miopen::batchnorm::StashUpdater<fp_accum_c_type>;
        StashUpdater updater(miopen::batchnorm::cast<fp_accum_c_type>(mean),
                             miopen::batchnorm::cast<fp_accum_c_type>(variance),
                             miopen::batchnorm::cast<fp_accum_c_type>(expAvgFactor));

        miopen::batchnorm::running_stash<fp_accum_c_type, fp_prec_c_type, StashUpdater>(
            runningMean, runningVariance, updater, grpid);
#endif
#if (MIO_SAVE_MEAN_VARIANCE == 1)
        miopen::batchnorm::saved_stash<fp_accum_c_type, fp_prec_c_type>(
            savedMean, savedInvVariance, mean, invVariance, grpid);
#endif
    }
}
