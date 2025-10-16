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

#if defined(__AMDGCN__) && !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX120X || MIO_BN_GFX115X)
#define MIOPEN_USE_AMDGCN 1
#else
#define MIOPEN_USE_AMDGCN 0
#endif

#include "float_types.h"

#include "activation_functions.hpp"
#include "reduction_functions.hpp"

template <typename TYPE>
using TYPE4 = std::conditional<
    std::is_same<TYPE, half>::value,
    ushort4,
    typename std::conditional<std::is_same<TYPE, double>::value, double4, float4>::type>::type;

using FLOAT_ACCUM4 = TYPE4<FLOAT_ACCUM>;

constexpr static auto SEGTMP   = H * W * (LOCAL_SIZE_X / (H * W));
constexpr static auto SEGMENT  = SEGTMP > BATCH_SIZE* H* W ? BATCH_SIZE* H* W : SEGTMP;
constexpr static auto NLOOP    = SEGMENT > 0 ? (BATCH_SIZE* H* W + SEGMENT - 1) / SEGMENT : 1;
constexpr static auto SEGIHW   = SEGMENT / (H * W);
constexpr static auto NLOOPM   = NLOOP - 1;
constexpr static auto SNHW     = NLOOPM * SEGIHW;
constexpr static auto LDS_SIZE = MIOPEN_USE_AMDGCN ? LDSGCN_SIZE : LDSNOGCN_SIZE;

constexpr static auto MAX_READ = 2;
constexpr static auto GRPRD    = LOCAL_SIZE_X * 4;
constexpr static auto REM4     = BATCH_SIZE * H * W - (BATCH_SIZE * H * W / GRPRD) * GRPRD;
constexpr static auto LESS4    = BATCH_SIZE * H * W - REM4;
constexpr static auto CHUNK    = MAX_READ * LOCAL_SIZE_X;
constexpr static auto REMOUT   = BATCH_SIZE * H * W - (BATCH_SIZE * H * W / CHUNK) * CHUNK;
constexpr static auto LESSOUT  = BATCH_SIZE * H * W - REMOUT;

constexpr static auto MAX_N = 65;

constexpr static auto VALUES_BUFFER_SIZE = VARIANT == 0                         ? NLOOP
                                           : VARIANT == 3 && BATCH_SIZE < MAX_N ? BATCH_SIZE
                                                                                : 1;

template <typename TI, typename TO>
__device__ void activbwdspatial(const TI* __restrict__ x,
                                const TI* __restrict__ y,
                                const TI* __restrict__ dy,
                                TO* __restrict__ dx,
                                const TI diff_scale,
                                const TI gamma,
                                const TI beta,
                                const TI alpha,
                                const float* __restrict__ bn_scale,
                                const float* __restrict__ bn_bias,
                                float* __restrict__ dscale,
                                float* __restrict__ dbias,
                                const float* __restrict__ saved_mean,
                                const float* __restrict__ saved_inv_variance,
                                const float INHW)
{
    using TI4 = TYPE4<TI>;

    FLOAT_ACCUM mean{0};
    FLOAT_ACCUM inv_variance{0};
    FLOAT_ACCUM p_scale{0};
    FLOAT_ACCUM ds{0};
    FLOAT_ACCUM db{0};
    FLOAT_ACCUM xhat{0};

    FLOAT_ACCUM batch_values[VALUES_BUFFER_SIZE];
    FLOAT_ACCUM dy_values[VALUES_BUFFER_SIZE];
    __shared__ FLOAT_ACCUM lscale, lbias;
    __shared__ FLOAT_ACCUM lmean, lvar;

    auto index  = 0;
    auto lid    = threadIdx.x;
    auto gid    = blockIdx.x;
    auto chwid  = gid * H * W + (VARIANT == 0 ? lid % (H * W) : 0);
    auto lidihw = lid / (H * W);
    auto nidx   = 0;
    auto hwidx  = 0;
    FLOAT_ACCUM tmp1{0}, tmp2{0}, tmp3{0};

    if(lid == 0)
    {
        lscale = CVT_FP32_2ACCUM(bn_scale[gid]);
        lbias  = CVT_FP32_2ACCUM(bn_bias[gid]);
        lmean  = CVT_FP32_2ACCUM(saved_mean[gid]);
        lvar   = CVT_FP32_2ACCUM(saved_inv_variance[gid]);
    }
    __syncthreads();
    mean         = lmean;
    inv_variance = lvar;

    if constexpr(VARIANT == 0)
    {
        if(lid < SEGMENT)
        {
            for(auto n = 0; n < NLOOPM; ++n)
            {
                nidx  = n * SEGIHW + lidihw;
                index = nidx * CHANNELS * H * W + chwid;
                xhat  = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));
                dy_values[n] = bn_dy[0];
                db += dy_values[n];
                batch_values[n] = xhat;
                ds              = batch_values[n] * dy_values[n] + ds;
            }
            nidx  = SNHW + lidihw;
            index = nidx * CHANNELS * H * W + chwid;
            if(index < BATCH_SIZE * CHANNELS * H * W)
            {
                xhat = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));
                dy_values[NLOOPM] = bn_dy[0];
            }
            else
            {
                dy_values[NLOOPM] = CVT_FP32_2ACCUM(0);
            }
            db += dy_values[NLOOPM];

            batch_values[NLOOPM] = index < BATCH_SIZE * CHANNELS * H * W
                                       ? (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance
                                       : CVT_FP32_2ACCUM(0);

            // batchvalues is now xhat
            ds = batch_values[NLOOPM] * dy_values[NLOOPM] + ds;
        }
        __syncthreads();

        __shared__ FLOAT_ACCUM lcl_data_x2[LDS_SIZE];
        __shared__ FLOAT_ACCUM lcl_data_y2[LDS_SIZE];
        if constexpr(MIOPEN_USE_AMDGCN)
        {
            miopen::reduction::gcn_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }
        else
        {
            miopen::reduction::lds_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }

        if(lid < SEGMENT)
        {
            p_scale = lscale;

            for(auto n = 0; n < NLOOPM; ++n)
            {
                nidx      = n * SEGIHW + lidihw;
                index     = nidx * CHANNELS * H * W + chwid;
                tmp1      = BATCH_SIZE * H * W * dy_values[n] - db;
                tmp2      = -batch_values[n] * ds;
                tmp3      = p_scale * inv_variance * CVT_FP32_2ACCUM(INHW);
                dx[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
            nidx  = SNHW + lidihw;
            index = nidx * CHANNELS * H * W + chwid;
            if(index < BATCH_SIZE * CHANNELS * H * W)
            {
                tmp1      = BATCH_SIZE * H * W * dy_values[NLOOPM] - db;
                tmp2      = -batch_values[NLOOPM] * ds;
                tmp3      = p_scale * inv_variance * CVT_FP32_2ACCUM(INHW);
                dx[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
        }
        if(lid == 0)
        {
            dbias[gid]  = CVT_ACCUM2FP32(db);
            dscale[gid] = CVT_ACCUM2FP32(ds);
        }
    }
    else if constexpr(VARIANT == 1)
    {
        TI4 xread4;
        FLOAT_ACCUM4 xhat4;
        TI4 act_dy4, act_y4;
        FLOAT_ACCUM4 bn_y4;

        for(auto k = lid << 2; k < LESS4; k += GRPRD)
        {
            nidx    = k / (H * W);
            hwidx   = k - nidx * H * W;
            index   = nidx * CHANNELS * H * W + chwid + hwidx;
            xread4  = *reinterpret_cast<const TI4*>(&x[index]);
            act_dy4 = *reinterpret_cast<const TI4*>(&dy[index]);
            act_y4  = *reinterpret_cast<const TI4*>(&y[index]);

            xhat4.x = (CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&xread4.x)) - mean) * inv_variance;
            xhat4.y = (CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&xread4.y)) - mean) * inv_variance;
            xhat4.z = (CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&xread4.z)) - mean) * inv_variance;
            xhat4.w = (CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&xread4.w)) - mean) * inv_variance;

            bn_y4.x = xhat4.x * lscale + lbias;
            bn_y4.y = xhat4.y * lscale + lbias;
            bn_y4.z = xhat4.z * lscale + lbias;
            bn_y4.w = xhat4.w * lscale + lbias;

            FLOAT_ACCUM p_bn_dy[1];
            FLOAT_ACCUM p_act_dy[1] = {CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.x))};
            FLOAT_ACCUM p_bn_y[1]   = {bn_y4.x};
            FLOAT_ACCUM p_act_y[1]  = {CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.x))};
            ActivationFunction_Diff(p_bn_dy,
                                    p_act_dy,
                                    p_bn_y,
                                    p_act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));

            db += p_bn_dy[0];
            ds += xhat4.x * p_bn_dy[0];
            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.y));
            p_bn_y[0]   = bn_y4.y;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.y));
            ActivationFunction_Diff(p_bn_dy,
                                    p_act_dy,
                                    p_bn_y,
                                    p_act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));

            db += p_bn_dy[0];
            ds += xhat4.y * p_bn_dy[0];
            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.z));
            p_bn_y[0]   = bn_y4.z;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.z));
            ActivationFunction_Diff(p_bn_dy,
                                    p_act_dy,
                                    p_bn_y,
                                    p_act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));

            db += p_bn_dy[0];
            ds += xhat4.z * p_bn_dy[0];
            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.w));
            p_bn_y[0]   = bn_y4.w;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.w));
            ActivationFunction_Diff(p_bn_dy,
                                    p_act_dy,
                                    p_bn_y,
                                    p_act_y,
                                    CVT_FLOAT2ACCUM(diff_scale),
                                    CVT_FLOAT2ACCUM(gamma),
                                    CVT_FLOAT2ACCUM(beta),
                                    CVT_FLOAT2ACCUM(alpha));

            db += p_bn_dy[0];
            ds += xhat4.w * p_bn_dy[0];
        }

        if constexpr(REM4)
        {
            auto remkey = (lid << 2) + LESS4;
            nidx        = remkey / (H * W);
            hwidx       = remkey - nidx * H * W;
            index       = nidx * CHANNELS * H * W + chwid + hwidx;
            if(index < BATCH_SIZE * CHANNELS * H * W)
            {
                xread4  = *reinterpret_cast<const TI4*>(&x[index]);
                act_dy4 = *reinterpret_cast<const TI4*>(&dy[index]);
                act_y4  = *reinterpret_cast<const TI4*>(&y[index]);

                xhat4.x = (CVT_FLOAT2ACCUM(xread4.x) - mean) * inv_variance;
                xhat4.y = (CVT_FLOAT2ACCUM(xread4.y) - mean) * inv_variance;
                xhat4.z = (CVT_FLOAT2ACCUM(xread4.z) - mean) * inv_variance;
                xhat4.w = (CVT_FLOAT2ACCUM(xread4.w) - mean) * inv_variance;

                bn_y4.x = xhat4.x * lscale + lbias;
                bn_y4.y = xhat4.y * lscale + lbias;
                bn_y4.z = xhat4.z * lscale + lbias;
                bn_y4.w = xhat4.w * lscale + lbias;

                FLOAT_ACCUM p_bn_dy[1];
                FLOAT_ACCUM p_act_dy[1] = {CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.x))};
                FLOAT_ACCUM p_bn_y[1]   = {bn_y4.x};
                FLOAT_ACCUM p_act_y[1]  = {CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.x))};
                ActivationFunction_Diff(p_bn_dy,
                                        p_act_dy,
                                        p_bn_y,
                                        p_act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                db += p_bn_dy[0];
                ds += xhat4.x * p_bn_dy[0];
                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.y));
                p_bn_y[0]   = bn_y4.y;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.y));
                ActivationFunction_Diff(p_bn_dy,
                                        p_act_dy,
                                        p_bn_y,
                                        p_act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                db += p_bn_dy[0];
                ds += xhat4.y * p_bn_dy[0];
                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.z));
                p_bn_y[0]   = bn_y4.z;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.z));
                ActivationFunction_Diff(p_bn_dy,
                                        p_act_dy,
                                        p_bn_y,
                                        p_act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                db += p_bn_dy[0];
                ds += xhat4.z * p_bn_dy[0];
                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_dy4.w));
                p_bn_y[0]   = bn_y4.w;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<TI*>(&act_y4.w));
                ActivationFunction_Diff(p_bn_dy,
                                        p_act_dy,
                                        p_bn_y,
                                        p_act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                db += p_bn_dy[0];
                ds += xhat4.w * p_bn_dy[0];
            }
        }

        __syncthreads();

        __shared__ FLOAT_ACCUM lcl_data_x2[LDS_SIZE];
        __shared__ FLOAT_ACCUM lcl_data_y2[LDS_SIZE];
        if constexpr(MIOPEN_USE_AMDGCN)
        {
            miopen::reduction::gcn_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }
        else
        {
            miopen::reduction::lds_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }

        p_scale = lscale;
        tmp3    = p_scale * inv_variance * CVT_FP32_2ACCUM(INHW);
        __syncthreads();
        if(lid == 0)
        {
            dbias[gid]  = CVT_ACCUM2FP32(db);
            dscale[gid] = CVT_ACCUM2FP32(ds);
        }

        FLOAT_ACCUM values[MAX_READ];
        for(auto k = MAX_READ * lid; k < LESSOUT; k += CHUNK)
        {
            for(auto j = 0; j < MAX_READ; ++j)
            {
                auto l = k + j;
                nidx   = l / (H * W);
                hwidx  = l - nidx * H * W;
                index  = nidx * CHANNELS * H * W + chwid + hwidx;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                xhat                  = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));
                tmp1      = BATCH_SIZE * H * W * bn_dy[0] - db;
                tmp2      = -xhat * ds;
                values[j] = tmp3 * (tmp2 + tmp1);
            }
            __syncthreads();
            for(auto j = 0; j < MAX_READ; ++j)
            {
                auto l    = k + j;
                nidx      = l / (H * W);
                hwidx     = l - nidx * H * W;
                index     = nidx * CHANNELS * H * W + chwid + hwidx;
                dx[index] = CVT_ACCUM2FLOAT(values[j]);
            }
        }

        if constexpr(REMOUT)
        {
            auto remkeyout = MAX_READ * lid + LESSOUT;
            for(auto j = 0; j < MAX_READ; ++j)
            {
                auto l = remkeyout + j;
                nidx   = l / (H * W);
                hwidx  = l - nidx * H * W;
                index  = nidx * CHANNELS * H * W + chwid + hwidx;
                if(index < BATCH_SIZE * CHANNELS * H * W)
                {
                    FLOAT_ACCUM bn_dy[1];
                    FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                    xhat                  = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                    FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                    FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                    ActivationFunction_Diff(bn_dy,
                                            act_dy,
                                            bn_y,
                                            act_y,
                                            CVT_FLOAT2ACCUM(diff_scale),
                                            CVT_FLOAT2ACCUM(gamma),
                                            CVT_FLOAT2ACCUM(beta),
                                            CVT_FLOAT2ACCUM(alpha));

                    tmp1      = BATCH_SIZE * H * W * bn_dy[0] - db;
                    tmp2      = -xhat * ds;
                    values[j] = tmp3 * (tmp2 + tmp1);
                }
            }
            __syncthreads();
            for(auto j = 0; j < MAX_READ; ++j)
            {
                auto l = remkeyout + j;
                nidx   = l / (H * W);
                hwidx  = l - nidx * H * W;
                index  = nidx * CHANNELS * H * W + chwid + hwidx;
                if(index < BATCH_SIZE * CHANNELS * H * W)
                {
                    dx[index] = CVT_ACCUM2FLOAT(values[j]);
                }
            }
        }
    }
    else if constexpr(VARIANT == 2)
    {
        // Unused
    }
    else if constexpr(VARIANT == 3)
    {
        if(lid < H * W)
        {
#pragma unroll
            for(auto n = 0; n < BATCH_SIZE; ++n)
            {
                index = n * CHANNELS * H * W + chwid + lid;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                xhat                  = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                if constexpr(BATCH_SIZE < MAX_N)
                {
                    batch_values[n] = xhat;
                    dy_values[n]    = bn_dy[0];
                }

                db += bn_dy[0];
                ds += xhat * bn_dy[0];
            }
        }
        else
        {
            db = 0;
            ds = 0;
        }

        __syncthreads();

        __shared__ FLOAT_ACCUM lcl_data_x2[LDS_SIZE];
        __shared__ FLOAT_ACCUM lcl_data_y2[LDS_SIZE];
        if constexpr(MIOPEN_USE_AMDGCN)
        {
            miopen::reduction::gcn_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }
        else
        {
            miopen::reduction::lds_reduce2<FLOAT_ACCUM, LDS_SIZE>(
                ds, db, CVT_FP32_2ACCUM(1.0), lcl_data_x2, lcl_data_y2, lid);
        }
        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < H * W)
        {
            p_scale = lscale;

#pragma unroll
            for(auto n = 0; n < BATCH_SIZE; ++n)
            {
                index = n * CHANNELS * H * W + chwid + lid;
                if constexpr(BATCH_SIZE < MAX_N)
                {
                    tmp1 = BATCH_SIZE * H * W * dy_values[n] - db;
                    tmp2 = -batch_values[n] * ds;
                }
                else
                {
                    FLOAT_ACCUM bn_dy[1];
                    FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                    xhat                  = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                    FLOAT_ACCUM bn_y[1]   = {xhat * lscale + lbias};
                    FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                    ActivationFunction_Diff(bn_dy,
                                            act_dy,
                                            bn_y,
                                            act_y,
                                            CVT_FLOAT2ACCUM(diff_scale),
                                            CVT_FLOAT2ACCUM(gamma),
                                            CVT_FLOAT2ACCUM(beta),
                                            CVT_FLOAT2ACCUM(alpha));

                    tmp1 = BATCH_SIZE * H * W * bn_dy[0] - db;
                    tmp2 = -xhat * ds;
                }
                tmp3      = p_scale * inv_variance * CVT_FP32_2ACCUM(INHW);
                dx[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
        }
        if(lid == 0)
        {
            dbias[gid]  = CVT_ACCUM2FP32(db);
            dscale[gid] = CVT_ACCUM2FP32(ds);
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) //
    void ActivBwdSpatial(const INPUT_TYPE* __restrict__ x,
                         const INPUT_TYPE* __restrict__ y,
                         const INPUT_TYPE* __restrict__ dy,
                         OUTPUT_TYPE* __restrict__ dx,
                         const INPUT_TYPE diff_scale,
                         const INPUT_TYPE gamma,
                         const INPUT_TYPE beta,
                         const INPUT_TYPE alpha,
                         const float* __restrict__ bn_scale,
                         const float* __restrict__ bn_bias,
                         float* __restrict__ dscale,
                         float* __restrict__ dbias,
                         const float* __restrict__ saved_mean,
                         const float* __restrict__ saved_inv_variance,
                         const float INHW)
{
    activbwdspatial<INPUT_TYPE, OUTPUT_TYPE>(x,
                                             y,
                                             dy,
                                             dx,
                                             diff_scale,
                                             gamma,
                                             beta,
                                             alpha,
                                             bn_scale,
                                             bn_bias,
                                             dscale,
                                             dbias,
                                             saved_mean,
                                             saved_inv_variance,
                                             INHW);
}
