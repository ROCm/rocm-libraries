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
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#if defined(__AMDGCN__) && \
    !(MIO_BN_GFX103X || MIO_BN_GFX110X || MIO_BN_GFX115X || MIO_BN_GFX120X || MIO_BN_GFX125X)
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

constexpr static unsigned int SEGTMP   = MIO_BN_HW * (LOCAL_SIZE_X / (MIO_BN_HW));
constexpr static unsigned int SEGMENT  = SEGTMP > MIO_BN_NHW ? MIO_BN_NHW : SEGTMP;
constexpr static unsigned int NLOOP    = SEGMENT > 0 ? (MIO_BN_NHW + SEGMENT - 1) / SEGMENT : 1;
constexpr static unsigned int SEGIHW   = SEGMENT / (MIO_BN_HW);
constexpr static unsigned int NLOOPM   = NLOOP - 1;
constexpr static unsigned int SNHW     = NLOOPM * SEGIHW;
constexpr static unsigned int LDS_SIZE = MIOPEN_USE_AMDGCN ? LDSGCN_SIZE : LDSNOGCN_SIZE;

constexpr static unsigned int MAX_READ = 2;
constexpr static unsigned int GRPRD    = LOCAL_SIZE_X * 4;
constexpr static unsigned int REM4     = MIO_BN_NHW - (MIO_BN_NHW / GRPRD) * GRPRD;
constexpr static unsigned int LESS4    = MIO_BN_NHW - REM4;
constexpr static unsigned int CHUNK    = MAX_READ * LOCAL_SIZE_X;
constexpr static unsigned int REMOUT   = MIO_BN_NHW - (MIO_BN_NHW / CHUNK) * CHUNK;
constexpr static unsigned int LESSOUT  = MIO_BN_NHW - REMOUT;

constexpr static unsigned int MAX_N = 65;

template <typename T>
__forceinline__ __device__ void activbwdspatial(const T* __restrict__ x,
                                                const T* __restrict__ y,
                                                const T* __restrict__ dy,
                                                T* __restrict__ dx,
                                                const T diff_scale,
                                                const T gamma,
                                                const T beta,
                                                const T alpha,
                                                const float* __restrict__ bn_scale,
                                                const float* __restrict__ bn_bias,
                                                float* __restrict__ dscale,
                                                float* __restrict__ dbias,
                                                const float* __restrict__ saved_mean,
                                                const float* __restrict__ saved_inv_variance)
{
    const unsigned int lid   = threadIdx.x;
    const unsigned int gid   = blockIdx.x;
    const unsigned int chwid = gid * MIO_BN_HW + (MIO_BN_VARIANT == 0 ? lid % (MIO_BN_HW) : 0);

    __shared__ FLOAT_ACCUM scale, bias, mean, inv_variance;
    if(lid == 0)
    {
        scale        = CVT_FP32_2ACCUM(bn_scale[gid]);
        bias         = CVT_FP32_2ACCUM(bn_bias[gid]);
        mean         = CVT_FP32_2ACCUM(saved_mean[gid]);
        inv_variance = CVT_FP32_2ACCUM(saved_inv_variance[gid]);
    }
    __syncthreads();

    FLOAT_ACCUM tmp3 = scale * inv_variance * CVT_FP32_2ACCUM(1.0f / (MIO_BN_NHW));

    FLOAT_ACCUM ds{0};
    FLOAT_ACCUM db{0};

    if constexpr(MIO_BN_VARIANT == 0)
    {
        FLOAT_ACCUM batch_values[NLOOP];
        FLOAT_ACCUM dy_values[NLOOP];

        const unsigned int lidihw = lid / (MIO_BN_HW);

        if(lid < SEGMENT)
        {
            for(unsigned int n = 0; n < NLOOPM; ++n)
            {
                unsigned int nidx  = n * SEGIHW + lidihw;
                unsigned int index = nidx * MIO_BN_CHW + chwid;
                FLOAT_ACCUM xhat   = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
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
                ds += xhat * dy_values[n];
            }
            unsigned int nidx  = SNHW + lidihw;
            unsigned int index = nidx * MIO_BN_CHW + chwid;
            FLOAT_ACCUM xhat   = index < MIO_BN_NCHW
                                     ? (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance
                                     : CVT_FP32_2ACCUM(0);
            if(index < MIO_BN_NCHW)
            {
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
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
            ds += xhat * dy_values[NLOOPM];
            batch_values[NLOOPM] = xhat;
        }
        __syncthreads();

        miopen::reduction::reduce2<FLOAT_ACCUM, LOCAL_SIZE_X>(ds, db, CVT_FP32_2ACCUM(1.0f), lid);

        if(lid < SEGMENT)
        {
            for(unsigned int n = 0; n < NLOOPM; ++n)
            {
                unsigned int nidx  = n * SEGIHW + lidihw;
                unsigned int index = nidx * MIO_BN_CHW + chwid;
                FLOAT_ACCUM tmp1   = MIO_BN_NHW * dy_values[n] - db;
                FLOAT_ACCUM tmp2   = -batch_values[n] * ds;
                dx[index]          = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
            unsigned int nidx  = SNHW + lidihw;
            unsigned int index = nidx * MIO_BN_CHW + chwid;
            if(index < MIO_BN_NCHW)
            {
                FLOAT_ACCUM tmp1 = MIO_BN_NHW * dy_values[NLOOPM] - db;
                FLOAT_ACCUM tmp2 = -batch_values[NLOOPM] * ds;
                dx[index]        = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
        }
    }
    else if constexpr(MIO_BN_VARIANT == 1)
    {
        using T4           = TYPE4<T>;
        using FLOAT_ACCUM4 = TYPE4<FLOAT_ACCUM>;

        for(unsigned int k = lid << 2; k < LESS4; k += GRPRD)
        {
            unsigned int nidx  = k / (MIO_BN_HW);
            unsigned int hwidx = k - nidx * MIO_BN_HW;
            unsigned int index = nidx * MIO_BN_CHW + chwid + hwidx;
            T4 xread4          = *reinterpret_cast<const T4*>(&x[index]);
            T4 act_dy4         = *reinterpret_cast<const T4*>(&dy[index]);
            T4 act_y4          = *reinterpret_cast<const T4*>(&y[index]);

            FLOAT_ACCUM4 xhat4;
            xhat4.x = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.x)) - mean) * inv_variance;
            xhat4.y = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.y)) - mean) * inv_variance;
            xhat4.z = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.z)) - mean) * inv_variance;
            xhat4.w = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.w)) - mean) * inv_variance;

            FLOAT_ACCUM4 bn_y4;
            bn_y4.x = xhat4.x * scale + bias;
            bn_y4.y = xhat4.y * scale + bias;
            bn_y4.z = xhat4.z * scale + bias;
            bn_y4.w = xhat4.w * scale + bias;

            FLOAT_ACCUM p_bn_dy[1];
            FLOAT_ACCUM p_act_dy[1] = {CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.x))};
            FLOAT_ACCUM p_bn_y[1]   = {bn_y4.x};
            FLOAT_ACCUM p_act_y[1]  = {CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.x))};
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

            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.y));
            p_bn_y[0]   = bn_y4.y;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.y));
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

            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.z));
            p_bn_y[0]   = bn_y4.z;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.z));
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

            p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.w));
            p_bn_y[0]   = bn_y4.w;
            p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.w));
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
            unsigned int remkey = (lid << 2) + LESS4;
            unsigned int nidx   = remkey / (MIO_BN_HW);
            unsigned int hwidx  = remkey - nidx * MIO_BN_HW;
            unsigned int index  = nidx * MIO_BN_CHW + chwid + hwidx;
            if(index < MIO_BN_NCHW)
            {
                T4 xread4  = *reinterpret_cast<const T4*>(&x[index]);
                T4 act_dy4 = *reinterpret_cast<const T4*>(&dy[index]);
                T4 act_y4  = *reinterpret_cast<const T4*>(&y[index]);

                FLOAT_ACCUM4 xhat4;
                xhat4.x = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.x)) - mean) * inv_variance;
                xhat4.y = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.y)) - mean) * inv_variance;
                xhat4.z = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.z)) - mean) * inv_variance;
                xhat4.w = (CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&xread4.w)) - mean) * inv_variance;

                FLOAT_ACCUM4 bn_y4;
                bn_y4.x = xhat4.x * scale + bias;
                bn_y4.y = xhat4.y * scale + bias;
                bn_y4.z = xhat4.z * scale + bias;
                bn_y4.w = xhat4.w * scale + bias;

                FLOAT_ACCUM p_bn_dy[1];
                FLOAT_ACCUM p_act_dy[1] = {CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.x))};
                FLOAT_ACCUM p_bn_y[1]   = {bn_y4.x};
                FLOAT_ACCUM p_act_y[1]  = {CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.x))};
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

                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.y));
                p_bn_y[0]   = bn_y4.y;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.y));
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

                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.z));
                p_bn_y[0]   = bn_y4.z;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.z));
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

                p_act_dy[0] = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_dy4.w));
                p_bn_y[0]   = bn_y4.w;
                p_act_y[0]  = CVT_FLOAT2ACCUM(*reinterpret_cast<T*>(&act_y4.w));
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

        miopen::reduction::reduce2<FLOAT_ACCUM, LOCAL_SIZE_X>(ds, db, CVT_FP32_2ACCUM(1.0f), lid);

        tmp3 = scale * inv_variance * CVT_FP32_2ACCUM(1.0f / (MIO_BN_NHW));
        __syncthreads();

        FLOAT_ACCUM values[MAX_READ];
        for(unsigned int k = MAX_READ * lid; k < LESSOUT; k += CHUNK)
        {
            for(unsigned int j = 0; j < MAX_READ; ++j)
            {
                unsigned int l     = k + j;
                unsigned int nidx  = l / (MIO_BN_HW);
                unsigned int hwidx = l - nidx * MIO_BN_HW;
                unsigned int index = nidx * MIO_BN_CHW + chwid + hwidx;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));
                FLOAT_ACCUM tmp1 = MIO_BN_NHW * bn_dy[0] - db;
                FLOAT_ACCUM tmp2 = -xhat * ds;
                values[j]        = tmp3 * (tmp2 + tmp1);
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
            for(unsigned int j = 0; j < MAX_READ; ++j)
            {
                unsigned int l     = k + j;
                unsigned int nidx  = l / (MIO_BN_HW);
                unsigned int hwidx = l - nidx * MIO_BN_HW;
                unsigned int index = nidx * MIO_BN_CHW + chwid + hwidx;
                dx[index]          = CVT_ACCUM2FLOAT(values[j]);
            }
        }

        if constexpr(REMOUT)
        {
            unsigned int remkeyout = MAX_READ * lid + LESSOUT;
            for(unsigned int j = 0; j < MAX_READ; ++j)
            {
                unsigned int l     = remkeyout + j;
                unsigned int nidx  = l / (MIO_BN_HW);
                unsigned int hwidx = l - nidx * MIO_BN_HW;
                unsigned int index = nidx * MIO_BN_CHW + chwid + hwidx;
                if(index < MIO_BN_NCHW)
                {
                    FLOAT_ACCUM bn_dy[1];
                    FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                    FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                    FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
                    FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                    ActivationFunction_Diff(bn_dy,
                                            act_dy,
                                            bn_y,
                                            act_y,
                                            CVT_FLOAT2ACCUM(diff_scale),
                                            CVT_FLOAT2ACCUM(gamma),
                                            CVT_FLOAT2ACCUM(beta),
                                            CVT_FLOAT2ACCUM(alpha));

                    FLOAT_ACCUM tmp1 = MIO_BN_NHW * bn_dy[0] - db;
                    FLOAT_ACCUM tmp2 = -xhat * ds;
                    values[j]        = tmp3 * (tmp2 + tmp1);
                }
            }
            __syncthreads();
            for(unsigned int j = 0; j < MAX_READ; ++j)
            {
                unsigned int l     = remkeyout + j;
                unsigned int nidx  = l / (MIO_BN_HW);
                unsigned int hwidx = l - nidx * MIO_BN_HW;
                unsigned int index = nidx * MIO_BN_CHW + chwid + hwidx;
                if(index < MIO_BN_NCHW)
                {
                    dx[index] = CVT_ACCUM2FLOAT(values[j]);
                }
            }
        }
    }
    else if constexpr(MIO_BN_VARIANT == 2)
    {
        // Unused
    }
    else if constexpr(MIO_BN_VARIANT == 3)
    {
        FLOAT_ACCUM batch_values[MIO_BN_N];
        FLOAT_ACCUM dy_values[MIO_BN_N];
        if(lid < MIO_BN_HW)
        {
#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; ++n)
            {
                unsigned int index = n * MIO_BN_CHW + chwid + lid;
                FLOAT_ACCUM bn_dy[1];
                FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
                FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                ActivationFunction_Diff(bn_dy,
                                        act_dy,
                                        bn_y,
                                        act_y,
                                        CVT_FLOAT2ACCUM(diff_scale),
                                        CVT_FLOAT2ACCUM(gamma),
                                        CVT_FLOAT2ACCUM(beta),
                                        CVT_FLOAT2ACCUM(alpha));

                if constexpr(MIO_BN_N < MAX_N)
                {
                    batch_values[n] = xhat;
                    dy_values[n]    = bn_dy[0];
                }

                db += bn_dy[0];
                ds += xhat * bn_dy[0];
            }
        }

        __syncthreads();

        miopen::reduction::reduce2<FLOAT_ACCUM, LOCAL_SIZE_X>(ds, db, CVT_FP32_2ACCUM(1.0f), lid);
        __syncthreads();

        // Group level reduction
        // Need to reduce over all elements in NxHxW
        // move across the sections of an image in the mini_batch stack
        if(lid < MIO_BN_HW)
        {
#pragma unroll
            for(unsigned int n = 0; n < MIO_BN_N; ++n)
            {
                unsigned int index = n * MIO_BN_CHW + chwid + lid;
                FLOAT_ACCUM tmp1, tmp2;
                if constexpr(MIO_BN_N < MAX_N)
                {
                    tmp1 = MIO_BN_NHW * dy_values[n] - db;
                    tmp2 = -batch_values[n] * ds;
                }
                else
                {
                    FLOAT_ACCUM bn_dy[1];
                    FLOAT_ACCUM act_dy[1] = {CVT_FLOAT2ACCUM(dy[index])};
                    FLOAT_ACCUM xhat      = (CVT_FLOAT2ACCUM(x[index]) - mean) * inv_variance;
                    FLOAT_ACCUM bn_y[1]   = {xhat * scale + bias};
                    FLOAT_ACCUM act_y[1]  = {CVT_FLOAT2ACCUM(y[index])};
                    ActivationFunction_Diff(bn_dy,
                                            act_dy,
                                            bn_y,
                                            act_y,
                                            CVT_FLOAT2ACCUM(diff_scale),
                                            CVT_FLOAT2ACCUM(gamma),
                                            CVT_FLOAT2ACCUM(beta),
                                            CVT_FLOAT2ACCUM(alpha));

                    tmp1 = MIO_BN_NHW * bn_dy[0] - db;
                    tmp2 = -xhat * ds;
                }
                tmp3      = scale * inv_variance * CVT_FP32_2ACCUM(1.0f / (MIO_BN_NHW));
                dx[index] = CVT_ACCUM2FLOAT(tmp3 * (tmp2 + tmp1));
            }
        }
    }

    if(lid == 0)
    {
        dbias[gid]  = CVT_ACCUM2FP32(db);
        dscale[gid] = CVT_ACCUM2FP32(ds);
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) //
    void MIOpenBatchNormActivBwdSpatial(const DATA_TYPE* __restrict__ x,
                                        const DATA_TYPE* __restrict__ y,
                                        const DATA_TYPE* __restrict__ dy,
                                        DATA_TYPE* __restrict__ dx,
                                        const DATA_TYPE diff_scale,
                                        const DATA_TYPE gamma,
                                        const DATA_TYPE beta,
                                        const DATA_TYPE alpha,
                                        const float* __restrict__ bn_scale,
                                        const float* __restrict__ bn_bias,
                                        float* __restrict__ dscale,
                                        float* __restrict__ dbias,
                                        const float* __restrict__ saved_mean,
                                        const float* __restrict__ saved_inv_variance)
{
    activbwdspatial<DATA_TYPE>(x,
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
                               saved_inv_variance);
}
