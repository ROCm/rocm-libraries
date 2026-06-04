/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include "float_types.h"
#include "miopen_cstdint.hpp"

template <int N>
struct log2_floor
{
    constexpr static int value = log2_floor<(N >> 1)>::value + 1;
};

template <>
struct log2_floor<1>
{
    constexpr static int value = 0;
};

template <int N>
constexpr static int log2_floor_v = log2_floor<N>::value;

template <int N>
struct log2_ceil
{
    constexpr static int value = log2_floor_v<N> + (1 << log2_floor_v<N> == N ? 0 : 1);
};

template <int N>
constexpr static int log2_ceil_v = log2_ceil<N>::value;

using load_t = int4;

template <typename T, unsigned int n>
struct array
{
    T data[n];
};

template <typename T>
static constexpr int load_factor = sizeof(load_t) / sizeof(T);

template <typename T>
using vec_t = array<T, load_factor<T>>;

template <typename T, unsigned int BOUND = INNER_SIZE, unsigned int I_STRIDE = STRIDE>
__forceinline__ __device__ static vec_t<T>
load(uint64_t i, const uint64_t i_offset, const T* __restrict__ src)
{
    if(I_STRIDE == 1 && i + load_factor<T> < BOUND)
    {
        __builtin_amdgcn_sched_barrier(1);
        const load_t value = *reinterpret_cast<const load_t*>(&src[i + i_offset]);
        const auto values  = *reinterpret_cast<const vec_t<T>*>(&value);
        return values;
    }
    else
    {
        __builtin_amdgcn_sched_barrier(1);
        vec_t<T> values{{}};
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i + k < BOUND)
            {
                values.data[k] = src[(i + k) * I_STRIDE + i_offset];
            }
        }
        return values;
    }
}

template <typename T, unsigned int BOUND = INNER_SIZE, bool USE_DEFAULT = false>
__forceinline__ __device__ static vec_t<T> load_contiguous(
    uint64_t i, const T* __restrict__ src, const T default_value = CVT_FP32_2FLOAT(0.0f))
{
    if(!USE_DEFAULT && i + load_factor<T> < BOUND)
    {
        const load_t value = *reinterpret_cast<const load_t*>(&src[i]);
        const auto values  = *reinterpret_cast<const vec_t<T>*>(&value);
        return values;
    }
    else
    {
        __builtin_amdgcn_sched_barrier(1);
        vec_t<T> values = {{}};
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i + k < BOUND)
            {
                if constexpr(USE_DEFAULT)
                {
                    values.data[k] = static_cast<T>(default_value);
                }
                else
                {
                    values.data[k] = src[i + k];
                }
            }
        }
        return values;
    }
}

template <typename T>
__forceinline__ __device__ static void
store(uint64_t i, const uint64_t i_offset, T* __restrict__ dst, vec_t<T>& data)
{
    if(STRIDE == 1 && i + load_factor<T> < INNER_SIZE)
    {
        *reinterpret_cast<load_t*>(&dst[i * STRIDE + i_offset]) = *reinterpret_cast<load_t*>(&data);
    }
    else
    {
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i + k < INNER_SIZE)
            {
                dst[(i + k) * STRIDE + i_offset] = data.data[k];
            }
        }
    }
}

__forceinline__ __device__ void get_indices(unsigned int& gid, unsigned int& o, unsigned int& s)
{
    if constexpr(SEPARATE_STRIDE)
    {
        o = blockIdx.x;
        if constexpr(LOCAL_SIZE_Y > 1)
        {
            s = threadIdx.y;
        }
        else
        {
            s = blockIdx.y;
        }
        gid = o * STRIDE + s;
    }
    else
    {
        gid = blockIdx.x;
        o   = blockIdx.x / STRIDE;
        s   = blockIdx.x % STRIDE;
    }
}

template <typename T>
__forceinline__ __device__ void layernormfwd(const T* __restrict__ x,
                                             const T* __restrict__ weight,
                                             const T* __restrict__ bias,
                                             T* __restrict__ y,
                                             T* __restrict__ mean,
                                             T* __restrict__ rstd,
                                             const float epsilon)
{
    /*
     * Each group works on a single channel.
     * Example)
     * x dim = {N, C, L}, normalized shape = {C, L}, layout = NCHW or NHWC
     * outer_size = N, inner_size = C * L, stride = 1
     *
     * Example2)
     * x dim = {N, C, L}, normalized shape = {L}, layout = NCHW
     * outer_size = N * C, inner_size = L, stride = 1
     *
     * Example3)
     * x dim = {N, C, L}, normalized shape = {L}, layout = NHWC
     * outer_size = N, inner_size = L, stride = C
     *
     * => gws = {outer_size * LOCAL_SIZE_X, stride}, lws = {LOCAL_SIZE_X, stride}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    FLOAT_ACCUM pmean = CVT_FP32_2ACCUM(0.0f);
    FLOAT_ACCUM pvar  = CVT_FP32_2ACCUM(0.0f);

    // reduce sum for mean and var
    unsigned int gid, o, s;
    get_indices(gid, o, s);
    const unsigned int offset = o * INNER_SIZE * STRIDE + s;
    const unsigned int lid    = threadIdx.x;
    if constexpr(VECTORIZED)
    {
        unsigned int i = lid * load_factor<T>;
        auto tmpx      = load(i, offset, x);
        i += LOCAL_SIZE_X * load_factor<T>;
        for(; i < INNER_SIZE; i += LOCAL_SIZE_X * load_factor<T>)
        {
            auto tmp = load(i, offset, x);
            __builtin_amdgcn_sched_barrier(1);
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM px = CVT_FLOAT2ACCUM(tmpx.data[k]);
                pmean += px;
                pvar += px * px;
            }
            __builtin_amdgcn_sched_barrier(1);
            tmpx = tmp;
        }
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            FLOAT_ACCUM px = CVT_FLOAT2ACCUM(tmpx.data[k]);
            pmean += px;
            pvar += px * px;
        }
    }
    else
    {
        for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE_X)
        {
            unsigned int idx = i * STRIDE + offset;

            FLOAT_ACCUM tmp = CVT_FLOAT2ACCUM(x[idx]);
            pmean += tmp;
            pvar += tmp * tmp;
        }
    }

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE_X];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE_X];
    FLOAT_ACCUM prstd;
    if constexpr(LOCAL_SIZE_X > 1)
    {
        if constexpr(LOCAL_SIZE_Y > 1)
        {
            for(unsigned int j = 0; j < STRIDE; ++j)
            {
                if(j == s)
                {
                    ltmp1[lid] = pmean;
                    ltmp2[lid] = pvar;
                }
                __syncthreads();
                for(unsigned int k = LOCAL_SIZE_X >> 1; k > 0; k >>= 1)
                {
                    if(j == s && lid < k)
                    {
                        ltmp1[lid] += ltmp1[lid + k];
                        ltmp2[lid] += ltmp2[lid + k];
                    }
                    __syncthreads();
                }
                if(j == s)
                {
                    pmean = ltmp1[0];
                    pvar  = ltmp2[0];
                }
                __syncthreads();
            }
        }
        else
        {
            ltmp1[lid] = pmean;
            ltmp2[lid] = pvar;
            __syncthreads();
            for(unsigned int k = LOCAL_SIZE_X >> 1; k > 0; k >>= 1)
            {
                if(lid < k)
                {
                    ltmp1[lid] += ltmp1[lid + k];
                    ltmp2[lid] += ltmp2[lid + k];
                }
                __syncthreads();
            }
            pmean = ltmp1[0];
            pvar  = ltmp2[0];
        }
    }
    pmean = pmean * CVT_FP32_2ACCUM(1.0f / INNER_SIZE);
    pvar  = pvar * CVT_FP32_2ACCUM(1.0f / INNER_SIZE) - pmean * pmean;
    prstd = rsqrtf(pvar + CVT_FP32_2ACCUM(epsilon));
    if(lid == 0)
    {
        if(mean)
        {
            mean[gid] = CVT_ACCUM2FLOAT(pmean);
        }
        if(rstd)
        {
            rstd[gid] = CVT_ACCUM2FLOAT(prstd);
        }
    }

    // forward calculation
    if constexpr(VECTORIZED)
    {
        unsigned int i = lid * load_factor<T>;
        auto tmpx      = load(i, offset, x);
        auto tmpweight = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
            i, weight, CVT_FP32_2FLOAT(1.0f));
        auto tmpbias = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
            i, bias, CVT_FP32_2FLOAT(0.0f));
        vec_t<T> tmpy{{}};
        i += LOCAL_SIZE_X * load_factor<T>;
        for(; i < INNER_SIZE; i += LOCAL_SIZE_X * load_factor<T>)
        {
            auto tmp1 = load(i, offset, x);
            auto tmp2 = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
                i, weight, CVT_FP32_2FLOAT(1.0f));
            auto tmp3 = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
                i, bias, CVT_FP32_2FLOAT(0.0f));
            vec_t<T> tmp4{{}};
            __builtin_amdgcn_sched_barrier(1);
#pragma unroll
            for(unsigned int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);
                FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
                FLOAT_ACCUM pbias   = CVT_FLOAT2ACCUM(tmpbias.data[k]);

                tmp4.data[k] = CVT_ACCUM2FLOAT((px - pmean) * prstd * pweight + pbias);
            }
            __builtin_amdgcn_sched_barrier(1);
            tmpx      = tmp1;
            tmpweight = tmp2;
            tmpbias   = tmp3;
            tmpy      = tmp4;
            store(i - LOCAL_SIZE_X * load_factor<T>, offset, y, tmpy);
        }
        tmpy = {{}};
#pragma unroll
        for(unsigned int k = 0; k < load_factor<T>; ++k)
        {
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM pbias   = CVT_FLOAT2ACCUM(tmpbias.data[k]);

            tmpy.data[k] = CVT_ACCUM2FLOAT((px - pmean) * prstd * pweight + pbias);
        }
        store(i - LOCAL_SIZE_X * load_factor<T>, offset, y, tmpy);
    }
    else
    {
        for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE_X)
        {
            unsigned int idx = i * STRIDE + offset;

            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(x[idx]);
            FLOAT_ACCUM pweight = (MODE == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(1.0f)
                                                                      : CVT_FLOAT2ACCUM(weight[i]);
            FLOAT_ACCUM pbias   = (MODE == MIOPEN_ELEMENTWISE_AFFINE) ? CVT_FP32_2ACCUM(0.0f)
                                                                      : CVT_FLOAT2ACCUM(bias[i]);

            y[idx] = CVT_ACCUM2FLOAT((px - pmean) * prstd * pweight + pbias);
        }
    }
}

template <typename T>
__forceinline__ __device__ void layernormbwd(const T* __restrict__ dy,
                                             const T* __restrict__ x,
                                             const T* __restrict__ weight,
                                             const T* __restrict__ mean,
                                             const T* __restrict__ rstd,
                                             T* __restrict__ dx)
{
    FLOAT_ACCUM sum_dy_weight   = CVT_FP32_2ACCUM(0.0f);
    FLOAT_ACCUM sum_dy_weight_x = CVT_FP32_2ACCUM(0.0f);

    // Reduce sums
    unsigned int gid, o, s;
    get_indices(gid, o, s);
    const unsigned int offset = o * INNER_SIZE * STRIDE + s;
    const unsigned int lid    = threadIdx.x;
    if constexpr(VECTORIZED)
    {
        unsigned int i = lid * load_factor<T>;
        auto tmpdy     = load(i, offset, dy);
        auto tmpweight = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
            i, weight, CVT_FP32_2FLOAT(1.0f));
        auto tmpx = load(i, offset, x);
        i += LOCAL_SIZE_X * load_factor<T>;
        for(; i < INNER_SIZE; i += LOCAL_SIZE_X * load_factor<T>)
        {
            auto tmp1 = load(i, offset, dy);
            auto tmp2 = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
                i, weight, CVT_FP32_2FLOAT(1.0f));
            auto tmp3 = load(i, offset, x);
            __builtin_amdgcn_sched_barrier(1);
#pragma unroll
            for(unsigned int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
                FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

                sum_dy_weight += pdy * pweight;
                sum_dy_weight_x += pdy * pweight * px;
            }
            __builtin_amdgcn_sched_barrier(1);
            tmpdy     = tmp1;
            tmpweight = tmp2;
            tmpx      = tmp3;
        }
#pragma unroll
        for(unsigned int k = 0; k < load_factor<T>; ++k)
        {
            FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

            sum_dy_weight += pdy * pweight;
            sum_dy_weight_x += pdy * pweight * px;
        }
    }
    else
    {
        for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE_X)
        {
            unsigned int idx = i * STRIDE + offset;

            FLOAT_ACCUM px         = CVT_FLOAT2ACCUM(x[idx]);
            FLOAT_ACCUM pdy_weight = CVT_FLOAT2ACCUM(dy[idx]) * ((MODE == MIOPEN_ELEMENTWISE_AFFINE)
                                                                     ? CVT_FP32_2ACCUM(1.0f)
                                                                     : CVT_FLOAT2ACCUM(weight[i]));

            sum_dy_weight += pdy_weight;
            sum_dy_weight_x += pdy_weight * px;
        }
    }

    __shared__ FLOAT_ACCUM ltmp1[LOCAL_SIZE_X];
    __shared__ FLOAT_ACCUM ltmp2[LOCAL_SIZE_X];
    if constexpr(LOCAL_SIZE_X > 1)
    {
        if constexpr(LOCAL_SIZE_Y > 1)
        {
            for(unsigned int j = 0; j < STRIDE; ++j)
            {
                if(j == s)
                {
                    ltmp1[lid] = sum_dy_weight;
                    ltmp2[lid] = sum_dy_weight_x;
                }
                __syncthreads();
                for(unsigned int k = LOCAL_SIZE_X >> 1; k > 0; k >>= 1)
                {
                    if(j == s && lid < k)
                    {
                        ltmp1[lid] += ltmp1[lid + k];
                        ltmp2[lid] += ltmp2[lid + k];
                    }
                    __syncthreads();
                }
                if(j == s)
                {
                    sum_dy_weight   = ltmp1[0];
                    sum_dy_weight_x = ltmp2[0];
                }
                __syncthreads();
            }
        }
        else
        {
            ltmp1[lid] = sum_dy_weight;
            ltmp2[lid] = sum_dy_weight_x;
            __syncthreads();
            for(unsigned int k = LOCAL_SIZE_X >> 1; k > 0; k >>= 1)
            {
                if(lid < k)
                {
                    ltmp1[lid] += ltmp1[lid + k];
                    ltmp2[lid] += ltmp2[lid + k];
                }
                __syncthreads();
            }
            sum_dy_weight   = ltmp1[0];
            sum_dy_weight_x = ltmp2[0];
        }
    }

    constexpr FLOAT_ACCUM scale = CVT_FP32_2ACCUM(1.0f / INNER_SIZE);
    FLOAT_ACCUM prstd           = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM pmean           = CVT_FLOAT2ACCUM(mean[gid]);
    FLOAT_ACCUM a = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
    FLOAT_ACCUM b = prstd * sum_dy_weight * scale - a * pmean;

    // Backward calculation
    if constexpr(VECTORIZED)
    {
        unsigned int i = lid * load_factor<T>;
        auto tmpdy     = load(i, offset, dy);
        auto tmpweight = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
            i, weight, CVT_FP32_2FLOAT(1.0f));
        auto tmpx = load(i, offset, x);
        vec_t<T> tmpdx{{}};
        i += LOCAL_SIZE_X * load_factor<T>;
        for(; i < INNER_SIZE; i += LOCAL_SIZE_X * load_factor<T>)
        {
            auto tmp1 = load(i, offset, dy);
            auto tmp2 = load_contiguous<T, INNER_SIZE, MODE == MIOPEN_ELEMENTWISE_AFFINE>(
                i, weight, CVT_FP32_2FLOAT(1.0f));
            auto tmp3 = load(i, offset, x);
            vec_t<T> tmp4{{}};
            __builtin_amdgcn_sched_barrier(1);
#pragma unroll
            for(unsigned int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
                FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

                tmp4.data[k] = CVT_ACCUM2FLOAT(prstd * pdy * pweight - a * px - b);
            }
            __builtin_amdgcn_sched_barrier(1);
            tmpdy     = tmp1;
            tmpweight = tmp2;
            tmpx      = tmp3;
            tmpdx     = tmp4;
            store(i - LOCAL_SIZE_X * load_factor<T>, offset, dx, tmpdx);
        }
        tmpdx = {{}};
#pragma unroll
        for(unsigned int k = 0; k < load_factor<T>; ++k)
        {
            FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

            tmpdx.data[k] = CVT_ACCUM2FLOAT(prstd * pdy * pweight - a * px - b);
        }
        store(i - LOCAL_SIZE_X * load_factor<T>, offset, dx, tmpdx);
    }
    else
    {
        for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE_X)
        {
            unsigned int idx = i * STRIDE + offset;

            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(x[idx]);
            FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(dy[idx]);
            FLOAT_ACCUM pweight = MODE == MIOPEN_ELEMENTWISE_AFFINE ? CVT_FP32_2ACCUM(1.0f)
                                                                    : CVT_FLOAT2ACCUM(weight[i]);

            dx[idx] = CVT_ACCUM2FLOAT(prstd * pdy * pweight - a * px - b);
        }
    }
}

template <typename T>
__forceinline__ __device__ void layernormbwdweightbias(const T* __restrict__ dy,
                                                       const T* __restrict__ x,
                                                       const T* __restrict__ mean,
                                                       const T* __restrict__ rstd,
                                                       T* __restrict__ dw,
                                                       T* __restrict__ db)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE_X;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM sum_db = CVT_FP32_2ACCUM(0.0f);

        // Backward calculation
        if constexpr(VECTORIZED)
        {
            for(unsigned int o = 0; o < OUTER_SIZE; ++o)
            {
                unsigned int s = 0;
                auto tmpdy     = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + gid * STRIDE, dy);
                auto tmpx      = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + gid * STRIDE, x);
                auto tmprstd   = load<T, STRIDE, 1>(s, o * STRIDE, rstd);
                auto tmpmean   = load<T, STRIDE, 1>(s, o * STRIDE, mean);
                s += load_factor<T>;
                for(; s < STRIDE; s += load_factor<T>)
                {
                    auto tmp1 = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + gid * STRIDE, dy);
                    auto tmp2 = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + gid * STRIDE, x);
                    auto tmp3 = load<T, STRIDE, 1>(s, o * STRIDE, rstd);
                    auto tmp4 = load<T, STRIDE, 1>(s, o * STRIDE, mean);
                    __builtin_amdgcn_sched_barrier(1);
#pragma unroll
                    for(unsigned int k = 0; k < load_factor<T>; ++k)
                    {
                        FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                        FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(tmpx.data[k]);
                        FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(tmprstd.data[k]);
                        FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(tmpmean.data[k]);

                        sum_dw += prstd * pdy * (px - pmean);
                        sum_db += pdy;
                    }
                    __builtin_amdgcn_sched_barrier(1);
                    tmpdy   = tmp1;
                    tmpx    = tmp2;
                    tmprstd = tmp3;
                    tmpmean = tmp4;
                }
#pragma unroll
                for(unsigned int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                    FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(tmpx.data[k]);
                    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(tmprstd.data[k]);
                    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(tmpmean.data[k]);

                    sum_dw += prstd * pdy * (px - pmean);
                    sum_db += pdy;
                }
            }
        }
        else
        {
            for(unsigned int o = 0; o < OUTER_SIZE; ++o)
            {
                for(unsigned int s = 0; s < STRIDE; ++s)
                {
                    unsigned int idx = o * INNER_SIZE * STRIDE + gid * STRIDE + s;

                    FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(x[idx]);
                    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[o * STRIDE + s]);
                    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[o * STRIDE + s]);
                    FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[idx]);

                    sum_dw += prstd * pdy * (px - pmean);
                    sum_db += pdy;
                }
            }
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

template <typename T>
__forceinline__ __device__ void layernormbwdweightbiasparallel(const T* __restrict__ dy,
                                                               const T* __restrict__ x,
                                                               const T* __restrict__ mean,
                                                               const T* __restrict__ rstd,
                                                               T* __restrict__ workspace)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE_X;

    if(gid >= INNER_SIZE * PARALLEL_SIZE)
        return;

    const unsigned int pid   = gid / INNER_SIZE;
    const unsigned int s_lid = (gid % INNER_SIZE) * STRIDE;

    FLOAT_ACCUM sum_dw = CVT_FP32_2ACCUM(0.0f);
    FLOAT_ACCUM sum_db = CVT_FP32_2ACCUM(0.0f);

    // Backward calculation
    if constexpr(VECTORIZED)
    {
        for(unsigned int o = pid; o < OUTER_SIZE; o += PARALLEL_SIZE)
        {
            unsigned int s = 0;
            auto tmpdy     = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + s_lid, dy);
            auto tmpx      = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + s_lid, x);
            auto tmprstd   = load<T, STRIDE, 1>(s, o * STRIDE, rstd);
            auto tmpmean   = load<T, STRIDE, 1>(s, o * STRIDE, mean);
            s += load_factor<T>;
            for(; s < STRIDE; s += load_factor<T>)
            {
                auto tmp1 = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + s_lid, dy);
                auto tmp2 = load<T, STRIDE, 1>(s, o * INNER_SIZE * STRIDE + s_lid, x);
                auto tmp3 = load<T, STRIDE, 1>(s, o * STRIDE, rstd);
                auto tmp4 = load<T, STRIDE, 1>(s, o * STRIDE, mean);
                __builtin_amdgcn_sched_barrier(1);
#pragma unroll
                for(unsigned int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                    FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(tmpx.data[k]);
                    FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(tmprstd.data[k]);
                    FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(tmpmean.data[k]);

                    sum_dw += prstd * pdy * (px - pmean);
                    sum_db += pdy;
                }
                __builtin_amdgcn_sched_barrier(1);
                tmpdy   = tmp1;
                tmpx    = tmp2;
                tmprstd = tmp3;
                tmpmean = tmp4;
            }
#pragma unroll
            for(unsigned int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(tmpx.data[k]);
                FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(tmprstd.data[k]);
                FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(tmpmean.data[k]);

                sum_dw += pdy * prstd * (px - pmean);
                sum_db += pdy;
            }
        }
    }
    else
    {
        for(unsigned int i = pid; i < OUTER_SIZE * STRIDE; i += PARALLEL_SIZE)
        {
            unsigned int o   = i / STRIDE;
            unsigned int s   = i % STRIDE;
            unsigned int idx = o * INNER_SIZE * STRIDE + s_lid + s;

            FLOAT_ACCUM px    = CVT_FLOAT2ACCUM(x[idx]);
            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[idx]);

            sum_dw += pdy * prstd * (px - pmean);
            sum_db += pdy;
        }
    }

    workspace[gid]                              = CVT_ACCUM2FLOAT(sum_dw);
    workspace[gid + PARALLEL_SIZE * INNER_SIZE] = CVT_ACCUM2FLOAT(sum_db);
}

template <typename T>
__forceinline__ __device__ void
layernormbwdreducesum(const T* __restrict__ workspace, T* __restrict__ dw, T* __restrict__ db)
{
    const unsigned int gid = threadIdx.x + blockIdx.x * LOCAL_SIZE_X;

    if(gid >= INNER_SIZE)
        return;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM sum_db = CVT_FP32_2ACCUM(0.0f);

        for(unsigned int i = 0; i < PARALLEL_SIZE; ++i)
        {
            unsigned int idx = i * INNER_SIZE + gid;
            sum_dw += CVT_FLOAT2ACCUM(workspace[idx]);
            sum_db += CVT_FLOAT2ACCUM(workspace[idx + PARALLEL_SIZE * INNER_SIZE]);
        }

        if(dw)
        {
            dw[gid] = CVT_ACCUM2FLOAT(sum_dw);
        }
        if(db)
        {
            db[gid] = CVT_ACCUM2FLOAT(sum_db);
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void LayernormFwd(
    const DATA_TYPE* __restrict__ x,
    const DATA_TYPE* __restrict__ weight,
    const DATA_TYPE* __restrict__ bias,
    DATA_TYPE* __restrict__ y,
    DATA_TYPE* __restrict__ mean,
    DATA_TYPE* __restrict__ rstd,
    const float epsilon)
{
    layernormfwd<DATA_TYPE>(x, weight, bias, y, mean, rstd, epsilon);
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void LayernormBwd(
    const DATA_TYPE* __restrict__ dy,
    const DATA_TYPE* __restrict__ x,
    const DATA_TYPE* __restrict__ weight,
    const DATA_TYPE* __restrict__ mean,
    const DATA_TYPE* __restrict__ rstd,
    DATA_TYPE* __restrict__ dx)
{
    layernormbwd<DATA_TYPE>(dy, x, weight, mean, rstd, dx);
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void LayernormBwdWeightBias(
    const DATA_TYPE* __restrict__ dy,
    const DATA_TYPE* __restrict__ x,
    const DATA_TYPE* __restrict__ mean,
    const DATA_TYPE* __restrict__ rstd,
    DATA_TYPE* __restrict__ dw,
    DATA_TYPE* __restrict__ db)
{
    layernormbwdweightbias<DATA_TYPE>(dy, x, mean, rstd, dw, db);
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void LayernormBwdWeightBiasParallel(
    const DATA_TYPE* __restrict__ dy,
    const DATA_TYPE* __restrict__ x,
    const DATA_TYPE* __restrict__ mean,
    const DATA_TYPE* __restrict__ rstd,
    DATA_TYPE* __restrict__ workspace)
{
    layernormbwdweightbiasparallel<DATA_TYPE>(dy, x, mean, rstd, workspace);
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE_X* LOCAL_SIZE_Y) void LayernormBwdReduceSum(
    const DATA_TYPE* __restrict__ workspace, DATA_TYPE* __restrict__ dw, DATA_TYPE* __restrict__ db)
{
    layernormbwdreducesum<DATA_TYPE>(workspace, dw, db);
}
