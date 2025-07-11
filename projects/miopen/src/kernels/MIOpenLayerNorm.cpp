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
#include "miopen_cstdint.hpp"

#include "float_types.h"

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
    constexpr static int value = log2_floor_v<N> + ((1 << log2_floor_v<N>) == N ? 0 : 1);
};
template <int N>
constexpr static int log2_ceil_v = log2_ceil<N>::value;

constexpr static int ADJUSTED_LOCAL_SIZE =
    STRIDE <= LOCAL_SIZE ? LOCAL_SIZE / (1 << log2_ceil_v<STRIDE>) : LOCAL_SIZE;

using load_t = int4;

template <typename T, size_t n>
struct array
{
    T data[n];
};

template <typename T, bool contiguous = false, bool use_default = false>
struct loader
{
    static const auto load_factor = sizeof(load_t) / sizeof(T);
    using vec_t                   = array<T, load_factor>;
    __device__ static auto
    load(uint64_t i, const T* __restrict__ src, const FLOAT default_value = CVT_FP32_2FLOAT(0.0f))
    {
        const uint64_t o       = blockIdx.x;
        const unsigned int s   = STRIDE <= LOCAL_SIZE ? threadIdx.y : blockIdx.y;
        const unsigned int lid = threadIdx.x;
        const auto j           = i + lid * load_factor;
        if(!use_default && (STRIDE == 1 || contiguous) && j + load_factor < INNER_SIZE)
        {
            const size_t idx  = o * INNER_SIZE * STRIDE + j * STRIDE + s;
            const auto value  = *reinterpret_cast<const load_t*>(&src[contiguous ? j : idx]);
            const auto values = *reinterpret_cast<const vec_t*>(&value);

            return values;
        }
        else
        {
            vec_t values = {{}};
#pragma unroll
            for(int k = 0; k < load_factor; ++k)
            {
                if(j + k < INNER_SIZE)
                {
                    size_t idx = o * INNER_SIZE * STRIDE + (j + k) * STRIDE + s;

                    values.data[k] =
                        use_default ? static_cast<T>(default_value) : src[contiguous ? j + k : idx];
                }
            }
            return values;
        }
    }
};

template <typename TI, typename TO>
struct storer
{
    __device__ static auto store(uint64_t i, TO* __restrict__ dst, loader<TI>::vec_t& data)
    {
        const uint64_t o       = blockIdx.x;
        const unsigned int s   = STRIDE <= LOCAL_SIZE ? threadIdx.y : blockIdx.y;
        const unsigned int lid = threadIdx.x;
        const auto j           = i + lid * loader<TI>::load_factor;
        if(STRIDE == 1 && j + loader<TI>::load_factor < INNER_SIZE)
        {
            const size_t idx = o * INNER_SIZE * STRIDE + j * STRIDE + s;

            const auto value                      = *reinterpret_cast<load_t*>(&data);
            *reinterpret_cast<load_t*>(&dst[idx]) = value;
        }
        else
        {
#pragma unroll
            for(size_t k = 0; k < loader<TI>::load_factor; ++k)
            {
                if(j + k < INNER_SIZE)
                {
                    size_t idx = o * INNER_SIZE * STRIDE + (j + k) * STRIDE + s;

                    dst[idx] = data.data[k];
                }
            }
        }
    };
};

template <typename TI, typename TO>
__device__ void layernormfwd(const TI* __restrict__ x,
                             const TI* __restrict__ weight,
                             const TI* __restrict__ bias,
                             TO* __restrict__ y,
                             TO* __restrict__ mean,
                             TO* __restrict__ rstd,
                             const float eps)
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
     * => gws = {outer_size * ADJUSTED_LOCAL_SIZE, stride}, lws = {ADJUSTED_LOCAL_SIZE, stride}
     */

    /*
     * Reduction to calculate mean and rstd
     */

    FLOAT_ACCUM pmean = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM pvar  = static_cast<FLOAT_ACCUM>(0);

    // reduce sum for mean and var
    uint64_t i = 0;
    auto tmpx  = loader<TI>::load(i, x);
    i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor;
    for(; i < INNER_SIZE; i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor)
    {
        auto tmp = loader<TI>::load(i, x);
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(int k = 0; k < loader<TI>::load_factor; ++k)
        {
            FLOAT_ACCUM px = CVT_FLOAT2ACCUM(tmpx.data[k]);
            pmean += px;
            auto tmpvar = px * px;
            asm volatile("" : "+v"(tmpvar)); // Due to compiler bug with FMA
            pvar += tmpvar;
            // pvar += px * px;
        }
        __builtin_amdgcn_sched_barrier(0);
        tmpx = tmp;
    }
#pragma unroll
    for(int k = 0; k < loader<TI>::load_factor; ++k)
    {
        FLOAT_ACCUM px = CVT_FLOAT2ACCUM(tmpx.data[k]);
        pmean += px;
        auto tmpvar = px * px;
        asm volatile("" : "+v"(tmpvar)); // Due to compiler bug with FMA
        pvar += tmpvar;
        // pvar += px * px;
    }

    __shared__ FLOAT_ACCUM ltmp1[ADJUSTED_LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[ADJUSTED_LOCAL_SIZE];
    const int lid = threadIdx.x;
    const int s   = STRIDE <= LOCAL_SIZE ? threadIdx.y : blockIdx.y;
    FLOAT_ACCUM prstd;
    for(uint64_t j = 0; j < STRIDE; ++j)
    {
        if(j == s)
        {
            ltmp1[lid] = pmean;
            ltmp2[lid] = pvar;
        }
        __syncthreads();
        for(uint32_t k = ADJUSTED_LOCAL_SIZE >> 1; k > 0; k >>= 1)
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
            pmean = ltmp1[0] / INNER_SIZE;
            pvar  = ltmp2[0] / INNER_SIZE - pmean * pmean;
            prstd = rsqrt(pvar + FLOAT_ACCUM(eps));

            if(lid == 0)
            {
                const uint64_t gid = blockIdx.x * STRIDE + s;
                if(mean)
                    mean[gid] = CVT_ACCUM2FLOAT(pmean);
                if(rstd)
                    rstd[gid] = CVT_ACCUM2FLOAT(prstd);
            }
        }
        __syncthreads();
    }

    // forward calculation
    i    = 0;
    tmpx = loader<TI>::load(i, x);
    auto tmpweight =
        loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(i, weight, CVT_FP32_2FLOAT(1.0f));
    auto tmpbias =
        loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(i, bias, CVT_FP32_2FLOAT(0.0f));
    typename loader<TI>::vec_t tmpy = {{}};
    i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor;
    for(; i < INNER_SIZE; i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor)
    {
        auto tmp1 = loader<TI>::load(i, x);
        auto tmp2 = loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(i, weight, 1);
        auto tmp3 = loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(i, bias, 0);
        typename loader<TI>::vec_t tmp4 = {{}};
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(size_t k = 0; k < loader<TI>::load_factor; ++k)
        {
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM pbias   = CVT_FLOAT2ACCUM(tmpbias.data[k]);

            tmp4.data[k] = CVT_ACCUM2FLOAT((px - pmean) * prstd * pweight + pbias);
        }
        __builtin_amdgcn_sched_barrier(0);
        tmpx      = tmp1;
        tmpweight = tmp2;
        tmpbias   = tmp3;
        tmpy      = tmp4;
        storer<TI, TO>::store(i - ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor, y, tmpy);
    }
    tmpy = {{}};
#pragma unroll
    for(size_t k = 0; k < loader<TI>::load_factor; ++k)
    {
        FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);
        FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
        FLOAT_ACCUM pbias   = CVT_FLOAT2ACCUM(tmpbias.data[k]);

        tmpy.data[k] = CVT_ACCUM2FLOAT((px - pmean) * prstd * pweight + pbias);
    }
    storer<TI, TO>::store(i - ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor, y, tmpy);
}

template <typename TI, typename TO>
__device__ void layernormbwd(const TI* __restrict__ dy,
                             const TI* __restrict__ x,
                             const TI* __restrict__ weight,
                             const TI* __restrict__ mean,
                             const TI* __restrict__ rstd,
                             TO* __restrict__ dx)
{
    FLOAT_ACCUM sum_dy_weight   = static_cast<FLOAT_ACCUM>(0);
    FLOAT_ACCUM sum_dy_weight_x = static_cast<FLOAT_ACCUM>(0);

    // Reduce sums
    if(dy)
    {
        uint64_t i     = 0;
        auto tmpdy     = loader<TI>::load(i, dy);
        auto tmpweight = loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(
            i, weight, CVT_FP32_2FLOAT(1.0f));
        auto tmpx = loader<TI>::load(i, x);
        i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor;
        for(; i < INNER_SIZE; i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor)
        {
            auto tmp1 = loader<TI>::load(i, dy);
            auto tmp2 = loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(
                i, weight, CVT_FP32_2FLOAT(1.0f));
            auto tmp3 = loader<TI>::load(i, x);
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for(size_t k = 0; k < loader<TI>::load_factor; ++k)
            {
                FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
                FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

                sum_dy_weight += pdy * pweight;
                sum_dy_weight_x += pdy * pweight * px;
            }
            __builtin_amdgcn_sched_barrier(0);
            tmpdy     = tmp1;
            tmpweight = tmp2;
            tmpx      = tmp3;
        }
#pragma unroll
        for(size_t k = 0; k < loader<TI>::load_factor; ++k)
        {
            FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

            sum_dy_weight += pdy * pweight;
            sum_dy_weight_x += pdy * pweight * px;
        }
    }

    __shared__ FLOAT_ACCUM ltmp1[ADJUSTED_LOCAL_SIZE];
    __shared__ FLOAT_ACCUM ltmp2[ADJUSTED_LOCAL_SIZE];
    const uint64_t lid = threadIdx.x;
    const uint64_t s   = STRIDE <= LOCAL_SIZE ? threadIdx.y : blockIdx.y;
    for(int i = 0; i < STRIDE; ++i)
    {
        if(i == s)
        {
            ltmp1[lid] = sum_dy_weight;
            ltmp2[lid] = sum_dy_weight_x;
        }
        __syncthreads();
        for(uint32_t j = ADJUSTED_LOCAL_SIZE >> 1; j > 0; j >>= 1)
        {
            if(i == s && lid < j)
            {
                ltmp1[lid] += ltmp1[lid + j];
                ltmp2[lid] += ltmp2[lid + j];
            }
            __syncthreads();
        }
        if(i == s)
        {
            sum_dy_weight   = ltmp1[0];
            sum_dy_weight_x = ltmp2[0];
        }
        __syncthreads();
    }

    const uint64_t gid = blockIdx.x * STRIDE + s;
    FLOAT_ACCUM scale  = 1.0f / INNER_SIZE;
    FLOAT_ACCUM prstd  = CVT_FLOAT2ACCUM(rstd[gid]);
    FLOAT_ACCUM pmean  = CVT_FLOAT2ACCUM(mean[gid]);
    FLOAT_ACCUM a      = prstd * prstd * prstd * scale * (sum_dy_weight_x - sum_dy_weight * pmean);
    FLOAT_ACCUM b      = prstd * sum_dy_weight * scale - a * pmean;

    // Backward calculation
    uint64_t i = 0;
    auto tmpdy = loader<TI>::load(i, dy);
    auto tmpweight =
        loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(i, weight, CVT_FP32_2FLOAT(1.0f));
    auto tmpx                        = loader<TI>::load(i, x);
    typename loader<TI>::vec_t tmpdx = {{}};
    i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor;
    for(; i < INNER_SIZE; i += ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor)
    {
        auto tmp1 = loader<TI>::load(i, dy);
        auto tmp2 = loader<TI, true, MODE == MIOPEN_ELEMENTWISE_AFFINE>::load(
            i, weight, CVT_FP32_2FLOAT(1.0f));
        auto tmp3                       = loader<TI>::load(i, x);
        typename loader<TI>::vec_t tmp4 = {{}};
        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
        for(size_t k = 0; k < loader<TI>::load_factor; ++k)
        {
            FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
            FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
            FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

            tmp4.data[k] = CVT_ACCUM2FLOAT(prstd * pdy * pweight - a * px - b);
        }
        __builtin_amdgcn_sched_barrier(0);
        tmpdy     = tmp1;
        tmpweight = tmp2;
        tmpx      = tmp3;
        tmpdx     = tmp4;
        storer<TI, TO>::store(i - ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor, dx, tmpdx);
    }
    tmpdx = {{}};
#pragma unroll
    for(size_t k = 0; k < loader<TI>::load_factor; ++k)
    {
        FLOAT_ACCUM pdy     = CVT_FLOAT2ACCUM(tmpdy.data[k]);
        FLOAT_ACCUM pweight = CVT_FLOAT2ACCUM(tmpweight.data[k]);
        FLOAT_ACCUM px      = CVT_FLOAT2ACCUM(tmpx.data[k]);

        tmpdx.data[k] = CVT_ACCUM2FLOAT(prstd * pdy * pweight - a * px - b);
    }
    storer<TI, TO>::store(i - ADJUSTED_LOCAL_SIZE * loader<TI>::load_factor, dx, tmpdx);
}

template <typename TI, typename TO>
__device__ void layernormbwdweightbias(const TI* __restrict__ dy,
                                       const TI* __restrict__ x,
                                       const TI* __restrict__ mean,
                                       const TI* __restrict__ rstd,
                                       TO* __restrict__ dw,
                                       TO* __restrict__ db)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        // Backward calculation
        for(uint64_t o = 0; o < OUTER_SIZE; ++o)
        {
            for(uint64_t s = 0; s < STRIDE; ++s)
            {
                uint64_t input_idx = o * INNER_SIZE * STRIDE + gid * STRIDE + s;

                FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[o * STRIDE + s]);
                FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[o * STRIDE + s]);
                FLOAT_ACCUM pdy   = dy ? CVT_FLOAT2ACCUM(dy[input_idx]) : 0;

                sum_dw += prstd * pdy * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
                sum_db += pdy;
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

template <typename TI, typename TO>
__device__ void layernormbwdweightbiasparallel(const TI* __restrict__ dy,
                                               const TI* __restrict__ x,
                                               const TI* __restrict__ mean,
                                               const TI* __restrict__ rstd,
                                               TO* __restrict__ workspace)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(gid >= INNER_SIZE * PARALLEL_SIZE)
        return;

    uint64_t pid   = gid / INNER_SIZE;
    uint64_t s_lid = (gid % INNER_SIZE) * STRIDE;

    FLOAT_ACCUM sum_dw = 0;
    FLOAT_ACCUM sum_db = 0;

    if(dy)
    {
        // Backward calculation
        for(uint64_t i = pid; i < OUTER_SIZE * STRIDE; i += PARALLEL_SIZE)
        {
            uint64_t o         = i / STRIDE;
            uint64_t s         = i % STRIDE;
            uint64_t input_idx = o * INNER_SIZE * STRIDE + s_lid + s;

            FLOAT_ACCUM prstd = CVT_FLOAT2ACCUM(rstd[i]);
            FLOAT_ACCUM pmean = CVT_FLOAT2ACCUM(mean[i]);
            FLOAT_ACCUM pdy   = CVT_FLOAT2ACCUM(dy[input_idx]);

            sum_dw += pdy * prstd * (CVT_FLOAT2ACCUM(x[input_idx]) - pmean);
            sum_db += pdy;
        }
    }

    workspace[gid]                              = CVT_ACCUM2FLOAT(sum_dw);
    workspace[gid + PARALLEL_SIZE * INNER_SIZE] = CVT_ACCUM2FLOAT(sum_db);
}

template <typename TI, typename TO>
__device__ void
layernormbwdreducesum(const TI* __restrict__ workspace, TO* __restrict__ dw, TO* __restrict__ db)
{
    const uint64_t gid = threadIdx.x + blockIdx.x * LOCAL_SIZE;

    if(gid >= INNER_SIZE)
        return;

    if(dw || db)
    {
        FLOAT_ACCUM sum_dw = 0;
        FLOAT_ACCUM sum_db = 0;

        for(uint64_t i = 0; i < PARALLEL_SIZE; ++i)
        {
            uint64_t input_idx = i * INNER_SIZE + gid;
            sum_dw += CVT_FLOAT2ACCUM(workspace[input_idx]);
            sum_db += CVT_FLOAT2ACCUM(workspace[input_idx + PARALLEL_SIZE * INNER_SIZE]);
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

extern "C" __global__ void LayernormFwd(const INPUT_TYPE* __restrict__ x,
                                        const INPUT_TYPE* __restrict__ weight,
                                        const INPUT_TYPE* __restrict__ bias,
                                        OUTPUT_TYPE* __restrict__ y,
                                        OUTPUT_TYPE* __restrict__ mean,
                                        OUTPUT_TYPE* __restrict__ rstd,
                                        const float eps)
{
    // instantiate the kernel
    layernormfwd<INPUT_TYPE, OUTPUT_TYPE>(x, weight, bias, y, mean, rstd, eps);
}

extern "C" __global__ void LayernormBwd(const INPUT_TYPE* __restrict__ dy,
                                        const INPUT_TYPE* __restrict__ x,
                                        const INPUT_TYPE* __restrict__ weight,
                                        const INPUT_TYPE* __restrict__ mean,
                                        const INPUT_TYPE* __restrict__ rstd,
                                        OUTPUT_TYPE* __restrict__ dx)
{
    // instantiate the kernel
    layernormbwd<INPUT_TYPE, OUTPUT_TYPE>(dy, x, weight, mean, rstd, dx);
}

extern "C" __global__ void LayernormBwdWeightBias(const INPUT_TYPE* __restrict__ dy,
                                                  const INPUT_TYPE* __restrict__ x,
                                                  const INPUT_TYPE* __restrict__ mean,
                                                  const INPUT_TYPE* __restrict__ rstd,
                                                  OUTPUT_TYPE* __restrict__ dw,
                                                  OUTPUT_TYPE* __restrict__ db)
{
    layernormbwdweightbias<INPUT_TYPE, OUTPUT_TYPE>(dy, x, mean, rstd, dw, db);
}

extern "C" __global__ void LayernormBwdWeightBiasParallel(const INPUT_TYPE* __restrict__ dy,
                                                          const INPUT_TYPE* __restrict__ x,
                                                          const INPUT_TYPE* __restrict__ mean,
                                                          const INPUT_TYPE* __restrict__ rstd,
                                                          OUTPUT_TYPE* __restrict__ workspace)
{
    layernormbwdweightbiasparallel<INPUT_TYPE, OUTPUT_TYPE>(dy, x, mean, rstd, workspace);
}

extern "C" __global__ void LayernormBwdReduceSum(const INPUT_TYPE* __restrict__ workspace,
                                                 OUTPUT_TYPE* __restrict__ dw,
                                                 OUTPUT_TYPE* __restrict__ db)
{
    layernormbwdreducesum<INPUT_TYPE, OUTPUT_TYPE>(workspace, dw, db);
}
