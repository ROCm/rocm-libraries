// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef MIOPEN_HIP_RUNTIME_COMPILE
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename T>
constexpr T NEGATIVE_CUTOFF_VAL = T{-1e20};

template <typename T>
constexpr T EPSILON = T{1e-12};

using load_t = int4;

template <typename T, unsigned int N>
struct array
{
    T data[N];
};

template <typename T>
constexpr static unsigned int load_factor = sizeof(load_t) / sizeof(T);

template <typename T>
using vec_t = array<T, load_factor<T>>;

template <typename T,
          bool DEFAULT_NEGATIVE_CUTOFF_VAL = false,
          unsigned int BOUND               = INNER_SIZE,
          unsigned int I_STRIDE            = STRIDE>
__forceinline__ __device__ static vec_t<T>
load(unsigned int i, const unsigned int i_offset, const T* __restrict__ src)
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
            else if constexpr(DEFAULT_NEGATIVE_CUTOFF_VAL)
            {
                values.data[k] = CVT_FP32_2FLOAT(NEGATIVE_CUTOFF_VAL<float>);
            }
        }
        return values;
    }
}

template <typename T, unsigned int BOUND = INNER_SIZE, unsigned int I_STRIDE = STRIDE>
__forceinline__ __device__ static void
store(unsigned int i, const unsigned int i_offset, T* __restrict__ dst, vec_t<T>& data)
{
    if(I_STRIDE == 1 && i + load_factor<T> < BOUND)
    {
        *reinterpret_cast<load_t*>(&dst[i + i_offset]) = *reinterpret_cast<load_t*>(&data);
    }
    else
    {
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i + k < BOUND)
            {
                dst[(i + k) * I_STRIDE + i_offset] = data.data[k];
            }
        }
    }
}

template <typename T>
__device__ T logaddexp(T x, T y)
{
    T a = max(x, y);
    T b = min(x, y);
    T c = b - a;

    // Cppcheck doesn't properly recognize that NEGATIVE_CUTOFF_VAL<T> is a template instantiation
    // cppcheck-suppress internalAstError
    return c <= NEGATIVE_CUTOFF_VAL<T> ? max(a, NEGATIVE_CUTOFF_VAL<T>)
                                       : max(T{a + log(T{1} + exp(c))}, NEGATIVE_CUTOFF_VAL<T>);
}

template <int ARRAY_SIZE, typename FUNCTION>
__device__ void reduce(FLOAT_ACCUM array[ARRAY_SIZE],
                       const unsigned int lid,
                       FLOAT_ACCUM value_lid,
                       FUNCTION&& function)
{
    array[lid] = value_lid;
    __syncthreads();

#pragma nounroll
    for(auto i = ARRAY_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            array[lid] = function(array[lid], array[lid + i]);
        }
        __syncthreads();
    }
}

template <int ARRAY_SIZE, int BLOCK_SIZE, typename FUNCTION>
__device__ void reduce_block(FLOAT_ACCUM array[ARRAY_SIZE],
                             const unsigned int lid,
                             const unsigned int batch_lid,
                             FLOAT_ACCUM value_batch_lid,
                             FUNCTION&& function)
{
    array[lid] = value_batch_lid;
    __syncthreads();

#pragma nounroll
    for(auto i = BLOCK_SIZE >> 1; i > 0; i >>= 1)
    {
        if(batch_lid < i)
        {
            array[lid] = function(array[lid], array[lid + i]);
        }
        __syncthreads();
    }
}

constexpr struct
{
    template <typename T>
    __forceinline__ __device__ constexpr T operator()(T a, T b) const
    {
        return a + b;
    }
} reduce_sum;

constexpr struct
{
    template <typename T>
    __forceinline__ __device__ constexpr T operator()(T a, T b) const
    {
        if constexpr(USE_SOFTMAX_LOG)
        {
            return logaddexp(a, b);
        }
        else
        {
            return a + b;
        }
    }
} reduce_sum_log;

constexpr struct
{
    template <typename T>
    __forceinline__ __device__ constexpr T operator()(T a, T b) const
    {
        return max(a, b);
    }
} reduce_max;

template <int BOUND, int STEP, typename LAMBDA>
__device__ void loop(const unsigned int lid, LAMBDA&& lambda)
{
    auto i = 0;
#pragma nounroll
    for(; i + STEP < BOUND; i += STEP)
    {
        lambda(i + lid);
    }
    if(i + lid < BOUND)
    {
        lambda(i + lid);
    }
}

__forceinline__ __device__ void get_indices(unsigned int& gid, unsigned int& o, unsigned int& s)
{
    if constexpr(SEPARATE_STRIDE)
    {
        o   = blockIdx.x;
        s   = blockIdx.y;
        gid = o * STRIDE + s;
    }
    else
    {
        gid = blockIdx.x;
        o   = blockIdx.x / STRIDE;
        s   = blockIdx.x % STRIDE;
    }
}

__forceinline__ __device__ void
get_indices_stream(unsigned int& o, unsigned int& s, const unsigned int batch)
{
    if constexpr(SEPARATE_STRIDE)
    {
        o = NUM_BATCH * blockIdx.x + batch;
        s = blockIdx.y;
    }
    else
    {
        o = (NUM_BATCH * blockIdx.x + batch) / STRIDE;
        s = (NUM_BATCH * blockIdx.x + batch) % STRIDE;
    }
}

template <typename T>
__forceinline__ __device__ void
softmaxfwd(const T* __restrict__ x, T* __restrict__ y, const float alpha, const float beta)
{
    const unsigned int lid = threadIdx.x;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    if constexpr(NUM_BATCH == 1) // CSR-Vector like approach
    {
        unsigned int gid, o, s;
        get_indices(gid, o, s);
        const unsigned int offset = o * INNER_SIZE * STRIDE + s;
        FLOAT_ACCUM tmp           = -MAX_VAL_ACCUM;
        FLOAT_ACCUM channel_max   = 0;
        if constexpr(!USE_SOFTMAX_FAST)
        {
            if constexpr(VECTORIZED)
            {
                unsigned int i = lid * load_factor<T>;
                auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        tmp = max(CVT_FLOAT2ACCUM(xdata.data[k]), tmp);
                    }
                    xdata = xtmp;
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    tmp = max(CVT_FLOAT2ACCUM(xdata.data[k]), tmp);
                }
            }
            else
            {
                loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                    auto x_idx = i * STRIDE + offset + X_OFFSET;
                    tmp        = max(CVT_FLOAT2ACCUM(x[x_idx]), tmp);
                });
            }
            if constexpr(LOCAL_SIZE > 1)
            {
                reduce<LOCAL_SIZE>(ltmp, lid, tmp, reduce_max);
                channel_max = ltmp[0];
                __syncthreads();
            }
            else
            {
                channel_max = tmp;
            }
        }

        if constexpr(USE_SOFTMAX_LOG)
        {
            tmp = NEGATIVE_CUTOFF_VAL<FLOAT_ACCUM>;
        }
        else
        {
            tmp = 0;
        }
        if constexpr(VECTORIZED)
        {
            unsigned int i = lid * load_factor<T>;
            auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
            i += LOCAL_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
            {
                auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(xdata.data[k]) - channel_max;
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        tmp = logaddexp(value, tmp);
                    }
                    else
                    {
                        tmp += exp(value);
                    }
                }
                xdata = xtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(xdata.data[k]) - channel_max;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    tmp = logaddexp(value, tmp);
                }
                else
                {
                    tmp += exp(value);
                }
            }
        }
        else
        {
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx        = i * STRIDE + offset + X_OFFSET;
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]) - channel_max;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    tmp = logaddexp(value, tmp);
                }
                else
                {
                    tmp += exp(value);
                }
            });
        }
        FLOAT_ACCUM channel_sum;
        if constexpr(LOCAL_SIZE > 1)
        {
            reduce<LOCAL_SIZE>(ltmp, lid, tmp, reduce_sum_log);
            channel_sum = ltmp[0];
        }
        else
        {
            channel_sum = tmp;
        }

        if constexpr(VECTORIZED)
        {
            unsigned int i = lid * load_factor<T>;
            auto xdata     = load<T, !USE_SOFTMAX_LOG>(i, offset + X_OFFSET, x);
            auto ydata     = load(i, offset + Y_OFFSET, y);
            i += LOCAL_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
            {
                auto xtmp = load<T, !USE_SOFTMAX_LOG>(i, offset + X_OFFSET, x);
                auto ytmp = load(i, offset + Y_OFFSET, y);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(xdata.data[k]) - channel_max;
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        value = exp(value);
                    }
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        value -= channel_sum;
                    }
                    else
                    {
                        // Multiply by approximate reciprocal of channel_sum. The approximate
                        // reciprocal is somewhat less accurate (1 ULP) than a full division, but is
                        // noticeably more performant.
                        value *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                    }
                    value = value * CVT_FP32_2ACCUM(alpha) +
                            CVT_FLOAT2ACCUM(ydata.data[k]) * CVT_FP32_2ACCUM(beta);
                    ydata.data[k] = CVT_ACCUM2FLOAT(value);
                }
                store(i - LOCAL_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
                xdata = xtmp;
                ydata = ytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(xdata.data[k]) - channel_max;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    value -= channel_sum;
                }
                else
                {
                    value = exp(value);
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    value *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }
                value = value * CVT_FP32_2ACCUM(alpha) +
                        CVT_FLOAT2ACCUM(ydata.data[k]) * CVT_FP32_2ACCUM(beta);
                ydata.data[k] = CVT_ACCUM2FLOAT(value);
            }
            store(i - LOCAL_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
        }
        else
        {
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx        = i * STRIDE + offset + X_OFFSET;
                auto y_idx        = i * STRIDE + offset + Y_OFFSET;
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]) - channel_max;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    value -= channel_sum;
                }
                else
                {
                    value = exp(value);
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    value *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }
                value = value * CVT_FP32_2ACCUM(alpha) +
                        CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);
                y[y_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        const unsigned int batch_lid = lid % BATCH_SIZE;
        const unsigned int batch     = lid / BATCH_SIZE;
        unsigned int o, s;
        get_indices_stream(o, s, batch);
        if(o >= OUTER_SIZE)
        {
            return;
        }
        const unsigned int offset = o * INNER_SIZE * STRIDE + s;
        FLOAT_ACCUM tmp           = -MAX_VAL_ACCUM;
        FLOAT_ACCUM x_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& x_value : x_values)
        {
            x_value = -MAX_VAL_ACCUM;
        }
        unsigned int index = 0;
        if constexpr(VECTORIZED)
        {
            unsigned int i = batch_lid * load_factor<T>;
            auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    x_values[index] = CVT_FLOAT2ACCUM(xdata.data[k]);
                    if constexpr(!USE_SOFTMAX_FAST)
                    {
                        tmp = max(x_values[index], tmp);
                    }
                    ++index;
                }
                xdata = xtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                x_values[index] = CVT_FLOAT2ACCUM(xdata.data[k]);
                if constexpr(!USE_SOFTMAX_FAST)
                {
                    tmp = max(x_values[index], tmp);
                }
                ++index;
            }
        }
        else
        {
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto x_idx      = i * STRIDE + offset + X_OFFSET;
                x_values[index] = CVT_FLOAT2ACCUM(x[x_idx]);
                if constexpr(!USE_SOFTMAX_FAST)
                {
                    tmp = max(x_values[index], tmp);
                }
                ++index;
            });
        }

        FLOAT_ACCUM channel_max = 0;
        if constexpr(!USE_SOFTMAX_FAST)
        {
            if constexpr(BATCH_SIZE > 1)
            {
                reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_max);
                channel_max = ltmp[batch * BATCH_SIZE];
                __syncthreads();
            }
            else
            {
                channel_max = tmp;
            }
        }

        if constexpr(USE_SOFTMAX_LOG)
        {
            tmp = NEGATIVE_CUTOFF_VAL<FLOAT_ACCUM>;
        }
        else
        {
            tmp = 0;
        }
        index = 0;
        if constexpr(VECTORIZED)
        {
            for(unsigned int i = batch_lid * load_factor<T>; i < INNER_SIZE;
                i += BATCH_SIZE * load_factor<T>)
            {
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM x_value = x_values[index] - channel_max;
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        tmp = logaddexp(tmp, x_value);
                    }
                    else
                    {
                        x_value = exp(x_value);
                        tmp += x_value;
                    }
                    if constexpr(!USE_SOFTMAX_FAST || !USE_SOFTMAX_LOG)
                    {
                        x_values[index] = x_value;
                    }
                    ++index;
                }
            }
        }
        else
        {
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                FLOAT_ACCUM x_value = x_values[index] - channel_max;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    tmp = logaddexp(tmp, x_value);
                }
                else
                {
                    x_value = exp(x_value);
                    tmp += x_value;
                }
                if constexpr(!USE_SOFTMAX_FAST || !USE_SOFTMAX_LOG)
                {
                    x_values[index] = x_value;
                }
                ++index;
            });
        }
        FLOAT_ACCUM channel_sum;
        if constexpr(BATCH_SIZE > 1)
        {
            reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_sum_log);
            channel_sum = ltmp[batch * BATCH_SIZE];
        }
        else
        {
            channel_sum = tmp;
        }

        index = 0;
        if constexpr(VECTORIZED)
        {
            unsigned int i = batch_lid * load_factor<T>;
            auto ydata     = load(i, offset + Y_OFFSET, y);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto ytmp = load(i, offset + Y_OFFSET, y);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        x_values[index] -= channel_sum;
                    }
                    else
                    {
                        // Multiply by approximate reciprocal of channel_sum. The approximate
                        // reciprocal is somewhat less accurate (1 ULP) than a full division, but is
                        // noticeably more performant.
                        x_values[index] *=
                            __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                    }
                    x_values[index] = x_values[index] * CVT_FP32_2ACCUM(alpha) +
                                      CVT_FLOAT2ACCUM(ydata.data[k]) * CVT_FP32_2ACCUM(beta);
                    ydata.data[k] = CVT_ACCUM2FLOAT(x_values[index]);
                    ++index;
                }
                store(i - BATCH_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
                ydata = ytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if constexpr(USE_SOFTMAX_LOG)
                {
                    x_values[index] -= channel_sum;
                }
                else
                {
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    x_values[index] *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }
                x_values[index] = x_values[index] * CVT_FP32_2ACCUM(alpha) +
                                  CVT_FLOAT2ACCUM(ydata.data[k]) * CVT_FP32_2ACCUM(beta);
                ydata.data[k] = CVT_ACCUM2FLOAT(x_values[index]);
                ++index;
            }
            store(i - BATCH_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
        }
        else
        {
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto y_idx = i * STRIDE + offset + Y_OFFSET;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    x_values[index] -= channel_sum;
                }
                else
                {
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    x_values[index] *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }
                x_values[index] = x_values[index] * CVT_FP32_2ACCUM(alpha) +
                                  CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);
                y[y_idx] = CVT_ACCUM2FLOAT(x_values[index]);
                ++index;
            });
        }
    }
}

template <typename T>
__forceinline__ __device__ void softmaxbwd(const T* __restrict__ y,
                                           const T* __restrict__ dy,
                                           T* __restrict__ dx,
                                           const float alpha,
                                           const float beta)
{
    const unsigned int lid = threadIdx.x;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    if constexpr(NUM_BATCH == 1) // CSR-Vector like approach
    {
        unsigned int gid, o, s;
        get_indices(gid, o, s);
        const unsigned int offset = o * INNER_SIZE * STRIDE + s;
        FLOAT_ACCUM channel_dot   = 0;
        if constexpr(VECTORIZED)
        {
            unsigned int i = lid * load_factor<T>;
            auto dydata    = load(i, offset + DY_OFFSET, dy);
            vec_t<T> ydata;
            if constexpr(!USE_SOFTMAX_LOG)
            {
                ydata = load(i, offset + Y_OFFSET, y);
            }
            i += LOCAL_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
            {
                auto dytmp = load(i, offset + DY_OFFSET, dy);
                vec_t<T> ytmp;
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    ytmp = load(i, offset + Y_OFFSET, y);
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dydata.data[k]);
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        value *= CVT_FLOAT2ACCUM(ydata.data[k]);
                    }
                    channel_dot += value;
                }
                dydata = dytmp;
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    ydata = ytmp;
                }
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dydata.data[k]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    value *= CVT_FLOAT2ACCUM(ydata.data[k]);
                }
                channel_dot += value;
            }
        }
        else
        {
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx       = i * STRIDE + offset + DY_OFFSET;
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    auto y_idx = i * STRIDE + offset + Y_OFFSET;
                    value *= CVT_FLOAT2ACCUM(y[y_idx]);
                }
                channel_dot += value;
            });
        }
        if constexpr(LOCAL_SIZE > 1)
        {
            reduce<LOCAL_SIZE>(ltmp, lid, channel_dot, reduce_sum);
            channel_dot = ltmp[0];
        }

        if constexpr(VECTORIZED)
        {
            unsigned int i = lid * load_factor<T>;
            auto dydata    = load(i, offset + DY_OFFSET, dy);
            auto ydata     = load(i, offset + Y_OFFSET, y);
            auto dxdata    = load(i, offset + DX_OFFSET, dx);
            i += LOCAL_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
            {
                auto dytmp = load(i, offset + DY_OFFSET, dy);
                auto ytmp  = load(i, offset + Y_OFFSET, y);
                auto dxtmp = load(i, offset + DX_OFFSET, dx);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dydata.data[k]);
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        value -= channel_dot * exp(CVT_FLOAT2ACCUM(ydata.data[k]));
                    }
                    else
                    {
                        value = (value - channel_dot) * CVT_FLOAT2ACCUM(ydata.data[k]);
                    }
                    value = value * CVT_FP32_2ACCUM(alpha) +
                            CVT_FLOAT2ACCUM(dxdata.data[k]) * CVT_FP32_2ACCUM(beta);
                    dxdata.data[k] = CVT_ACCUM2FLOAT(value);
                }
                store(i - LOCAL_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
                dydata = dytmp;
                ydata  = ytmp;
                dxdata = dxtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dydata.data[k]);
                if constexpr(USE_SOFTMAX_LOG)
                {
                    value -= channel_dot * exp(CVT_FLOAT2ACCUM(ydata.data[k]));
                }
                else
                {
                    value = (value - channel_dot) * CVT_FLOAT2ACCUM(ydata.data[k]);
                }
                value = value * CVT_FP32_2ACCUM(alpha) +
                        CVT_FLOAT2ACCUM(dxdata.data[k]) * CVT_FP32_2ACCUM(beta);
                dxdata.data[k] = CVT_ACCUM2FLOAT(value);
            }
            store(i - LOCAL_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
        }
        else
        {
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx       = i * STRIDE + offset + DY_OFFSET;
                auto y_idx        = i * STRIDE + offset + Y_OFFSET;
                auto dx_idx       = i * STRIDE + offset + DX_OFFSET;
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(USE_SOFTMAX_LOG)
                {
                    value -= channel_dot * exp(CVT_FLOAT2ACCUM(y[y_idx]));
                }
                else
                {
                    value = (value - channel_dot) * CVT_FLOAT2ACCUM(y[y_idx]);
                }
                value = value * CVT_FP32_2ACCUM(alpha) +
                        CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta);
                dx[dx_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        unsigned int gid, o, s;
        get_indices(gid, o, s);
        const unsigned int batch_lid = lid % BATCH_SIZE;
        const unsigned int batch     = lid / BATCH_SIZE;
        o                            = (NUM_BATCH * gid + batch) / STRIDE;
        if(o >= OUTER_SIZE)
        {
            return;
        }
        s                         = (NUM_BATCH * gid + batch) % STRIDE;
        const unsigned int offset = o * INNER_SIZE * STRIDE + s;
        FLOAT_ACCUM channel_dot   = 0;
        FLOAT_ACCUM y_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& y_value : y_values)
        {
            y_value = 0;
        }
        FLOAT_ACCUM dy_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& dy_value : dy_values)
        {
            dy_value = 0;
        }
        unsigned int index = 0;
        if constexpr(VECTORIZED)
        {
            unsigned int i = batch_lid * load_factor<T>;
            auto ydata     = load(i, offset + Y_OFFSET, y);
            auto dydata    = load(i, offset + DY_OFFSET, dy);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto ytmp  = load(i, offset + Y_OFFSET, y);
                auto dytmp = load(i, offset + DY_OFFSET, dy);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    y_values[index]  = CVT_FLOAT2ACCUM(ydata.data[k]);
                    dy_values[index] = CVT_FLOAT2ACCUM(dydata.data[k]);
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        dy_values[index] *= y_values[index];
                    }
                    channel_dot += dy_values[index];
                    ++index;
                }
                ydata  = ytmp;
                dydata = dytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                y_values[index]  = CVT_FLOAT2ACCUM(ydata.data[k]);
                dy_values[index] = CVT_FLOAT2ACCUM(dydata.data[k]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    dy_values[index] *= y_values[index];
                }
                channel_dot += dy_values[index];
                ++index;
            }
        }
        else
        {
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto y_idx       = i * STRIDE + offset + Y_OFFSET;
                auto dy_idx      = i * STRIDE + offset + DY_OFFSET;
                y_values[index]  = CVT_FLOAT2ACCUM(y[y_idx]);
                dy_values[index] = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    dy_values[index] *= y_values[index];
                }
                channel_dot += dy_values[index];
                ++index;
            });
        }
        if constexpr(BATCH_SIZE > 1)
        {
            reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, channel_dot, reduce_sum);
            channel_dot = ltmp[batch * BATCH_SIZE];
        }

        index = 0;
        if constexpr(VECTORIZED)
        {
            unsigned int i = batch_lid * load_factor<T>;
            auto dxdata    = load(i, offset + DX_OFFSET, dx);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto dxtmp = load(i, offset + DX_OFFSET, dx);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        dy_values[index] -= channel_dot * exp(y_values[index]);
                    }
                    else
                    {
                        dy_values[index] -= channel_dot * y_values[index];
                    }
                    dxdata.data[k] =
                        CVT_ACCUM2FLOAT(dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                        CVT_FLOAT2ACCUM(dxdata.data[k]) * CVT_FP32_2ACCUM(beta));
                    ++index;
                }
                store(i - BATCH_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
                dxdata = dxtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if constexpr(USE_SOFTMAX_LOG)
                {
                    dy_values[index] -= channel_dot * exp(y_values[index]);
                }
                else
                {
                    dy_values[index] -= channel_dot * y_values[index];
                }
                dxdata.data[k] =
                    CVT_ACCUM2FLOAT(dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                    CVT_FLOAT2ACCUM(dxdata.data[k]) * CVT_FP32_2ACCUM(beta));
                ++index;
            }
            store(i - BATCH_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
        }
        else
        {
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto dx_idx = i * STRIDE + offset + DX_OFFSET;
                if constexpr(USE_SOFTMAX_LOG)
                {
                    dy_values[index] -= channel_dot * exp(y_values[index]);
                }
                else
                {
                    dy_values[index] -= channel_dot * y_values[index];
                }
                dx[dx_idx] = CVT_ACCUM2FLOAT(dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                             CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta));
                ++index;
            });
        }
    }
}

extern "C" __global__ __launch_bounds__(LOCAL_SIZE) void SoftmaxFwd(const DATA_TYPE* __restrict__ x,
                                                                    DATA_TYPE* __restrict__ y,
                                                                    const float alpha,
                                                                    const float beta)
{
    softmaxfwd<DATA_TYPE>(x, y, alpha, beta);
}

extern "C" __global__
__launch_bounds__(LOCAL_SIZE) void SoftmaxBwd(const DATA_TYPE* __restrict__ y,
                                              const DATA_TYPE* __restrict__ dy,
                                              DATA_TYPE* __restrict__ dx,
                                              const float alpha,
                                              const float beta)
{
    softmaxbwd<DATA_TYPE>(y, dy, dx, alpha, beta);
}
