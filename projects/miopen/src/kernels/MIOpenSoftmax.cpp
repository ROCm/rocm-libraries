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

// The number of elements of T that fit inside of load_t (i.e. int4)
template <typename T>
constexpr static unsigned int load_factor = sizeof(load_t) / sizeof(T);

// The largest possible array of elements of T that fit inside of load_t (i.e. int4)
template <typename T>
using vec_t = array<T, load_factor<T>>;

// Load up to load_factor elements from src into a vec_t, doing a vectorised load if possible
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

// Store up to load_factor elements into dst from a vec_t, doing a vectorized store if possible
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

// Calculate log(exp(x) + exp(y))
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

// Perform a reduction of function across the entire array
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

// Perform a reduction of function across blocks of the array
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

// Perform a loop of lambda given a bound, a step and an offset lid
// This loop is mostly known on compile time, in contrast to
// for(int i = lid; i < BOUND; i += step)
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

// Obtain the indices gid, offset and stride for CSR-vector from blockIdx depending on an indexing
// mode
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

// Obtain the indices gid, offset and stride for CSR-stream from blockIdx depending on an indexing
// mode
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

// Deduplicated operations needed at multiple points in the kernels

template <typename T>
__forceinline__ __device__ void max_operation(const T& x, FLOAT_ACCUM& tmp)
{
    tmp = max(CVT_FLOAT2ACCUM(x), tmp);
}

template <typename T>
__forceinline__ __device__ void
max_batch_operation(const T& x, FLOAT_ACCUM& x_value, unsigned int& index, FLOAT_ACCUM& tmp)
{
    x_value = CVT_FLOAT2ACCUM(x);
    if constexpr(!USE_SOFTMAX_FAST)
    {
        tmp = max(x_value, tmp);
    }
    ++index;
}

template <typename T>
__forceinline__ __device__ void
sum_exp_operation(const T& x, const FLOAT_ACCUM& channel_max, FLOAT_ACCUM& tmp)
{
    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x) - channel_max;
    if constexpr(USE_SOFTMAX_LOG)
    {
        tmp = logaddexp(value, tmp);
    }
    else
    {
        tmp += exp(value);
    }
}

__forceinline__ __device__ void sum_exp_batch_operation(FLOAT_ACCUM& x_value,
                                                        const FLOAT_ACCUM& channel_max,
                                                        unsigned int& index,
                                                        FLOAT_ACCUM& tmp)
{
    FLOAT_ACCUM value = x_value - channel_max;
    if constexpr(USE_SOFTMAX_LOG)
    {
        tmp = logaddexp(tmp, value);
    }
    else
    {
        value = exp(value);
        tmp += value;
    }
    if constexpr(!USE_SOFTMAX_FAST || !USE_SOFTMAX_LOG)
    {
        x_value = value;
    }
    ++index;
}

template <typename T>
__forceinline__ __device__ void softmax_fwd_operation(const T& x,
                                                      T& y,
                                                      const FLOAT_ACCUM& channel_max,
                                                      const FLOAT_ACCUM& channel_sum,
                                                      const float& alpha,
                                                      const float& beta)
{
    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x) - channel_max;
    if constexpr(USE_SOFTMAX_LOG)
    {
        value -= channel_sum;
    }
    else
    {
        value = exp(value) * channel_sum;
    }
    value = value * CVT_FP32_2ACCUM(alpha);
    if constexpr(!ZERO_BETA)
    {
        value += CVT_FLOAT2ACCUM(y) * CVT_FP32_2ACCUM(beta);
    }
    y = CVT_ACCUM2FLOAT(value);
}

template <typename T>
__forceinline__ __device__ void softmax_fwd_batch_operation(FLOAT_ACCUM& x_value,
                                                            T& y,
                                                            const FLOAT_ACCUM& channel_sum,
                                                            const float& alpha,
                                                            const float& beta,
                                                            unsigned int& index)
{
    if constexpr(USE_SOFTMAX_LOG)
    {
        x_value -= channel_sum;
    }
    else
    {
        x_value *= channel_sum;
    }
    x_value = x_value * CVT_FP32_2ACCUM(alpha);
    if constexpr(!ZERO_BETA)
    {
        x_value += CVT_FLOAT2ACCUM(y) * CVT_FP32_2ACCUM(beta);
    }
    y = CVT_ACCUM2FLOAT(x_value);
    ++index;
}

template <typename T>
__forceinline__ __device__ void sum_dot_operation(const T& dy, const T& y, FLOAT_ACCUM& channel_dot)
{
    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy);
    if constexpr(!USE_SOFTMAX_LOG)
    {
        value *= CVT_FLOAT2ACCUM(y);
    }
    channel_dot += value;
}

template <typename T>
__forceinline__ __device__ void sum_dot_batch_operation(const T& dy,
                                                        FLOAT_ACCUM& dy_value,
                                                        const T& y,
                                                        FLOAT_ACCUM& y_value,
                                                        FLOAT_ACCUM& channel_dot,
                                                        unsigned int& index)
{
    dy_value = CVT_FLOAT2ACCUM(dy);
    y_value  = CVT_FLOAT2ACCUM(y);
    if constexpr(!USE_SOFTMAX_LOG)
    {
        dy_value *= y_value;
    }
    channel_dot += dy_value;
    ++index;
}

template <typename T>
__forceinline__ __device__ void softmax_bwd_operation(const T& dy,
                                                      const T& y,
                                                      T& dx,
                                                      const FLOAT_ACCUM& channel_dot,
                                                      const float& alpha,
                                                      const float& beta)
{
    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy);
    if constexpr(USE_SOFTMAX_LOG)
    {
        value -= channel_dot * exp(CVT_FLOAT2ACCUM(y));
    }
    else
    {
        value = (value - channel_dot) * CVT_FLOAT2ACCUM(y);
    }
    value = value * CVT_FP32_2ACCUM(alpha);
    if constexpr(!ZERO_BETA)
    {
        value += CVT_FLOAT2ACCUM(dx) * CVT_FP32_2ACCUM(beta);
    }
    dx = CVT_ACCUM2FLOAT(value);
}

template <typename T>
__forceinline__ __device__ void softmax_bwd_batch_operation(FLOAT_ACCUM& dy_value,
                                                            const FLOAT_ACCUM& y_value,
                                                            T& dx,
                                                            const FLOAT_ACCUM& channel_dot,
                                                            const float& alpha,
                                                            const float& beta,
                                                            unsigned int& index)
{
    if constexpr(USE_SOFTMAX_LOG)
    {
        dy_value -= channel_dot * exp(y_value);
    }
    else
    {
        dy_value -= channel_dot * y_value;
    }
    FLOAT_ACCUM value = dy_value * CVT_FP32_2ACCUM(alpha);
    if constexpr(!ZERO_BETA)
    {
        value += CVT_FLOAT2ACCUM(dx) * CVT_FP32_2ACCUM(beta);
    }
    dx = CVT_ACCUM2FLOAT(value);
    ++index;
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
                // Find the maximum value in a block of x with vectorised loading and double
                // buffering
                unsigned int i = lid * load_factor<T>;
                auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        max_operation(xdata.data[k], tmp);
                    }
                    xdata = xtmp;
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    max_operation(xdata.data[k], tmp);
                }
            }
            else
            {
                // Find the maximum value in a block of x
                loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                    auto x_idx = i * STRIDE + offset + X_OFFSET;
                    max_operation(x[x_idx], tmp);
                });
            }
            // Reduce all maxima if necessary
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
            // Sum exp(x - channel_max) in linear space or log space with vectorised loading and
            // double buffering
            unsigned int i = lid * load_factor<T>;
            auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
            i += LOCAL_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += LOCAL_SIZE * load_factor<T>)
            {
                auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    sum_exp_operation(xdata.data[k], channel_max, tmp);
                }
                xdata = xtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                sum_exp_operation(xdata.data[k], channel_max, tmp);
            }
        }
        else
        {
            // Sum exp(x - channel_max) in linear space or log space
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx = i * STRIDE + offset + X_OFFSET;
                sum_exp_operation(x[x_idx], channel_max, tmp);
            });
        }
        FLOAT_ACCUM channel_sum;
        // Reduce all sums if necessary
        if constexpr(LOCAL_SIZE > 1)
        {
            reduce<LOCAL_SIZE>(ltmp, lid, tmp, reduce_sum_log);
            channel_sum = ltmp[0];
        }
        else
        {
            channel_sum = tmp;
        }
        // Prepare reciprocal if needed later
        if constexpr(!USE_SOFTMAX_LOG)
        {
            // Calculate approximate reciprocal of channel_sum. The approximate reciprocal
            // is somewhat less accurate (1 ULP) than a full division, but is noticeably
            // more performant.
            channel_sum = __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
        }

        if constexpr(VECTORIZED)
        {
            // Calculate y = alpha * exp(x) / channel_sum + beta * y using vectorised loading and
            // double buffering
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
                    softmax_fwd_operation(
                        xdata.data[k], ydata.data[k], channel_max, channel_sum, alpha, beta);
                }
                store(i - LOCAL_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
                xdata = xtmp;
                ydata = ytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                softmax_fwd_operation(
                    xdata.data[k], ydata.data[k], channel_max, channel_sum, alpha, beta);
            }
            store(i - LOCAL_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
        }
        else
        {
            // Calculate y = alpha * exp(x) / channel_sum + beta * y
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx = i * STRIDE + offset + X_OFFSET;
                auto y_idx = i * STRIDE + offset + Y_OFFSET;
                softmax_fwd_operation(x[x_idx], y[y_idx], channel_max, channel_sum, alpha, beta);
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
            // Find the maximum value in a block of x with vectorised loading and double buffering
            unsigned int i = batch_lid * load_factor<T>;
            auto xdata     = load<T, true>(i, offset + X_OFFSET, x);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto xtmp = load<T, true>(i, offset + X_OFFSET, x);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    max_batch_operation(xdata.data[k], x_values[index], index, tmp);
                }
                xdata = xtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                max_batch_operation(xdata.data[k], x_values[index], index, tmp);
            }
        }
        else
        {
            // Find the maximum value in a block of x
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto x_idx = i * STRIDE + offset + X_OFFSET;
                max_batch_operation(x[x_idx], x_values[index], index, tmp);
            });
        }

        FLOAT_ACCUM channel_max = 0;
        if constexpr(!USE_SOFTMAX_FAST)
        {
            // Reduce all maxima if necessary
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
            // Sum exp(x - channel_max) in linear space or log space with vectorised loading and
            // double buffering
            for(unsigned int i = batch_lid * load_factor<T>; i < INNER_SIZE;
                i += BATCH_SIZE * load_factor<T>)
            {
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    sum_exp_batch_operation(x_values[index], channel_max, index, tmp);
                }
            }
        }
        else
        {
            // Sum exp(x - channel_max) in linear space or log space
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                sum_exp_batch_operation(x_values[index], channel_max, index, tmp);
            });
        }
        FLOAT_ACCUM channel_sum;
        // Reduce all sums if necessary
        if constexpr(BATCH_SIZE > 1)
        {
            reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_sum_log);
            channel_sum = ltmp[batch * BATCH_SIZE];
        }
        else
        {
            channel_sum = tmp;
        }
        // Prepare reciprocal if needed later
        if constexpr(!USE_SOFTMAX_LOG)
        {
            // Calculate approximate reciprocal of channel_sum. The approximate reciprocal
            // is somewhat less accurate (1 ULP) than a full division, but is noticeably
            // more performant.
            channel_sum = __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
        }

        index = 0;
        if constexpr(VECTORIZED)
        {
            // Calculate y = alpha * exp(x) / channel_sum + beta * y using vectorised loading and
            // double buffering
            unsigned int i = batch_lid * load_factor<T>;
            auto ydata     = load(i, offset + Y_OFFSET, y);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto ytmp = load(i, offset + Y_OFFSET, y);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    softmax_fwd_batch_operation(
                        x_values[index], ydata.data[k], channel_sum, alpha, beta, index);
                }
                store(i - BATCH_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
                ydata = ytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                softmax_fwd_batch_operation(
                    x_values[index], ydata.data[k], channel_sum, alpha, beta, index);
            }
            store(i - BATCH_SIZE * load_factor<T>, offset + Y_OFFSET, y, ydata);
        }
        else
        {
            // Calculate y = alpha * exp(x) / channel_sum + beta * y
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto y_idx = i * STRIDE + offset + Y_OFFSET;
                softmax_fwd_batch_operation(
                    x_values[index], y[y_idx], channel_sum, alpha, beta, index);
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
            // Calculate the sum of dy or the elementwise multiplication of dy and y using
            // vectorised loading and double buffering
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
                    sum_dot_operation(dydata.data[k], ydata.data[k], channel_dot);
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
                sum_dot_operation(dydata.data[k], ydata.data[k], channel_dot);
            }
        }
        else
        {
            // Calculate the sum of dy or the elementwise multiplication of dy and y
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx = i * STRIDE + offset + DY_OFFSET;
                auto y_idx  = i * STRIDE + offset + Y_OFFSET;
                sum_dot_operation(dy[dy_idx], y[y_idx], channel_dot);
            });
        }
        // Reduce all sums if needed
        if constexpr(LOCAL_SIZE > 1)
        {
            reduce<LOCAL_SIZE>(ltmp, lid, channel_dot, reduce_sum);
            channel_dot = ltmp[0];
        }

        if constexpr(VECTORIZED)
        {
            // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy using vectorised loading
            // and double buffering
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
                    softmax_bwd_operation(
                        dydata.data[k], ydata.data[k], dxdata.data[k], channel_dot, alpha, beta);
                }
                store(i - LOCAL_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
                dydata = dytmp;
                ydata  = ytmp;
                dxdata = dxtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                softmax_bwd_operation(
                    dydata.data[k], ydata.data[k], dxdata.data[k], channel_dot, alpha, beta);
            }
            store(i - LOCAL_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
        }
        else
        {
            // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy
            loop<INNER_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx = i * STRIDE + offset + DY_OFFSET;
                auto y_idx  = i * STRIDE + offset + Y_OFFSET;
                auto dx_idx = i * STRIDE + offset + DX_OFFSET;
                softmax_bwd_operation(dy[dy_idx], y[y_idx], dx[dx_idx], channel_dot, alpha, beta);
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
            // Calculate the sum of dy or the elementwise multiplication of dy and y using
            // vectorised loading and double buffering
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
                    sum_dot_batch_operation(dydata.data[k],
                                            dy_values[index],
                                            ydata.data[k],
                                            y_values[index],
                                            channel_dot,
                                            index);
                }
                ydata  = ytmp;
                dydata = dytmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                sum_dot_batch_operation(dydata.data[k],
                                        dy_values[index],
                                        ydata.data[k],
                                        y_values[index],
                                        channel_dot,
                                        index);
            }
        }
        else
        {
            // Calculate the sum of dy or the elementwise multiplication of dy and y
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto y_idx  = i * STRIDE + offset + Y_OFFSET;
                auto dy_idx = i * STRIDE + offset + DY_OFFSET;
                sum_dot_batch_operation(
                    dy[dy_idx], dy_values[index], y[y_idx], y_values[index], channel_dot, index);
            });
        }
        // Reduce all sums if needed
        if constexpr(BATCH_SIZE > 1)
        {
            reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, channel_dot, reduce_sum);
            channel_dot = ltmp[batch * BATCH_SIZE];
        }

        index = 0;
        if constexpr(VECTORIZED)
        {
            // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy using vectorised loading
            // and double buffering
            unsigned int i = batch_lid * load_factor<T>;
            auto dxdata    = load(i, offset + DX_OFFSET, dx);
            i += BATCH_SIZE * load_factor<T>;
            for(; i < INNER_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto dxtmp = load(i, offset + DX_OFFSET, dx);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    softmax_bwd_batch_operation(dy_values[index],
                                                y_values[index],
                                                dxdata.data[k],
                                                channel_dot,
                                                alpha,
                                                beta,
                                                index);
                }
                store(i - BATCH_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
                dxdata = dxtmp;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                softmax_bwd_batch_operation(dy_values[index],
                                            y_values[index],
                                            dxdata.data[k],
                                            channel_dot,
                                            alpha,
                                            beta,
                                            index);
            }
            store(i - BATCH_SIZE * load_factor<T>, offset + DX_OFFSET, dx, dxdata);
        }
        else
        {
            // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy
            loop<INNER_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
                auto dx_idx = i * STRIDE + offset + DX_OFFSET;
                softmax_bwd_batch_operation(
                    dy_values[index], y_values[index], dx[dx_idx], channel_dot, alpha, beta, index);
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
