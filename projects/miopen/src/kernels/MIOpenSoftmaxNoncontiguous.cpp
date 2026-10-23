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

template <bool IS_CONTIGUOUS>
__forceinline__ __device__ unsigned int get_index(unsigned int n,
                                                  unsigned int i,
                                                  unsigned int s,
                                                  unsigned int s0,
                                                  unsigned int s1,
                                                  const int offset)
{
    auto idx = offset;
    if constexpr(IS_CONTIGUOUS)
    {
        idx += (n * VECTOR_SIZE + i) * SPATIAL_DIM + s;
    }
    else if constexpr(USE_SOFTMAX_MODE_INSTANCE)
    {
        auto i0 = i / (HEIGHT * WIDTH);
        auto i1 = (i % (HEIGHT * WIDTH)) / WIDTH;
        auto i2 = (i % (HEIGHT * WIDTH)) % WIDTH;
        idx += n * N_STRIDE + i0 * C_STRIDE + i1 * H_STRIDE + i2 * W_STRIDE;
    }
    else
    {
        idx += n * N_STRIDE + i * C_STRIDE + s0 * H_STRIDE + s1 * W_STRIDE;
    }
    return idx;
}

__forceinline__ __device__ unsigned int get_x_index(unsigned int n,
                                                    unsigned int i,
                                                    unsigned int s,
                                                    unsigned int s0,
                                                    unsigned int s1,
                                                    const unsigned int x_offset)
{
    return get_index<IS_X_CONTIGUOUS>(n, i, s, s0, s1, x_offset);
}

__forceinline__ __device__ unsigned int get_y_index(unsigned int n,
                                                    unsigned int i,
                                                    unsigned int s,
                                                    unsigned int s0,
                                                    unsigned int s1,
                                                    const unsigned int y_offset)
{
    return get_index<IS_Y_CONTIGUOUS>(n, i, s, s0, s1, y_offset);
}

__forceinline__ __device__ unsigned int get_dx_index(unsigned int n,
                                                     unsigned int i,
                                                     unsigned int s,
                                                     unsigned int s0,
                                                     unsigned int s1,
                                                     const unsigned int dx_offset)
{
    return get_index<IS_DX_CONTIGUOUS>(n, i, s, s0, s1, dx_offset);
}

__forceinline__ __device__ unsigned int get_dy_index(unsigned int n,
                                                     unsigned int i,
                                                     unsigned int s,
                                                     unsigned int s0,
                                                     unsigned int s1,
                                                     const unsigned int dy_offset)
{
    return get_index<IS_DY_CONTIGUOUS>(n, i, s, s0, s1, dy_offset);
}

template <typename T>
__forceinline__ __device__ void softmaxfwd(const T* __restrict__ x,
                                           T* __restrict__ y,
                                           const int x_offset,
                                           const int y_offset,
                                           const float alpha,
                                           const float beta)
{
    const auto lid = threadIdx.x;

    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];
    FLOAT_ACCUM tmp;

    if constexpr(NUM_BATCH == 1) // CSR-Vector like approach
    {
        // Total number of workgroups launched can be less than the gridsize, hence iterate over.
        for(auto gid = blockIdx.x; gid < GRID_SIZE; gid += WORKGROUPS)
        {
            auto n  = gid / SPATIAL_DIM; // nth image
            auto s  = gid % SPATIAL_DIM; // spatial dimension (h * w)
            auto s0 = s / WIDTH;
            auto s1 = s % WIDTH;

            FLOAT_ACCUM channel_max = CVT_FP32_2ACCUM(0.0f);

            if constexpr(!USE_SOFTMAX_FAST)
            {
                ltmp[lid] = -MAX_VAL_ACCUM;
                tmp       = -MAX_VAL_ACCUM;

                // Find the maximum value in a block of x
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                    auto x_idx = get_x_index(n, i, s, s0, s1, x_offset);
                    tmp        = max(CVT_FLOAT2ACCUM(x[x_idx]), tmp);
                });

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

            // Sum exp(x - channel_max) in linear space or log space
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx        = get_x_index(n, i, s, s0, s1, x_offset);
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]);

                if constexpr(USE_SOFTMAX_LOG)
                {
                    tmp = logaddexp(value - channel_max, tmp);
                }
                else
                {
                    tmp += exp(value - channel_max);
                }
            });

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

            // Calculate y = alpha * exp(x) / channel_sum + beta * y
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto x_idx = get_x_index(n, i, s, s0, s1, x_offset);
                auto y_idx = get_y_index(n, i, s, s0, s1, y_offset);

                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]) - channel_max;

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
                    value += CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);
                }
                y[y_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        const auto gid = blockIdx.x;

        const auto batch_lid = lid & (BATCH_SIZE - 1);
        const auto batch     = lid / BATCH_SIZE;
        const auto batch_n   = (NUM_BATCH * gid + batch) / SPATIAL_DIM;
        const auto batch_s   = (NUM_BATCH * gid + batch) % SPATIAL_DIM;
        const auto batch_s0  = batch_s / WIDTH;
        const auto batch_s1  = batch_s % WIDTH;

        ltmp[lid] = -MAX_VAL_ACCUM;
        tmp       = -MAX_VAL_ACCUM;

        FLOAT_ACCUM x_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& x_value : x_values)
        {
            x_value = -MAX_VAL_ACCUM;
        }

        // Find the maximum value in a block of x
        unsigned int index = 0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto x_idx = get_x_index(batch_n, i, batch_s, batch_s0, batch_s1, x_offset);

                x_values[index] = CVT_FLOAT2ACCUM(x[x_idx]);
                if constexpr(!USE_SOFTMAX_FAST)
                {
                    tmp = max(x_values[index], tmp);
                }
            }
            ++index;
        });

        FLOAT_ACCUM channel_max = CVT_FP32_2ACCUM(0.0f);
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
        // Sum exp(x - channel_max) in linear space or log space
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int) {
            FLOAT_ACCUM value = x_values[index] - channel_max;
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
                x_values[index] = value;
            }
            ++index;
        });

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

        // Normalize each value in the channel by the channel_sum
        index = 0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto y_idx = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1, y_offset);

                if constexpr(USE_SOFTMAX_LOG)
                {
                    x_values[index] -= channel_sum;
                }
                else
                {
                    x_values[index] *= channel_sum;
                }

                x_values[index] = x_values[index] * CVT_FP32_2ACCUM(alpha);
                if constexpr(!ZERO_BETA)
                {
                    x_values[index] += CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);
                }
                y[y_idx] = CVT_ACCUM2FLOAT(x_values[index]);
            }
            ++index;
        });
    }
}

template <typename T>
__forceinline__ __device__ void softmaxbwd(const T* __restrict__ y,
                                           const T* __restrict__ dy,
                                           T* __restrict__ dx,
                                           const int y_offset,
                                           const int dy_offset,
                                           const int dx_offset,
                                           const float alpha,
                                           const float beta)
{
    const auto lid = threadIdx.x;
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    if constexpr(NUM_BATCH == 1) // CSR-Vector like approach
    {
        // Total number of workgroups launched can be less than the gridsize, hence iterate over.
        for(auto gid = blockIdx.x; gid < GRID_SIZE; gid += WORKGROUPS)
        {
            auto n  = gid / SPATIAL_DIM;
            auto s  = gid % SPATIAL_DIM;
            auto s0 = s / WIDTH;
            auto s1 = s % WIDTH;

            // Calculate the sum of dy or the elementwise multiplication of dy and y
            FLOAT_ACCUM channel_dot = CVT_FP32_2ACCUM(0.0f);
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx = get_dy_index(n, i, s, s0, s1, dy_offset);
                auto y_idx  = get_y_index(n, i, s, s0, s1, y_offset);

                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    value *= CVT_FLOAT2ACCUM(y[y_idx]);
                }
                channel_dot += value;
            });

            // Reduce all sums if needed
            if constexpr(LOCAL_SIZE > 1)
            {
                reduce<LOCAL_SIZE>(ltmp, lid, channel_dot, reduce_sum);
                channel_dot = ltmp[0];
            }

            // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](unsigned int i) {
                auto dy_idx = get_dy_index(n, i, s, s0, s1, dy_offset);
                auto y_idx  = get_y_index(n, i, s, s0, s1, y_offset);
                auto dx_idx = get_dx_index(n, i, s, s0, s1, dx_offset);

                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(USE_SOFTMAX_LOG)
                {
                    value -= channel_dot * exp(CVT_FLOAT2ACCUM(y[y_idx]));
                }
                else
                {
                    value = (value - channel_dot) * CVT_FLOAT2ACCUM(y[y_idx]);
                }
                value *= CVT_FP32_2ACCUM(alpha);
                if constexpr(!ZERO_BETA)
                {
                    value += CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta);
                }
                dx[dx_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        const auto gid = blockIdx.x;

        const auto batch_lid    = lid & (BATCH_SIZE - 1);
        const auto batch        = lid / BATCH_SIZE;
        const auto batch_n      = (NUM_BATCH * gid + batch) / SPATIAL_DIM;
        const auto batch_s      = (NUM_BATCH * gid + batch) % SPATIAL_DIM;
        const auto batch_s0     = batch_s / WIDTH;
        const auto batch_s1     = batch_s % WIDTH;
        FLOAT_ACCUM channel_dot = CVT_FP32_2ACCUM(0.0f);
        FLOAT_ACCUM y_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& y_value : y_values)
        {
            y_value = CVT_FP32_2ACCUM(0.0f);
        }
        FLOAT_ACCUM dy_values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& dy_value : dy_values)
        {
            dy_value = CVT_FP32_2ACCUM(0.0f);
        }

        unsigned int index = 0;
        // Calculate the sum of dy or the elementwise multiplication of dy and y
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto y_idx  = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1, y_offset);
                auto dy_idx = get_dy_index(batch_n, i, batch_s, batch_s0, batch_s1, dy_offset);

                y_values[index]  = CVT_FLOAT2ACCUM(y[y_idx]);
                dy_values[index] = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    dy_values[index] *= y_values[index];
                }
                channel_dot += dy_values[index];
            }
            ++index;
        });
        // Reduce all sums if needed
        if constexpr(BATCH_SIZE > 1)
        {
            reduce_block<LOCAL_SIZE, BATCH_SIZE>(ltmp, lid, batch_lid, channel_dot, reduce_sum);
            channel_dot = ltmp[batch * BATCH_SIZE];
        }

        // Calculate dx = (dy - channel_dot) * alpha * y + beta * dy
        index = 0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](unsigned int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto dx_idx = get_dx_index(batch_n, i, batch_s, batch_s0, batch_s1, dx_offset);

                if constexpr(USE_SOFTMAX_LOG)
                {
                    dy_values[index] -= channel_dot * exp(y_values[index]);
                }
                else
                {
                    dy_values[index] -= channel_dot * y_values[index];
                }

                auto value = dy_values[index] * CVT_FP32_2ACCUM(alpha);
                if constexpr(!ZERO_BETA)
                {
                    value += CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta);
                }
                dx[dx_idx] = CVT_ACCUM2FLOAT(value);
            }
            ++index;
        });
    }
}

extern "C" __global__ void SoftmaxFwd(const DATA_TYPE* __restrict__ x,
                                      DATA_TYPE* __restrict__ y,
                                      const int x_offset,
                                      const int y_offset,
                                      const float alpha,
                                      const float beta)
{
    softmaxfwd<DATA_TYPE>(x, y, x_offset, y_offset, alpha, beta);
}

extern "C" __global__ void SoftmaxBwd(const DATA_TYPE* __restrict__ y,
                                      const DATA_TYPE* __restrict__ dy,
                                      DATA_TYPE* __restrict__ dx,
                                      const int y_offset,
                                      const int dy_offset,
                                      const int dx_offset,
                                      const float alpha,
                                      const float beta)
{
    softmaxbwd<DATA_TYPE>(y, dy, dx, y_offset, dy_offset, dx_offset, alpha, beta);
}
