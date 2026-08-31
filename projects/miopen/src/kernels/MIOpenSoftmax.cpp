// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
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
                       const unsigned int batch_lid,
                       FLOAT_ACCUM value_lid,
                       FUNCTION&& function)
{
    array[lid] = value_lid;
    __syncthreads();

#pragma nounroll
    for(auto i = ARRAY_SIZE >> 1; i > 0; i >>= 1)
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

template <bool IS_CONTIGUOUS>
__forceinline__ __device__ int get_index(int n, int i, int s, int s0, int s1, const int offset)
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

__forceinline__ __device__ int get_x_index(int n, int i, int s, int s0, int s1, const int x_offset)
{
    return get_index<IS_INPUT_CONTIGUOUS>(n, i, s, s0, s1, x_offset);
}

__forceinline__ __device__ int get_y_index(int n, int i, int s, int s0, int s1, const int y_offset)
{
    return get_index<IS_OUTPUT_CONTIGUOUS>(n, i, s, s0, s1, y_offset);
}

__forceinline__ __device__ int
get_dx_index(int n, int i, int s, int s0, int s1, const int dx_offset)
{
    return get_index<IS_DINPUT_CONTIGUOUS>(n, i, s, s0, s1, dx_offset);
}

__forceinline__ __device__ int
get_dy_index(int n, int i, int s, int s0, int s1, const int dy_offset)
{
    return get_index<IS_DOUTPUT_CONTIGUOUS>(n, i, s, s0, s1, dy_offset);
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
        /* Entire workgroup works on one spatial_dim.
         * We use logarithmic reductions to compute max and sum per channel.
         * This approach reads in the same data thrice from DRAM but is still better
         * than launching three different kernels.
         * The workgroup begins by computing the nth image and s (spatial_dim) it
         * is working on and iterates over the entire grid until finished.
         */

        // Total number of workgroups launched can be less than the gridsize, hence iterate over.
        for(auto gid = blockIdx.x; gid < GRID_SIZE; gid += WORKGROUPS)
        {
            auto n  = gid / SPATIAL_DIM; // nth image
            auto s  = gid % SPATIAL_DIM; // spatial dimension (h * w)
            auto s0 = s / WIDTH;
            auto s1 = s % WIDTH;

            FLOAT_ACCUM channel_max;

            if constexpr(!USE_SOFTMAX_FAST)
            {
                ltmp[lid] = -MAX_VAL_ACCUM;
                tmp       = -MAX_VAL_ACCUM;

                // Compute max per channel
                // Iterate over all the channels one thread is supposed to loop over
                // and compute max
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                    auto x_idx = get_x_index(n, i, s, s0, s1, x_offset);
                    tmp        = max(CVT_FLOAT2ACCUM(x[x_idx]), tmp);
                });

                reduce<LOCAL_SIZE>(ltmp, lid, lid, tmp, reduce_max);

                channel_max = ltmp[0];
                __syncthreads();
            }

            if constexpr(USE_SOFTMAX_LOG)
            {
                tmp = NEGATIVE_CUTOFF_VAL<FLOAT_ACCUM>;
            }
            else
            {
                tmp = 0;
            }

            // Subtract channel_max from each value
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                auto x_idx        = get_x_index(n, i, s, s0, s1, x_offset);
                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]);

                // Compute exponent of each value
                // Then sum all the values touched by this thread
                if constexpr(USE_SOFTMAX_LOG)
                {
                    tmp = logaddexp(value - channel_max, tmp);
                }
                else if constexpr(USE_SOFTMAX_FAST)
                {
                    tmp += exp(value);
                }
                else
                {
                    tmp += exp(value - channel_max);
                }
            });

            reduce<LOCAL_SIZE>(ltmp, lid, lid, tmp, reduce_sum_log);

            FLOAT_ACCUM channel_sum = ltmp[0];

            // Normalize each value in the channel by the channel_sum
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                auto x_idx = get_x_index(n, i, s, s0, s1, x_offset);
                auto y_idx = get_y_index(n, i, s, s0, s1, y_offset);

                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]);

                // Subtracting max again because we do not write the output of
                // value-max to DRAM above. Doing a subtraction again is much
                // faster than writing uncoalesced to DRAM
                if constexpr(!USE_SOFTMAX_FAST)
                {
                    value = value - channel_max;
                }
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
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    value *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }

                value = value * CVT_FP32_2ACCUM(alpha) +
                        (beta != 0.0f ? CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta)
                                      : FLOAT_ACCUM{0});
                y[y_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        /* Each workgroup is computing the softmax for NUM_BATCH spatial_dims ala CSR-Stream.
         * The number of threads iterating over channels to compute softmax for one batch is
         * BATCH_SIZE. The number of values each thread works on is U_BATCH_SIZE (read micro batch
         * size). Each batch in the workgroup works on its nth image and s (spatial_dim). E.g. a 256
         * thread workgroup with c=31 has 8 batches and a batchsize of 32. The number of workgroups
         * launched are exactly the number as required hence, there is no for-loop.
         */

        const auto gid = blockIdx.x;

        // ID of the thread within the batch
        const auto batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
        const auto batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

        // Batch specific n and s
        const auto batch_n  = (NUM_BATCH * gid + batch) / SPATIAL_DIM; // nth image
        const auto batch_s  = (NUM_BATCH * gid + batch) % SPATIAL_DIM; // which spatial_dim/pixel
        const auto batch_s0 = batch_s / WIDTH;
        const auto batch_s1 = batch_s % WIDTH;

        if constexpr(!USE_SOFTMAX_FAST)
        {
            ltmp[lid] = -MAX_VAL_ACCUM;
            tmp       = -MAX_VAL_ACCUM;
        }

        FLOAT_ACCUM values[U_BATCH_SIZE];
        for(FLOAT_ACCUM& value : values)
        {
            value = -MAX_VAL_ACCUM;
        }

        // Compute max per channel
        // BATCH_SIZE threads iterate over the channels
        const auto index0 = batch_lid / BATCH_SIZE;
        auto index        = index0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto x_idx = get_x_index(batch_n, i, batch_s, batch_s0, batch_s1, x_offset);

                values[index] = CVT_FLOAT2ACCUM(x[x_idx]);
                if constexpr(!USE_SOFTMAX_FAST)
                {
                    tmp = max(values[index], tmp);
                }
            }
            ++index;
        });

        FLOAT_ACCUM channel_max;
        if constexpr(!USE_SOFTMAX_FAST)
        {
            reduce<BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_max);

            channel_max = ltmp[batch * BATCH_SIZE];
            __syncthreads();
        }

        if constexpr(USE_SOFTMAX_LOG)
        {
            tmp = NEGATIVE_CUTOFF_VAL<FLOAT_ACCUM>;
        }
        else
        {
            tmp = 0;
        }

        // Subtract channel_max from each value
        index = index0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int) {
            // Compute exponent of each value
            // Then sum all the values touched by this thread
            FLOAT_ACCUM value = values[index];
            if constexpr(!USE_SOFTMAX_FAST)
            {
                value -= channel_max;
            }
            if constexpr(!USE_SOFTMAX_LOG)
            {
                value = exp(value);
            }
            if constexpr(USE_SOFTMAX_LOG)
            {
                tmp = logaddexp(tmp, value);
            }
            else
            {
                tmp += value;
            }

            values[index] = value;
            ++index;
        });

        reduce<BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_sum_log);

        FLOAT_ACCUM channel_sum = ltmp[batch * BATCH_SIZE];

        // Normalize each value in the channel by the channel_sum
        index = index0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto y_idx = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1, y_offset);

                auto v_idx = index;

                if constexpr(USE_SOFTMAX_LOG)
                {
                    values[v_idx] -= channel_sum;
                }
                else
                {
                    // Multiply by approximate reciprocal of channel_sum. The approximate reciprocal
                    // is somewhat less accurate (1 ULP) than a full division, but is noticeably
                    // more performant.
                    values[v_idx] *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                }

                values[v_idx] = values[v_idx] * CVT_FP32_2ACCUM(alpha) +
                                (beta != 0.0f ? CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta)
                                              : FLOAT_ACCUM{0});

                y[y_idx] = CVT_ACCUM2FLOAT(values[v_idx]);
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
    __shared__ FLOAT_ACCUM ltmp[LOCAL_SIZE];

    const auto lid = threadIdx.x;

    if constexpr(NUM_BATCH == 1) // CSR-Vector like approach
    {
        // Total number of workgroups launched can be less than the gridsize, hence iterate over.
        for(auto gid = blockIdx.x; gid < GRID_SIZE; gid += WORKGROUPS)
        {
            auto n  = gid / SPATIAL_DIM; // nth image
            auto s  = gid % SPATIAL_DIM; // spatial dimension (H * W)
            auto s0 = s / WIDTH;
            auto s1 = s % WIDTH;

            // Compute dot product per channel
            // Iterate over all the channels one thread is supposed to loop over
            // and compute dot-product
            FLOAT_ACCUM channel_dot = static_cast<FLOAT_ACCUM>(0);
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                auto dy_idx = get_dy_index(n, i, s, s0, s1, dy_offset);
                auto y_idx  = get_y_index(n, i, s, s0, s1, y_offset);

                FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    value *= CVT_FLOAT2ACCUM(y[y_idx]);
                }
                channel_dot += value;
            });

            reduce<LOCAL_SIZE>(ltmp, lid, lid, channel_dot, reduce_sum);

            channel_dot = ltmp[0];

            // Subtract and element-wise multiplication
            loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
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
                value = value * CVT_FP32_2ACCUM(alpha) +
                        (beta != 0.0f ? CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta)
                                      : FLOAT_ACCUM{0});
                dx[dx_idx] = CVT_ACCUM2FLOAT(value);
            });
        }
    }
    else // CSR-Stream like approach
    {
        const auto gid = blockIdx.x;

        // ID of the thread within the batch
        const auto batch_lid = lid & (BATCH_SIZE - 1); // thread specific channel_st
        const auto batch     = lid / BATCH_SIZE;       // which spatial_dim or pixel

        // Batch specific n and s
        const auto batch_n  = (NUM_BATCH * gid + batch) / SPATIAL_DIM; // nth image
        const auto batch_s  = (NUM_BATCH * gid + batch) % SPATIAL_DIM; // which spatial_dim/pixel
        const auto batch_s0 = batch_s / WIDTH;
        const auto batch_s1 = batch_s % WIDTH;
        FLOAT_ACCUM channel_dot = static_cast<FLOAT_ACCUM>(0);

        // stores all the values touched by one thread so that we do not have load
        // again as the CSR-Vector approach
        FLOAT_ACCUM y_values[U_BATCH_SIZE]  = {0};
        FLOAT_ACCUM dy_values[U_BATCH_SIZE] = {0};

        // Compute dot product per channel
        // BATCH_SIZE threads iterate over the channels
        const auto index0 = batch_lid / BATCH_SIZE;
        auto index        = index0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto y_idx  = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1, y_offset);
                auto dy_idx = get_dy_index(batch_n, i, batch_s, batch_s0, batch_s1, dy_offset);

                y_values[index]  = CVT_FLOAT2ACCUM(y[y_idx]);
                dy_values[index] = CVT_FLOAT2ACCUM(dy[dy_idx]);
                auto value       = dy_values[index];
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    value *= y_values[index];
                }
                channel_dot += value;
            }
            ++index;
        });

        reduce<BATCH_SIZE>(ltmp, lid, batch_lid, channel_dot, reduce_sum);

        channel_dot = ltmp[batch * BATCH_SIZE];

        // Subtract and element-wise multiplication
        index = index0;
        loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                auto dx_idx = get_dx_index(batch_n, i, batch_s, batch_s0, batch_s1, dx_offset);

                if constexpr(USE_SOFTMAX_LOG)
                {
                    dy_values[index] -= channel_dot * exp(y_values[index]);
                }
                else
                {
                    dy_values[index] = (dy_values[index] - channel_dot) * y_values[index];
                }

                auto value = dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                             (beta != 0.0f ? CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta)
                                           : FLOAT_ACCUM{0});
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
