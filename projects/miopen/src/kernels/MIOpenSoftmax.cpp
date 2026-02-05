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

template <bool IS_CONTIGUOUS>
constexpr bool VECTORIZED_C =
    VECTORIZED &&
    ((IS_CONTIGUOUS && SPATIAL_DIM == 1) ||
     (!IS_CONTIGUOUS && USE_SOFTMAX_MODE_INSTANCE && HEIGHT * WIDTH == 1 && C_STRIDE == 1) ||
     (!IS_CONTIGUOUS && USE_SOFTMAX_MODE_CHANNEL && C_STRIDE == 1));
constexpr bool VECTORIZED_C_X  = VECTORIZED_C<IS_INPUT_CONTIGUOUS>;
constexpr bool VECTORIZED_C_Y  = VECTORIZED_C<IS_OUTPUT_CONTIGUOUS>;
constexpr bool VECTORIZED_C_DX = VECTORIZED_C<IS_DINPUT_CONTIGUOUS>;
constexpr bool VECTORIZED_C_DY = VECTORIZED_C<IS_DOUTPUT_CONTIGUOUS>;

using load_t = int4;

template <typename T, unsigned int N>
struct array
{
    T data[N];
};

template <typename T>
constexpr static unsigned int load_factor = sizeof(load_t) / sizeof(T);

template <typename T, bool IS_VECTORIZED>
constexpr static unsigned int ADJUSTED_U_BATCH_SIZE =
    IS_VECTORIZED ? U_BATCH_SIZE * load_factor<T> : U_BATCH_SIZE;

template <typename T>
using vec_t = array<T, load_factor<T>>;

template <typename T,
          unsigned int BOUND,
          bool IS_CONTIGUOUS,
          bool DEFAULT_NEGATIVE_CUTOFF_VAL = false>
__forceinline__ __device__ static vec_t<T>
load(unsigned long i, const unsigned long i_offset, const T* __restrict__ src)
{
    if(IS_CONTIGUOUS && i + load_factor<T> < BOUND)
    {
        const load_t value = *reinterpret_cast<const load_t*>(&src[i + i_offset]);
        const auto values  = *reinterpret_cast<const vec_t<T>*>(&value);
        return values;
    }
    else
    {
        vec_t<T> values = {{}};
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i + k < BOUND)
            {
                values.data[k] = src[i + k + i_offset];
            }
            else if constexpr(DEFAULT_NEGATIVE_CUTOFF_VAL)
            {
                values.data[k] = CVT_FP32_2FLOAT(NEGATIVE_CUTOFF_VAL<float>);
            }
        }
        return values;
    }
}

template <typename T, unsigned int BOUND, bool IS_CONTIGUOUS>
__forceinline__ __device__ static void
store(unsigned long i, const unsigned long i_offset, T* __restrict__ dst, vec_t<T>& data)
{
    if(IS_CONTIGUOUS && i + load_factor<T> < BOUND)
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
                dst[i + k + i_offset] = data.data[k];
            }
        }
    }
}

template <typename T, unsigned int BOUND, bool IS_CONTIGUOUS, unsigned int I_COND_STRIDE = 1>
__forceinline__ __device__ static void store_if(unsigned long i,
                                                const unsigned long i_offset,
                                                unsigned long i_cond,
                                                T* __restrict__ dst,
                                                vec_t<T>& data)
{
    if(IS_CONTIGUOUS && i_cond + I_COND_STRIDE * load_factor<T> < BOUND)
    {
        *reinterpret_cast<load_t*>(&dst[i + i_offset]) = *reinterpret_cast<load_t*>(&data);
    }
    else
    {
#pragma unroll
        for(int k = 0; k < load_factor<T>; ++k)
        {
            if(i_cond + k * I_COND_STRIDE < BOUND)
            {
                dst[i + k + i_offset] = data.data[k];
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

template <bool IS_CONTIGUOUS, unsigned long OFFSET>
__forceinline__ __device__ unsigned long get_index(int n, int i, int s, int s0, int s1)
{
    auto idx = OFFSET;
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

__forceinline__ __device__ unsigned long get_x_index(int n, int i, int s, int s0, int s1)
{
    return get_index<IS_INPUT_CONTIGUOUS, X_OFFSET>(n, i, s, s0, s1);
}

__forceinline__ __device__ unsigned long get_y_index(int n, int i, int s, int s0, int s1)
{
    return get_index<IS_OUTPUT_CONTIGUOUS, Y_OFFSET>(n, i, s, s0, s1);
}

__forceinline__ __device__ unsigned long get_dx_index(int n, int i, int s, int s0, int s1)
{
    return get_index<IS_DINPUT_CONTIGUOUS, DX_OFFSET>(n, i, s, s0, s1);
}

__forceinline__ __device__ unsigned long get_dy_index(int n, int i, int s, int s0, int s1)
{
    return get_index<IS_DOUTPUT_CONTIGUOUS, DY_OFFSET>(n, i, s, s0, s1);
}

template <typename T>
__forceinline__ __device__ void
softmaxfwd(const T* __restrict__ x, T* __restrict__ y, const float alpha, const float beta)
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
                if constexpr(VECTORIZED_C_X)
                {
                    unsigned long i = lid * load_factor<T>;
                    auto x_offset   = get_x_index(n, 0, s, s0, s1);
                    auto tmpx       = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
                    i += LOCAL_SIZE * load_factor<T>;
                    for(; i < VECTOR_SIZE; i += LOCAL_SIZE * load_factor<T>)
                    {
                        auto tmpdata = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
                        __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                        for(int k = 0; k < load_factor<T>; ++k)
                        {
                            tmp = max(CVT_FLOAT2ACCUM(tmpx.data[k]), tmp);
                        }
                        __builtin_amdgcn_sched_barrier(0);
                        tmpx = tmpdata;
                    }
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        tmp = max(CVT_FLOAT2ACCUM(tmpx.data[k]), tmp);
                    }
                }
                else
                {
                    loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                        auto x_idx = get_x_index(n, i, s, s0, s1);
                        tmp        = max(CVT_FLOAT2ACCUM(x[x_idx]), tmp);
                    });
                }

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

            if constexpr(VECTORIZED_C_X)
            {
                // Subtract channel_max from each value
                unsigned long i = lid * load_factor<T>;
                auto x_offset   = get_x_index(n, 0, s, s0, s1);
                auto tmpx       = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < VECTOR_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto tmpdata = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpx.data[k]);

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
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    tmpx = tmpdata;
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpx.data[k]);

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
                }
            }
            else
            {
                // Subtract channel_max from each value
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                    auto x_idx        = get_x_index(n, i, s, s0, s1);
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
            }

            reduce<LOCAL_SIZE>(ltmp, lid, lid, tmp, reduce_sum_log);

            FLOAT_ACCUM channel_sum = ltmp[0];

            if constexpr(VECTORIZED_C_X && VECTORIZED_C_Y)
            {
                // Normalize each value in the channel by the channel_sum
                unsigned long i = lid * load_factor<T>;
                auto x_offset   = get_x_index(n, 0, s, s0, s1);
                auto y_offset   = get_y_index(n, 0, s, s0, s1);
                auto tmpx       = load<T, VECTOR_SIZE, true>(i, x_offset, x);
                auto tmpy       = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < VECTOR_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto tmpdata  = load<T, VECTOR_SIZE, true>(i, x_offset, x);
                    auto tmpdata2 = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpx.data[k]);

                        // Subtracting max again because we do not write the output of
                        // value-max to DRAM above. Doing a subtraction again is much
                        // faster than writing uncoalesced to DRAM
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
                            value -= channel_sum;
                        }
                        else
                        {
                            // Multiply by approximate reciprocal of channel_sum. The approximate
                            // reciprocal is somewhat less accurate (1 ULP) than a full division,
                            // but is noticeably more performant.
                            value *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                        }
                        value = value * CVT_FP32_2ACCUM(alpha) +
                                CVT_FLOAT2ACCUM(tmpy.data[k]) * CVT_FP32_2ACCUM(beta);
                        tmpy.data[k] = CVT_ACCUM2FLOAT(value);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    auto tmpyout = tmpy;
                    tmpx         = tmpdata;
                    tmpy         = tmpdata2;
                    store<T, VECTOR_SIZE, true>(
                        i - LOCAL_SIZE * load_factor<T>, y_offset, y, tmpyout);
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpx.data[k]);

                    // Subtracting max again because we do not write the output of
                    // value-max to DRAM above. Doing a subtraction again is much
                    // faster than writing uncoalesced to DRAM
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
                            CVT_FLOAT2ACCUM(tmpy.data[k]) * CVT_FP32_2ACCUM(beta);
                    tmpy.data[k] = CVT_ACCUM2FLOAT(value);
                }
                store<T, VECTOR_SIZE, true>(i - LOCAL_SIZE * load_factor<T>, y_offset, y, tmpy);
            }
            else
            {
                // Normalize each value in the channel by the channel_sum
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                    auto x_idx = get_x_index(n, i, s, s0, s1);
                    auto y_idx = get_y_index(n, i, s, s0, s1);

                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(x[x_idx]);

                    // Subtracting max again because we do not write the output of
                    // value-max to DRAM above. Doing a subtraction again is much
                    // faster than writing uncoalesced to DRAM
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
                            CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);
                    y[y_idx] = CVT_ACCUM2FLOAT(value);
                });
            }
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

        // Vectorizations in CSR-Stream are dependent on each other due to the layout of `values`
        constexpr bool VECTORIZED_C_STREAM = VECTORIZED_C_X && VECTORIZED_C_Y;

        FLOAT_ACCUM values[ADJUSTED_U_BATCH_SIZE<T, VECTORIZED_C_STREAM>];
        for(FLOAT_ACCUM& value : values)
        {
            value = -MAX_VAL_ACCUM;
        }

        // Compute max per channel
        // BATCH_SIZE threads iterate over the channels
        const auto index0 = batch_lid / BATCH_SIZE;
        auto index        = index0;
        if constexpr(VECTORIZED_C_STREAM)
        {
            unsigned long i = batch_lid * load_factor<T>;
            auto x_offset   = get_x_index(batch_n, 0, batch_s, batch_s0, batch_s1);
            vec_t<T> tmpx;
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                tmpx = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
            }
            i += BATCH_SIZE * load_factor<T>;
            for(; i < VECTOR_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto tmpdata = load<T, VECTOR_SIZE, true, true>(i, x_offset, x);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                           batch_s <
                       VECTOR_SIZE * GRID_SIZE)
                    {
                        values[index] = CVT_FLOAT2ACCUM(tmpx.data[k]);
                        if constexpr(!USE_SOFTMAX_FAST)
                        {
                            tmp = max(values[index], tmp);
                        }
                    }
                    ++index;
                }
                __builtin_amdgcn_sched_barrier(0);
                tmpx = tmpdata;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                       batch_s <
                   VECTOR_SIZE * GRID_SIZE)
                {
                    values[index] = CVT_FLOAT2ACCUM(tmpx.data[k]);
                    if constexpr(!USE_SOFTMAX_FAST)
                    {
                        tmp = max(values[index], tmp);
                    }
                }
                ++index;
            }
        }
        else
        {
            loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
                if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
                {
                    auto x_idx = get_x_index(batch_n, i, batch_s, batch_s0, batch_s1);

                    values[index] = CVT_FLOAT2ACCUM(x[x_idx]);
                    if constexpr(!USE_SOFTMAX_FAST)
                    {
                        tmp = max(values[index], tmp);
                    }
                }
                ++index;
            });
        }

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
        if constexpr(VECTORIZED_C_STREAM)
        {
            // Compute exponent of each value
            // Then sum all the values touched by this thread
            for(unsigned long i = batch_lid * load_factor<T>; i < VECTOR_SIZE;
                i += BATCH_SIZE * load_factor<T>)
            {
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if((batch_n * VECTOR_SIZE + i + k) * SPATIAL_DIM + batch_s <
                       VECTOR_SIZE * GRID_SIZE)
                    {
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
                    }
                    ++index;
                }
            }
        }
        else
        {
            loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
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
        }

        reduce<BATCH_SIZE>(ltmp, lid, batch_lid, tmp, reduce_sum_log);

        FLOAT_ACCUM channel_sum = ltmp[batch * BATCH_SIZE];

        // Normalize each value in the channel by the channel_sum
        index = index0;
        if constexpr(VECTORIZED_C_STREAM)
        {
            unsigned long i = batch_lid * load_factor<T>;
            auto y_offset   = get_y_index(batch_n, 0, batch_s, batch_s0, batch_s1);
            vec_t<T> tmpy;
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                tmpy = load<T, VECTOR_SIZE, true>(i, y_offset, y);
            }
            i += BATCH_SIZE * load_factor<T>;
            for(; i < VECTOR_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto tmpdata    = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                vec_t<T> tmpout = {{}};
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                           batch_s <
                       VECTOR_SIZE * GRID_SIZE)
                    {
                        if constexpr(USE_SOFTMAX_LOG)
                        {
                            values[index] -= channel_sum;
                        }
                        else
                        {
                            // Multiply by approximate reciprocal of channel_sum. The approximate
                            // reciprocal is somewhat less accurate (1 ULP) than a full division,
                            // but is noticeably more performant.
                            values[index] *=
                                __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                        }

                        values[index] = values[index] * CVT_FP32_2ACCUM(alpha) +
                                        CVT_FLOAT2ACCUM(tmpy.data[k]) * CVT_FP32_2ACCUM(beta);

                        tmpy.data[k] = CVT_ACCUM2FLOAT(values[index]);
                    }
                    ++index;
                }
                __builtin_amdgcn_sched_barrier(0);
                auto tmpyout = tmpy;
                tmpy         = tmpdata;
                store_if<T, VECTOR_SIZE * GRID_SIZE, true, SPATIAL_DIM>(
                    i - BATCH_SIZE * load_factor<T>,
                    y_offset,
                    (batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T>)*SPATIAL_DIM + batch_s,
                    y,
                    tmpyout);
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                       batch_s <
                   VECTOR_SIZE * GRID_SIZE)
                {
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        values[index] -= channel_sum;
                    }
                    else
                    {
                        // Multiply by approximate reciprocal of channel_sum. The approximate
                        // reciprocal is somewhat less accurate (1 ULP) than a full division, but is
                        // noticeably more performant.
                        values[index] *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                    }

                    values[index] = values[index] * CVT_FP32_2ACCUM(alpha) +
                                    CVT_FLOAT2ACCUM(tmpy.data[k]) * CVT_FP32_2ACCUM(beta);

                    tmpy.data[k] = CVT_ACCUM2FLOAT(values[index]);
                }
                ++index;
            }
            store_if<T, VECTOR_SIZE * GRID_SIZE, true, SPATIAL_DIM>(
                i - BATCH_SIZE * load_factor<T>,
                y_offset,
                (batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T>)*SPATIAL_DIM + batch_s,
                y,
                tmpy);
        }
        else
        {
            loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
                if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
                {
                    auto y_idx = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1);

                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        values[index] -= channel_sum;
                    }
                    else
                    {
                        // Multiply by approximate reciprocal of channel_sum. The approximate
                        // reciprocal is somewhat less accurate (1 ULP) than a full division, but is
                        // noticeably more performant.
                        values[index] *= __builtin_amdgcn_rcpf(channel_sum + EPSILON<FLOAT_ACCUM>);
                    }

                    values[index] = values[index] * CVT_FP32_2ACCUM(alpha) +
                                    CVT_FLOAT2ACCUM(y[y_idx]) * CVT_FP32_2ACCUM(beta);

                    y[y_idx] = CVT_ACCUM2FLOAT(values[index]);
                }
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
            if constexpr(VECTORIZED_C_Y && VECTORIZED_C_DY)
            {
                unsigned long i = lid * load_factor<T>;
                auto dy_offset  = get_dy_index(n, 0, s, s0, s1);
                int y_offset;
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    y_offset = get_y_index(n, 0, s, s0, s1);
                }
                auto tmpdy = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
                vec_t<T> tmpy;
                if constexpr(!USE_SOFTMAX_LOG)
                {
                    tmpy = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                }
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < VECTOR_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto tmpdydata = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
                    vec_t<T> tmpydata;
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        tmpydata = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                        if constexpr(!USE_SOFTMAX_LOG)
                        {
                            value *= CVT_FLOAT2ACCUM(tmpy.data[k]);
                        }
                        channel_dot += value;
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    tmpdy = tmpdydata;
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        tmpy = tmpydata;
                    }
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        value *= CVT_FLOAT2ACCUM(tmpy.data[k]);
                    }
                    channel_dot += value;
                }
            }
            else
            {
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                    auto dy_idx = get_dy_index(n, i, s, s0, s1);
                    auto y_idx  = get_y_index(n, i, s, s0, s1);

                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(dy[dy_idx]);
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        value *= CVT_FLOAT2ACCUM(y[y_idx]);
                    }
                    channel_dot += value;
                });
            }

            reduce<LOCAL_SIZE>(ltmp, lid, lid, channel_dot, reduce_sum);

            channel_dot = ltmp[0];

            // Subtract and element-wise multiplication
            if constexpr(VECTORIZED_C_Y && VECTORIZED_C_DX && VECTORIZED_C_DY)
            {
                unsigned long i = lid * load_factor<T>;
                auto dy_offset  = get_dy_index(n, 0, s, s0, s1);
                auto y_offset   = get_y_index(n, 0, s, s0, s1);
                auto dx_offset  = get_dx_index(n, 0, s, s0, s1);
                auto tmpdy      = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
                auto tmpy       = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                auto tmpdx      = load<T, VECTOR_SIZE, true>(i, dx_offset, dx);
                i += LOCAL_SIZE * load_factor<T>;
                for(; i < VECTOR_SIZE; i += LOCAL_SIZE * load_factor<T>)
                {
                    auto tmpdydata = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
                    auto tmpydata  = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                    auto tmpdxdata = load<T, VECTOR_SIZE, true>(i, dx_offset, dx);
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for(int k = 0; k < load_factor<T>; ++k)
                    {
                        FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                        if constexpr(USE_SOFTMAX_LOG)
                        {
                            value -= channel_dot * exp(CVT_FLOAT2ACCUM(tmpy.data[k]));
                        }
                        else
                        {
                            value = (value - channel_dot) * CVT_FLOAT2ACCUM(tmpy.data[k]);
                        }
                        value = value * CVT_FP32_2ACCUM(alpha) +
                                CVT_FLOAT2ACCUM(tmpdx.data[k]) * CVT_FP32_2ACCUM(beta);
                        tmpdx.data[k] = CVT_ACCUM2FLOAT(value);
                    }
                    __builtin_amdgcn_sched_barrier(0);
                    auto tmpdxout = tmpdx;
                    tmpdy         = tmpdydata;
                    tmpy          = tmpydata;
                    tmpdx         = tmpdxdata;
                    store<T, VECTOR_SIZE, true>(
                        i - LOCAL_SIZE * load_factor<T>, dx_offset, dx, tmpdxout);
                }
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    FLOAT_ACCUM value = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        value -= channel_dot * exp(CVT_FLOAT2ACCUM(tmpy.data[k]));
                    }
                    else
                    {
                        value = (value - channel_dot) * CVT_FLOAT2ACCUM(tmpy.data[k]);
                    }
                    value = value * CVT_FP32_2ACCUM(alpha) +
                            CVT_FLOAT2ACCUM(tmpdx.data[k]) * CVT_FP32_2ACCUM(beta);
                    tmpdx.data[k] = CVT_ACCUM2FLOAT(value);
                }
                store<T, VECTOR_SIZE, true>(i - LOCAL_SIZE * load_factor<T>, dx_offset, dx, tmpdx);
            }
            else
            {
                loop<VECTOR_SIZE, LOCAL_SIZE>(lid, [&](int i) {
                    auto dy_idx = get_dy_index(n, i, s, s0, s1);
                    auto y_idx  = get_y_index(n, i, s, s0, s1);
                    auto dx_idx = get_dx_index(n, i, s, s0, s1);

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

        // Vectorizations in CSR-Stream are dependent on each other due to the layout of `y_values`
        // and `dy_values`
        constexpr bool VECTORIZED_C_STREAM = VECTORIZED_C_Y && VECTORIZED_C_DX && VECTORIZED_C_DY;

        // stores all the values touched by one thread so that we do not have load
        // again as the CSR-Vector approach
        FLOAT_ACCUM y_values[ADJUSTED_U_BATCH_SIZE<T, VECTORIZED_C_STREAM>]  = {0};
        FLOAT_ACCUM dy_values[ADJUSTED_U_BATCH_SIZE<T, VECTORIZED_C_STREAM>] = {0};

        // Compute dot product per channel
        // BATCH_SIZE threads iterate over the channels
        const auto index0 = batch_lid / BATCH_SIZE;
        auto index        = index0;
        if constexpr(VECTORIZED_C_STREAM)
        {
            unsigned long i = batch_lid * load_factor<T>;
            auto y_offset   = get_y_index(batch_n, 0, batch_s, batch_s0, batch_s1);
            auto dy_offset  = get_dy_index(batch_n, 0, batch_s, batch_s0, batch_s1);
            vec_t<T> tmpy;
            vec_t<T> tmpdy;
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                tmpy  = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                tmpdy = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
            }
            i += BATCH_SIZE * load_factor<T>;
            for(; i < VECTOR_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto tmpydata  = load<T, VECTOR_SIZE, true>(i, y_offset, y);
                auto tmpdydata = load<T, VECTOR_SIZE, true>(i, dy_offset, dy);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                           batch_s <
                       VECTOR_SIZE * GRID_SIZE)
                    {
                        y_values[index]  = CVT_FLOAT2ACCUM(tmpy.data[k]);
                        dy_values[index] = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                        auto value       = dy_values[index];
                        if constexpr(!USE_SOFTMAX_LOG)
                        {
                            value *= y_values[index];
                        }
                        channel_dot += value;
                    }
                    ++index;
                }
                __builtin_amdgcn_sched_barrier(0);
                tmpy  = tmpydata;
                tmpdy = tmpdydata;
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                       batch_s <
                   VECTOR_SIZE * GRID_SIZE)
                {
                    y_values[index]  = CVT_FLOAT2ACCUM(tmpy.data[k]);
                    dy_values[index] = CVT_FLOAT2ACCUM(tmpdy.data[k]);
                    auto value       = dy_values[index];
                    if constexpr(!USE_SOFTMAX_LOG)
                    {
                        value *= y_values[index];
                    }
                    channel_dot += value;
                }
                ++index;
            }
        }
        else
        {
            loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
                if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
                {
                    auto y_idx  = get_y_index(batch_n, i, batch_s, batch_s0, batch_s1);
                    auto dy_idx = get_dy_index(batch_n, i, batch_s, batch_s0, batch_s1);

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
        }

        reduce<BATCH_SIZE>(ltmp, lid, batch_lid, channel_dot, reduce_sum);

        channel_dot = ltmp[batch * BATCH_SIZE];

        // Subtract and element-wise multiplication
        index = index0;
        if constexpr(VECTORIZED_C_STREAM)
        {
            unsigned long i = batch_lid * load_factor<T>;
            auto dx_offset  = get_dx_index(batch_n, 0, batch_s, batch_s0, batch_s1);
            vec_t<T> tmpdx;
            if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
            {
                tmpdx = load<T, VECTOR_SIZE, true>(i, dx_offset, dx);
            }
            i += BATCH_SIZE * load_factor<T>;
            for(; i < VECTOR_SIZE; i += BATCH_SIZE * load_factor<T>)
            {
                auto tmpdata = load<T, VECTOR_SIZE, true>(i, dx_offset, dx);
                __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                for(int k = 0; k < load_factor<T>; ++k)
                {
                    if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                           batch_s <
                       VECTOR_SIZE * GRID_SIZE)
                    {
                        if constexpr(USE_SOFTMAX_LOG)
                        {
                            dy_values[index] -= channel_dot * exp(y_values[index]);
                        }
                        else
                        {
                            dy_values[index] = (dy_values[index] - channel_dot) * y_values[index];
                        }

                        auto value = dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                     CVT_FLOAT2ACCUM(tmpdx.data[k]) * CVT_FP32_2ACCUM(beta);
                        tmpdx.data[k] = CVT_ACCUM2FLOAT(value);
                    }
                    ++index;
                }
                __builtin_amdgcn_sched_barrier(0);
                auto tmpdxout = tmpdx;
                tmpdx         = tmpdata;
                store_if<T, VECTOR_SIZE * GRID_SIZE, true, SPATIAL_DIM>(
                    i - BATCH_SIZE * load_factor<T>,
                    dx_offset,
                    (batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T>)*SPATIAL_DIM + batch_s,
                    dx,
                    tmpdx);
            }
#pragma unroll
            for(int k = 0; k < load_factor<T>; ++k)
            {
                if((batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T> + k) * SPATIAL_DIM +
                       batch_s <
                   VECTOR_SIZE * GRID_SIZE)
                {
                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        dy_values[index] -= channel_dot * exp(y_values[index]);
                    }
                    else
                    {
                        dy_values[index] = (dy_values[index] - channel_dot) * y_values[index];
                    }

                    auto value = dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                 CVT_FLOAT2ACCUM(tmpdx.data[k]) * CVT_FP32_2ACCUM(beta);
                    tmpdx.data[k] = CVT_ACCUM2FLOAT(value);
                }
                ++index;
            }
            store_if<T, VECTOR_SIZE * GRID_SIZE, true, SPATIAL_DIM>(
                i - BATCH_SIZE * load_factor<T>,
                dx_offset,
                (batch_n * VECTOR_SIZE + i - BATCH_SIZE * load_factor<T>)*SPATIAL_DIM + batch_s,
                dx,
                tmpdx);
        }
        else
        {
            loop<VECTOR_SIZE, BATCH_SIZE>(batch_lid, [&](int i) {
                if((batch_n * VECTOR_SIZE + i) * SPATIAL_DIM + batch_s < VECTOR_SIZE * GRID_SIZE)
                {
                    auto dx_idx = get_dx_index(batch_n, i, batch_s, batch_s0, batch_s1);

                    if constexpr(USE_SOFTMAX_LOG)
                    {
                        dy_values[index] -= channel_dot * exp(y_values[index]);
                    }
                    else
                    {
                        dy_values[index] = (dy_values[index] - channel_dot) * y_values[index];
                    }

                    auto value = dy_values[index] * CVT_FP32_2ACCUM(alpha) +
                                 CVT_FLOAT2ACCUM(dx[dx_idx]) * CVT_FP32_2ACCUM(beta);
                    dx[dx_idx] = CVT_ACCUM2FLOAT(value);
                }
                ++index;
            });
        }
    }
}

extern "C" __global__ void SoftmaxFwd(const DATA_TYPE* __restrict__ x,
                                      DATA_TYPE* __restrict__ y,
                                      const float alpha,
                                      const float beta)
{
    softmaxfwd<DATA_TYPE>(x, y, alpha, beta);
}

extern "C" __global__ void SoftmaxBwd(const DATA_TYPE* __restrict__ y,
                                      const DATA_TYPE* __restrict__ dy,
                                      DATA_TYPE* __restrict__ dx,
                                      const float alpha,
                                      const float beta)
{
    softmaxbwd<DATA_TYPE>(y, dy, dx, alpha, beta);
}
