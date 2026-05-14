// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <type_traits>

#include "Bfloat16Dev.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr unsigned int INNER_SIZE = HIP_PLUGIN_RMSNORM_INNER_SIZE;

using InputType = HIP_PLUGIN_RMSNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_RMSNORM_OUTPUT_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

template <typename T>
struct Cast;

template <>
struct Cast<float>
{
    static __device__ __forceinline__ float to(float value)
    {
        return value;
    }
    static __device__ __forceinline__ float from(float value)
    {
        return value;
    }
};

template <>
struct Cast<half>
{
    static __device__ __forceinline__ float to(half value)
    {
        return __half2float(value);
    }
    static __device__ __forceinline__ half from(float value)
    {
        return __float2half(value);
    }
};

template <>
struct Cast<ushort>
{
    static __device__ __forceinline__ float to(ushort value)
    {
        return bfloat16_to_float(value);
    }
    static __device__ __forceinline__ ushort from(float value)
    {
        return float_to_bfloat16(value);
    }
};

template <typename T>
__device__ __forceinline__ float to_float32(T value)
{
    return Cast<T>::to(value);
}

template <typename T>
__device__ __forceinline__ T from_float32(float value)
{
    return Cast<T>::from(value);
}

extern "C" __global__ void RMSnormFwd(const InputType* __restrict__ x,
                                      const ScaleType* __restrict__ weight,
                                      const ScaleType* __restrict__ bias,
                                      OutputType* __restrict__ y,
                                      ComputeType* __restrict__ rstd,
                                      float eps)
{
    // ComputeType must be float to prevent precision loss
    static_assert(std::is_same<ComputeType, float>::value,
                  "ComputeType must be float for the RMSnormFwd kernel");

    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;

    float pvar = 0.0f;
    __shared__ float ltmp[LOCAL_SIZE];

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = gid * INNER_SIZE + i;
        float tmp = to_float32<InputType>(x[idx]);
        pvar += tmp * tmp;
    }

    ltmp[lid] = pvar;
    __syncthreads();
    for(unsigned int i = LOCAL_SIZE >> 1; i > 0; i >>= 1)
    {
        if(lid < i)
        {
            ltmp[lid] += ltmp[lid + i];
        }
        __syncthreads();
    }

    pvar = ltmp[0] / INNER_SIZE;
    float prstd = rsqrtf(pvar + eps);

    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // forward calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = gid * INNER_SIZE + i;
        float y_val = to_float32<InputType>(x[idx]) * prstd * to_float32<ScaleType>(weight[i]);
        if(bias != nullptr)
        {
            y_val += to_float32<ScaleType>(bias[i]);
        }
        y[idx] = from_float32<OutputType>(y_val);
    }
}
