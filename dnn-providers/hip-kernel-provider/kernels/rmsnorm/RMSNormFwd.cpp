// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Bfloat16Dev.hpp"

constexpr unsigned int LOCAL_SIZE = HIP_PLUGIN_RMSNORM_LOCAL_SIZE;
constexpr size_t C_SIZE = HIP_PLUGIN_RMSNORM_C_SIZE;
constexpr size_t C_STRIDE = HIP_PLUGIN_RMSNORM_C_STRIDE;
constexpr size_t N_STRIDE = C_SIZE * C_STRIDE;

using InputType = HIP_PLUGIN_RMSNORM_INPUT_TYPE;
using OutputType = HIP_PLUGIN_RMSNORM_OUTPUT_TYPE;
using ScaleType = HIP_PLUGIN_RMSNORM_SCALE_TYPE;
using ComputeType = HIP_PLUGIN_RMSNORM_COMPUTE_TYPE;

// Generic cast through float
template <typename From, typename To>
struct DataCast
{
    static __device__ __forceinline__ To run(From value)
    {
        return DataCast<float, To>::run(DataCast<From, float>::run(value));
    }
};

// No op for same type
template <typename T>
struct DataCast<T, T>
{
    static __device__ __forceinline__ T run(T value)
    {
        return value;
    }
};

// half to float
template <>
struct DataCast<half, float>
{
    static __device__ __forceinline__ float run(half value)
    {
        return __half2float(value);
    }
};

// float to half
template <>
struct DataCast<float, half>
{
    static __device__ __forceinline__ half run(float value)
    {
        return __float2half(value);
    }
};

// ushort (bfloat16) to float
template <>
struct DataCast<ushort, float>
{
    static __device__ __forceinline__ float run(ushort value)
    {
        return bfloat16_to_float(value);
    }
};

// float to ushort (bfloat16)
template <>
struct DataCast<float, ushort>
{
    static __device__ __forceinline__ ushort run(float value)
    {
        return float_to_bfloat16(value);
    }
};

// half to bfloat16
template <>
struct DataCast<half, ushort>
{
    static __device__ __forceinline__ ushort run(half value)
    {
        return float_to_bfloat16(__half2float(value));
    }
};

// bfloat16 to half
template <>
struct DataCast<ushort, half>
{
    static __device__ __forceinline__ half run(ushort value)
    {
        return __float2half(bfloat16_to_float(value));
    }
};

template <typename From, typename To>
__device__ __forceinline__ To Cast(From value)
{
    return DataCast<From, To>::run(value);
}

extern "C" __global__ void RMSnormFwd(const InputType* __restrict__ x,
                                      const ScaleType* __restrict__ weight,
                                      const ScaleType* __restrict__ bias,
                                      OutputType* __restrict__ y,
                                      ComputeType* __restrict__ rstd,
                                      float eps)
{
    const unsigned int gid = blockIdx.x;
    const unsigned int lid = threadIdx.x;

    ComputeType pvar = Cast<float, ComputeType>(0.0f);
    __shared__ ComputeType ltmp[LOCAL_SIZE];

    // reduce sum
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = gid * INNER_SIZE + i;
        ComputeType tmp = Cast<InputType, ComputeType>(x[idx]);
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

    ComputeType csize_c = Cast<float, ComputeType>(static_cast<float>(C_SIZE));
    ComputeType eps_c = Cast<float, ComputeType>(eps);
    pvar = ltmp[0] / csize_c;
    ComputeType prstd = Cast<float, ComputeType>(rsqrtf(Cast<ComputeType, float>(pvar + eps_c)));

    if(lid == 0 && rstd)
    {
        rstd[gid] = prstd;
    }

    // forward calculation
    for(unsigned int i = lid; i < INNER_SIZE; i += LOCAL_SIZE)
    {
        size_t idx = gid * INNER_SIZE + i;
        ComputeType y_val = Cast<InputType, ComputeType>(x[idx]) * prstd
                            * Cast<ScaleType, ComputeType>(weight[i]);
        if(bias != nullptr)
        {
            y_val += Cast<ScaleType, ComputeType>(bias[i]);
        }
        y[idx] = Cast<ComputeType, OutputType>(y_val);
    }
}
