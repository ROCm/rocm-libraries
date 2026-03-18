// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "device_prng.hpp"
#include "gtest_hip_utilities.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

namespace test::gtest {

template <typename T>
__global__ void philox_kernel(T* out, size_t n, uint64_t seed)
{
    const size_t tid    = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for(size_t i = tid; i < n; i += stride)
    {
        const uint64_t raw = philox_u64(seed, static_cast<uint64_t>(i));

        if constexpr(std::is_same_v<T, float>)
        {
            // raw >> 40 keeps the top 24 bits (FP32 mantissa size).
            // 1.0f / 16777216.0f is 2^-24, normalizing the integer to [0, 1).
            out[i] = static_cast<float>(raw >> 40) * (1.0f / 16777216.0f);
        }
        else if constexpr(std::is_same_v<T, double>)
        {
            // raw >> 11 keeps the top 53 bits (FP64 mantissa size).
            // 1.0 / 9007... is 2^-53, normalizing the integer to [0, 1).
            out[i] = static_cast<double>(raw >> 11) * (1.0 / 9007199254740992.0);
        }
        else if constexpr(std::is_same_v<T, hip_bfloat16>)
        {
            // raw >> 57 keeps top 7 bits (BFloat16 mantissa size).
            // 1.0f / 128.0f is 2^-7.
            const float f_val = static_cast<float>(raw >> 57) * (1.0f / 128.0f);
            out[i]            = static_cast<T>(f_val);
        }
        else if constexpr(std::is_same_v<T, __half>)
        {
            // raw >> 54 keeps top 10 bits (FP16 mantissa size).
            // 1.0f / 1024.0f is 2^-10.
            const float f_val = static_cast<float>(raw >> 54) * (1.0f / 1024.0f);
            out[i]            = static_cast<T>(f_val);
        }
        else if constexpr(std::is_integral_v<T>)
        {
            constexpr int shift = 64 - (sizeof(T) * 8);
            out[i]              = static_cast<T>(raw >> shift);
        }
        else
        {
            static_assert(false, "Unsupported data type for philox_kernel");
        }
    }
}

template <typename T>
void RandomizeBuffer(T* dev_ptr, size_t size, uint64_t seed, hipStream_t stream)
{
    if(size == 0)
        return;

    constexpr int threadsPerBlock = 256;
    const int numBlocks           = ComputeNumBlocks<T>(size);

    using DeviceT = typename ToDeviceType<T>::type;
    philox_kernel<DeviceT><<<numBlocks, threadsPerBlock, 0, stream>>>(
        reinterpret_cast<DeviceT*>(dev_ptr), size, seed);

    MIOPEN_GTEST_HIP_ERROR(hipGetLastError(), "philox_kernel launch failed");
}

template void RandomizeBuffer<float>(float*, size_t, uint64_t, hipStream_t);
template void RandomizeBuffer<double>(double*, size_t, uint64_t, hipStream_t);
template void RandomizeBuffer<half_float::half>(half_float::half*, size_t, uint64_t, hipStream_t);
template void RandomizeBuffer<bfloat16>(bfloat16*, size_t, uint64_t, hipStream_t);
template void RandomizeBuffer<int8_t>(int8_t*, size_t, uint64_t, hipStream_t);
template void RandomizeBuffer<int32_t>(int32_t*, size_t, uint64_t, hipStream_t);

} // namespace test::gtest
