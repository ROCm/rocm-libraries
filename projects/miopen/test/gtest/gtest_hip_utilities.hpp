// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_GTEST_UTILS_HPP
#define GUARD_MIOPEN_GTEST_UTILS_HPP

#include "miopen/bfloat16.hpp"
#include "miopen/errors.hpp"
#include <half/half.hpp>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#include <cstddef>

#ifndef MIOPEN_GTEST_HIP_ERROR
#define MIOPEN_GTEST_HIP_ERROR(status, message)    \
    do                                             \
    {                                              \
        hipError_t err = status;                   \
        if(err != hipSuccess)                      \
        {                                          \
            MIOPEN_THROW_HIP_STATUS(err, message); \
        }                                          \
    } while(0)
#endif

namespace test::gtest {

// --- HELPER: Map Host Types to Device Types ---
// Default: T maps to T (float, int8_t, etc.)
template <typename T>
struct ToDeviceType
{
    using type = T;
};

// Map half_float::half -> __half
template <>
struct ToDeviceType<half_float::half>
{
    using type = __half;
};

// Map miopen::bfloat16 -> __bfloat16
template <>
struct ToDeviceType<bfloat16>
{
    using type = __hip_bfloat16;
};

template <typename T>
inline int ComputeNumBlocks(size_t work_items)
{
    int deviceId = 0;
    MIOPEN_GTEST_HIP_ERROR(hipGetDevice(&deviceId), "Failed to get device ID");

    hipDeviceProp_t props{};
    MIOPEN_GTEST_HIP_ERROR(hipGetDeviceProperties(&props, deviceId),
                           "Failed to get device properties");

    constexpr int threadsPerBlock = 256;
    int numBlocks                 = props.multiProcessorCount * 4;

    if(work_items == 0)
        return 0;

    if(static_cast<size_t>(numBlocks) * threadsPerBlock > work_items)
        numBlocks = static_cast<int>((work_items + threadsPerBlock - 1) / threadsPerBlock);

    return (numBlocks > 0) ? numBlocks : 1;
}

template <typename DeviceT>
__global__ void FillKernel(DeviceT* __restrict__ buffer, DeviceT value, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        buffer[idx] = value;
    }
}

template <typename T>
void gpu_fill(T* d_ptr, T value, size_t size_in_bytes, hipStream_t stream)
{
    if(value == T{0})
    {
        MIOPEN_GTEST_HIP_ERROR(hipMemsetAsync(d_ptr, 0, size_in_bytes, stream),
                               "Failed to memset device buffer");
    }
    else
    {
        int blockSize           = 256;
        const auto num_elements = size_in_bytes / sizeof(T);
        int numBlocks           = (num_elements + blockSize - 1) / blockSize;

        using DeviceT = typename ToDeviceType<T>::type;

        // We must convert the host value 'value' to the device type before passing to kernel
        DeviceT device_val = static_cast<DeviceT>(value);

        FillKernel<DeviceT><<<numBlocks, blockSize, 0, stream>>>(
            reinterpret_cast<DeviceT*>(d_ptr), device_val, num_elements);
    }
}

} // namespace test::gtest

#endif // GUARD_MIOPEN_GTEST_UTILS_HPP
