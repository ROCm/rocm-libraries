// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <hip/hip_runtime.h>

namespace test_gpu_utils
{

constexpr int64_t UID_A = 1;
constexpr int64_t UID_B = 2;
constexpr int64_t UID_BIAS = 3;
constexpr int64_t UID_C = 4;

struct HipFreeDeleter
{
    void operator()(void* ptr) const
    {
        // Best-effort release; a failed free cannot be acted on here.
        std::ignore = hipFree(ptr);
    }
};

using DevicePtr = std::unique_ptr<void, HipFreeDeleter>;

inline DevicePtr deviceAlloc(size_t bytes)
{
    void* ptr = nullptr;
    if(hipMalloc(&ptr, bytes) != hipSuccess)
    {
        return nullptr;
    }
    return DevicePtr(ptr);
}

// Uniform values in [-1, 1) from a fixed seed.
inline std::vector<float> randomUniform(size_t count, unsigned seed)
{
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> values(count);
    for(auto& v : values)
    {
        v = dist(gen);
    }
    return values;
}

} // namespace test_gpu_utils
