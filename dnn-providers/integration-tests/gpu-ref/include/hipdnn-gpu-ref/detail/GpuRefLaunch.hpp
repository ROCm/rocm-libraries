// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <hip/hip_runtime_api.h>

namespace hipdnn_gpu_ref::detail
{

// Block size used by the reference kernels that do not expose their own.
inline constexpr int64_t DEFAULT_BLOCK_SIZE = 256;

// Narrows a 64-bit launch dimension to the unsigned int expected by the HIP driver API,
// throwing a descriptive error when the value is out of range.
unsigned int checkedNarrowToUInt(int64_t value, const char* what = "value");

// Launches a kernel compiled by GpuRefKernelCompiler with an explicit 3D geometry.
//
// The kernel arguments are passed through the HIP_LAUNCH_PARAM_* buffer protocol, the grid
// dimensions are validated against the device limits (each dimension limit is divided by the
// matching entry of gridSizeDivisor) and the device is synchronized before returning.
// When context is not null it is used as a prefix for any error message.
void launchKernel(hipFunction_t function,
                  std::array<int64_t, 3> gridSize,
                  std::array<int64_t, 3> blockSize,
                  void* argsPtr,
                  size_t argsSize,
                  const char* context = nullptr,
                  std::array<int64_t, 3> gridSizeDivisor = {1, 1, 1});

// Launches a kernel with a 1D geometry of gridSize blocks of blockSize threads.
void launchKernel1d(hipFunction_t function,
                    int64_t gridSize,
                    int64_t blockSize,
                    void* argsPtr,
                    size_t argsSize,
                    const char* context = nullptr,
                    int64_t gridSizeDivisor = 1);

// Launches a kernel with a 1D geometry covering totalElements, one thread per element.
void launchKernelForElements(hipFunction_t function,
                             int64_t totalElements,
                             void* argsPtr,
                             size_t argsSize,
                             const char* context = nullptr,
                             int64_t blockSize = DEFAULT_BLOCK_SIZE);

} // namespace hipdnn_gpu_ref::detail
