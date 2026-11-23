// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefHelpers.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref::detail
{

unsigned int checkedNarrowToUInt(int64_t value, const char* what)
{
    if(value < static_cast<int64_t>(std::numeric_limits<unsigned int>::min())
       || value > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error(std::string(what) + " value " + std::to_string(value)
                                 + " exceeds unsigned int range");
    }
    return static_cast<unsigned int>(value);
}

void launchKernel(hipFunction_t function,
                  std::array<int64_t, 3> gridSize,
                  std::array<int64_t, 3> blockSize,
                  void* argsPtr,
                  size_t argsSize)
{
    const unsigned int xGridSize = checkedNarrowToUInt(gridSize[0], "X grid size");
    const unsigned int yGridSize = checkedNarrowToUInt(gridSize[1], "Y grid size");
    const unsigned int zGridSize = checkedNarrowToUInt(gridSize[2], "Z grid size");
    const unsigned int xBlockSize = checkedNarrowToUInt(blockSize[0], "X block size");
    const unsigned int yBlockSize = checkedNarrowToUInt(blockSize[1], "Y block size");
    const unsigned int zBlockSize = checkedNarrowToUInt(blockSize[2], "Z block size");

    // Check the device limits for grid size
    assertValidGridSize(gridSize[0], gridSize[1], gridSize[2]);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    throwOnHipError(hipModuleLaunchKernel(function,
                                          xGridSize,
                                          yGridSize,
                                          zGridSize,
                                          xBlockSize,
                                          yBlockSize,
                                          zBlockSize,
                                          0,
                                          nullptr,
                                          nullptr,
                                          config),
                    "hipModuleLaunchKernel failed");

    throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

void launchKernel1d(
    hipFunction_t function, int64_t gridSize, int64_t blockSize, void* argsPtr, size_t argsSize)
{
    launchKernel(function, {gridSize, 1, 1}, {blockSize, 1, 1}, argsPtr, argsSize);
}

void launchKernelForElements(hipFunction_t function,
                             int64_t totalElements,
                             void* argsPtr,
                             size_t argsSize,
                             int64_t blockSize)
{
    if(blockSize <= 0)
    {
        throw std::runtime_error("block size must be positive");
    }

    if(totalElements <= 0)
    {
        throw std::runtime_error("total elements must be positive");
    }

    const int64_t gridSize = ((totalElements - 1) / blockSize) + 1;
    launchKernel1d(function, gridSize, blockSize, argsPtr, argsSize);
}

} // namespace hipdnn_gpu_ref::detail
