// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHelpers.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>

#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace hipdnn_gpu_ref::detail
{

namespace
{

std::string withContext(const char* context, const char* message)
{
    return context == nullptr ? std::string(message) : std::string(context) + ": " + message;
}

void throwOnHipErrorWithContext(hipError_t err, const char* context, const char* message)
{
    if(err != hipSuccess)
    {
        throwOnHipError(err, withContext(context, message).c_str());
    }
}

} // namespace

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
                  size_t argsSize,
                  const char* context,
                  std::array<int64_t, 3> gridSizeDivisor)
{
    const unsigned int xGridSize = checkedNarrowToUInt(gridSize[0], "X grid size");
    const unsigned int yGridSize = checkedNarrowToUInt(gridSize[1], "Y grid size");
    const unsigned int zGridSize = checkedNarrowToUInt(gridSize[2], "Z grid size");
    const unsigned int xBlockSize = checkedNarrowToUInt(blockSize[0], "X block size");
    const unsigned int yBlockSize = checkedNarrowToUInt(blockSize[1], "Y block size");
    const unsigned int zBlockSize = checkedNarrowToUInt(blockSize[2], "Z block size");

    // Check the device limits for grid size
    assertValidGridSize(gridSize[0],
                        gridSize[1],
                        gridSize[2],
                        gridSizeDivisor[0],
                        gridSizeDivisor[1],
                        gridSizeDivisor[2]);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    throwOnHipErrorWithContext(hipModuleLaunchKernel(function,
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
                               context,
                               "hipModuleLaunchKernel failed");

    throwOnHipErrorWithContext(hipDeviceSynchronize(), context, "hipDeviceSynchronize failed");
}

void launchKernel1d(hipFunction_t function,
                    int64_t gridSize,
                    int64_t blockSize,
                    void* argsPtr,
                    size_t argsSize,
                    const char* context,
                    int64_t gridSizeDivisor)
{
    launchKernel(function,
                 {gridSize, 1, 1},
                 {blockSize, 1, 1},
                 argsPtr,
                 argsSize,
                 context,
                 {gridSizeDivisor, 1, 1});
}

void launchKernelForElements(hipFunction_t function,
                             int64_t totalElements,
                             void* argsPtr,
                             size_t argsSize,
                             const char* context,
                             int64_t blockSize)
{
    if(blockSize <= 0)
    {
        throw std::runtime_error(withContext(context, "block size must be positive"));
    }

    const int64_t gridSize = (totalElements + blockSize - 1) / blockSize;
    launchKernel1d(function, gridSize, blockSize, argsPtr, argsSize, context);
}

} // namespace hipdnn_gpu_ref::detail
