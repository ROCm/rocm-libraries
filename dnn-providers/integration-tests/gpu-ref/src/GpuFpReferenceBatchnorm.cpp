// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <array>
#include <hipdnn-gpu-ref/GpuFpReferenceBatchnorm.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefBatchnormArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function,
                  std::array<unsigned int, 3> localSize,
                  std::array<unsigned int, 3> gridSize,
                  void* argsPtr,
                  size_t argsSize)
{
    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  gridSize[0],
                                                  gridSize[1],
                                                  gridSize[2],
                                                  localSize[0],
                                                  localSize[1],
                                                  localSize[2],
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

inline unsigned int checkedNarrowToUInt(int64_t value)
{
    if(value < static_cast<int64_t>(std::numeric_limits<unsigned int>::min())
       || value > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error(" value " + std::to_string(value) + " exceeds unsigned int range");
    }
    return static_cast<unsigned int>(value);
}

std::pair<std::array<unsigned int, 3>, std::array<unsigned int, 3>>
    calculateGrid(int64_t c, int64_t inCstride, int64_t n, bool isLayoutNhwc)
{
    std::array<unsigned int, 3> localSize;
    std::array<unsigned int, 3> gridSize;
    const unsigned int maxLocalsize = 256;
    const unsigned int cUint = checkedNarrowToUInt(c);
    const unsigned int cStrideUint = checkedNarrowToUInt(inCstride);
    const unsigned int nUint = checkedNarrowToUInt(n);

    if(isLayoutNhwc)
    {
        localSize[0] = std::min(cUint, maxLocalsize);
        localSize[1] = maxLocalsize / localSize[0];
    }
    else
    {
        localSize[0] = 1;
        localSize[1] = maxLocalsize;
    }
    gridSize[0] = cUint / localSize[0] + static_cast<unsigned int>(cUint % localSize[0] != 0);
    gridSize[1]
        = cStrideUint / localSize[1] + static_cast<unsigned int>(cStrideUint % localSize[1] != 0);

    // Check the device limits for grid size
    int deviceId;
    detail::throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");
    hipDeviceProp_t deviceProps;
    detail::throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                            "hipGetDeviceProperties failed");

    localSize[2] = 1;
    const uint64_t activeThreadsXy
        = static_cast<uint64_t>(gridSize[0]) * static_cast<uint64_t>(gridSize[1])
          * static_cast<uint64_t>(localSize[0]) * static_cast<uint64_t>(localSize[1]);
    const uint64_t maxActiveThreads = static_cast<uint64_t>(deviceProps.multiProcessorCount) * 32
                                      * static_cast<uint64_t>(deviceProps.warpSize);

    if(activeThreadsXy < maxActiveThreads)
    {
        gridSize[2]
            = std::min(static_cast<unsigned int>(maxActiveThreads / activeThreadsXy), nUint);
    }
    else
    {
        gridSize[2] = 1;
    }

    return std::make_pair(localSize, gridSize);
}

} // namespace

// --- Kernel launchers ---

void GpuFpReferenceBatchnorm::launchFwdInf(const void* inputPtr,
                                           const std::vector<int64_t>& inputDims,
                                           const std::vector<int64_t>& inputStrides,
                                           const void* scalePtr,
                                           const void* biasPtr,
                                           const void* estMeanPtr,
                                           const void* invVarPtr,
                                           void* outputPtr,
                                           std::vector<std::string>& defines)
{
    const int64_t n = inputDims[0];
    const int64_t c = inputDims[1];
    const int64_t nStride = inputStrides[0];
    const int64_t cStride = inputStrides[1];
    int64_t h = 0;
    int64_t w = 0;
    int64_t wStride = 0;

    if(inputDims.size() == 3)
    {
        h = inputDims[2];
        w = 1;
        wStride = inputStrides[2];
    }
    else if(inputDims.size() == 4)
    {
        h = inputDims[2];
        w = inputDims[3];
        wStride = inputStrides[3];
    }
    else if(inputDims.size() == 5)
    {
        // For 5D, combine D*H*W into spatial dimension
        auto d = inputDims[2];
        h = d * inputDims[3];
        w = inputDims[4];
        wStride = inputStrides[4];
    }

    const int64_t inCstride = h * w;
    const bool isLayoutNhwc = isChannelLastLayout(inputStrides);
    auto [localSize, gridSize] = calculateGrid(c, inCstride, n, isLayoutNhwc);

    defines.emplace_back(std::string("-DLOCAL_SIZE_X=") + std::to_string(localSize[0]));
    defines.emplace_back(std::string("-DLOCAL_SIZE_Y=") + std::to_string(localSize[1]));
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefBatchnormFwdInf.cpp", defines, "BatchnormFwdInfRef");

    BatchnormFwdArgs args{};
    args.input = inputPtr;
    args.scale = scalePtr;
    args.bias = biasPtr;
    args.estMean = estMeanPtr;
    args.invVar = invVarPtr;
    args.output = outputPtr;
    args.c = static_cast<long long>(c);
    args.hw = static_cast<long long>(inCstride);
    args.batchSize = static_cast<long long>(n);
    args.cStride = static_cast<long long>(cStride);
    args.hwStride = static_cast<long long>(wStride);
    args.batchStride = static_cast<long long>(nStride);

    launchKernel(kernel.function(), localSize, gridSize, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
