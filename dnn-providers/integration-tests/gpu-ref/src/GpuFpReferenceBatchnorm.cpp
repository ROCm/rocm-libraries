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

bool isChannelLastLayout(const std::vector<int64_t>& strides)
{
    if(strides.size() < 3)
    {
        throw std::invalid_argument(
            "Batchnorm forward requires tensor rank to be at least 3 for layout validation.");
    }

    const auto strideOrder = hipdnn_data_sdk::utilities::extractStrideOrder(strides);
    return strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NLC.strideOrder
           || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NHWC.strideOrder
           || strideOrder == hipdnn_data_sdk::utilities::TensorLayout::NDHWC.strideOrder;
}

struct BatchnormLaunchGeometry
{
    int64_t c, hw, batchSize, cStride, hwStride, batchStride;
    std::array<unsigned int, 3> localSize, gridSize;
};

BatchnormLaunchGeometry computeFwdInfGeometry(const std::vector<int64_t>& dims,
                                              const std::vector<int64_t>& strides,
                                              std::vector<std::string>& defines)
{

    BatchnormLaunchGeometry geometry{};
    geometry.batchSize = dims[0];
    geometry.c = dims[1];
    geometry.batchStride = strides[0];
    geometry.cStride = strides[1];
    int64_t h = 0;
    int64_t w = 0;
    int64_t wStride = 0;

    if(dims.size() == 3)
    {
        h = dims[2];
        w = 1;
        wStride = strides[2];
    }
    else if(dims.size() == 4)
    {
        h = dims[2];
        w = dims[3];
        wStride = strides[3];
    }
    else if(dims.size() == 5)
    {
        // For 5D, combine D*H*W into spatial dimension
        auto d = dims[2];
        h = d * dims[3];
        w = dims[4];
        wStride = strides[4];
    }
    geometry.hw = h * w;
    geometry.hwStride = wStride;

    const bool isLayoutNhwc = isChannelLastLayout(strides);
    std::tie(geometry.localSize, geometry.gridSize)
        = calculateGrid(geometry.c, geometry.hw, geometry.batchSize, isLayoutNhwc);

    defines.emplace_back(std::string("-DLOCAL_SIZE_X=") + std::to_string(geometry.localSize[0]));
    defines.emplace_back(std::string("-DLOCAL_SIZE_Y=") + std::to_string(geometry.localSize[1]));
    return geometry;
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
    auto geometry = computeFwdInfGeometry(inputDims, inputStrides, defines);
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefBatchnormFwdInf.cpp", defines, "BatchnormFwdInfRef");

    BatchnormFwdInfArgs args{};
    args.common.input = inputPtr;
    args.common.scale = scalePtr;
    args.common.bias = biasPtr;
    args.common.estMean = estMeanPtr;
    args.common.output = outputPtr;
    args.common.c = static_cast<long long>(geometry.c);
    args.common.hw = static_cast<long long>(geometry.hw);
    args.common.batchSize = static_cast<long long>(geometry.batchSize);
    args.common.cStride = static_cast<long long>(geometry.cStride);
    args.common.hwStride = static_cast<long long>(geometry.hwStride);
    args.common.batchStride = static_cast<long long>(geometry.batchStride);
    args.invVar = invVarPtr;

    launchKernel(kernel.function(), geometry.localSize, geometry.gridSize, &args, sizeof(args));
}

void GpuFpReferenceBatchnorm::launchFwdInfWithVar(const void* inputPtr,
                                                  const std::vector<int64_t>& inputDims,
                                                  const std::vector<int64_t>& inputStrides,
                                                  const void* scalePtr,
                                                  const void* biasPtr,
                                                  const void* estMeanPtr,
                                                  const void* estVarPtr,
                                                  void* outputPtr,
                                                  double epsilon,
                                                  std::vector<std::string>& defines)
{
    auto geometry = computeFwdInfGeometry(inputDims, inputStrides, defines);
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefBatchnormFwdInf.cpp", defines, "BatchnormFwdInfWithVarRef");

    BatchnormFwdInfWithVarArgs args{};
    args.common.input = inputPtr;
    args.common.scale = scalePtr;
    args.common.bias = biasPtr;
    args.common.estMean = estMeanPtr;
    args.common.output = outputPtr;
    args.common.c = static_cast<long long>(geometry.c);
    args.common.hw = static_cast<long long>(geometry.hw);
    args.common.batchSize = static_cast<long long>(geometry.batchSize);
    args.common.cStride = static_cast<long long>(geometry.cStride);
    args.common.hwStride = static_cast<long long>(geometry.hwStride);
    args.common.batchStride = static_cast<long long>(geometry.batchStride);
    args.estVar = estVarPtr;
    args.epsilon = epsilon;

    launchKernel(kernel.function(), geometry.localSize, geometry.gridSize, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
