// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn-gpu-ref/GpuFpReferenceReduction.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefReductionArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function, int64_t gridSize, void* argsPtr, size_t argsSize)
{
    // Check the device limits for grid size
    int deviceId;
    detail::throwOnHipError(hipGetDevice(&deviceId), "hipGetDevice failed");
    hipDeviceProp_t deviceProps;
    detail::throwOnHipError(hipGetDeviceProperties(&deviceProps, deviceId),
                            "hipGetDeviceProperties failed");
    const int64_t maxGridSize = static_cast<int64_t>(deviceProps.maxGridSize[0])
                                / static_cast<int64_t>(GpuFpReferenceReduction::BLOCK_SIZE);
    if(gridSize > maxGridSize)
    {
        throw std::runtime_error("Grid size exceeds device limit: " + std::to_string(gridSize)
                                 + " > " + std::to_string(maxGridSize));
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(gridSize),
                                                  1,
                                                  1,
                                                  GpuFpReferenceReduction::BLOCK_SIZE,
                                                  1,
                                                  1,
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

} // namespace

// --- Kernel launchers ---

void GpuFpReferenceReduction::launchReduction(const void* inputPtr,
                                              const std::vector<int64_t>& inputDims,
                                              const std::vector<int64_t>& inputStrides,
                                              void* outputPtr,
                                              const std::vector<int64_t>& outputDims,
                                              const std::vector<int64_t>& outputStrides,
                                              const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefReduction.cpp", defines, "ReductionRef");

    std::vector<long long> inputStridesLL(inputStrides.begin(), inputStrides.end());
    std::vector<long long> outputStridesLL(outputStrides.begin(), outputStrides.end());
    std::vector<long long> outputShapeStrides(outputDims.size());
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        outputShapeStrides[i] = 1;
        for(size_t j = i + 1; j < outputDims.size(); ++j)
        {
            outputShapeStrides[i] *= outputDims[j];
        }
    }

    ReductionArgs args{};
    args.input = inputPtr;
    args.inputStrides = inputStridesLL.data();
    args.output = outputPtr;
    args.outputStrides = outputStridesLL.data();
    args.outputShapeStrides = outputShapeStrides.data();

    // Compute the reduction domain axes, strides and size based on the input and output dimensions
    std::vector<long long> reductionDomainAxes;
    std::vector<long long> reductionDomainShape;
    long long reductionDomainSize = 1;
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        if(outputDims[i] == 1 && inputDims[i] > 1)
        {
            reductionDomainAxes.push_back(static_cast<long long>(i));
            reductionDomainShape.push_back(inputDims[i]);
            reductionDomainSize *= inputDims[i];
        }
    }

    std::vector<long long> reductionDomainStride(reductionDomainAxes.size());
    if(!reductionDomainStride.empty())
    {
        reductionDomainStride.back() = 1;
        for(size_t i = 1; i < reductionDomainStride.size(); ++i)
        {
            size_t j = reductionDomainStride.size() - 1 - i;
            reductionDomainStride[j] = reductionDomainStride[j + 1] * reductionDomainShape[j + 1];
        }
    }

    args.reductionRank = static_cast<long long>(reductionDomainAxes.size());
    args.reductionDomainSize = reductionDomainSize;
    args.reductionDomainAxes = reductionDomainAxes.data();
    args.reductionDomainStride = reductionDomainStride.data();

    launchKernel(
        kernel.function(),
        std::accumulate(outputDims.begin(), outputDims.end(), int64_t{1}, std::multiplies<>()),
        &args,
        sizeof(args));
}

} // namespace hipdnn_gpu_ref
