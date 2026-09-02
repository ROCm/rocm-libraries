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

void GpuFpReferenceReduction::launchReduction(
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode mode,
    const void* inputPtr,
    const std::vector<int64_t>& inputDims,
    const std::vector<int64_t>& inputStrides,
    void* outputPtr,
    const std::vector<int64_t>& outputDims,
    const std::vector<int64_t>& outputStrides,
    std::vector<std::string>& defines)
{
    ReductionMode reductionMode;
    switch(mode)
    {
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD:
        reductionMode = ReductionMode::ADD;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::AVG:
        reductionMode = ReductionMode::AVG;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::AMAX:
        reductionMode = ReductionMode::AMAX;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::NORM1:
        reductionMode = ReductionMode::NORM1;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::NORM2:
        reductionMode = ReductionMode::NORM2;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MUL:
        reductionMode = ReductionMode::MUL;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MUL_NO_ZEROS:
        reductionMode = ReductionMode::MUL_NO_ZEROS;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MIN_OP:
        reductionMode = ReductionMode::MIN_OP;
        break;
    case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MAX_OP:
        reductionMode = ReductionMode::MAX_OP;
        break;
    default:
        throw std::invalid_argument("Unsupported reduction mode: "
                                    + std::to_string(static_cast<int>(mode)));
    }
    defines.emplace_back(std::string("-DMODE=") + std::to_string(static_cast<int>(reductionMode)));

    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefReduction.cpp", defines, "ReductionRef");

    ReductionArgs args{};
    args.input = inputPtr;
    args.output = outputPtr;
    std::memcpy(args.inputStrides, inputStrides.data(), inputStrides.size() * sizeof(int64_t));
    std::memcpy(args.outputStrides, outputStrides.data(), outputStrides.size() * sizeof(int64_t));

    // Compute the output logical strides for unflattening the output index in the kernel
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        args.outputLogicalStrides[i] = 1;
        for(size_t j = i + 1; j < outputDims.size(); ++j)
        {
            args.outputLogicalStrides[i] *= static_cast<long long>(outputDims[j]);
        }
    }

    // Compute the reduction domain axes, strides and size based on the input and output dimensions
    long long reductionDomainShape[5];
    long long reductionDomainSize = 1;
    long long reductionRank = 0;
    for(size_t i = 0; i < outputDims.size(); ++i)
    {
        if(outputDims[i] == 1 && inputDims[i] > 1)
        {
            args.reductionDomainAxes[reductionRank] = static_cast<long long>(i);
            reductionDomainShape[reductionRank] = static_cast<long long>(inputDims[i]);
            reductionDomainSize *= static_cast<long long>(inputDims[i]);
            ++reductionRank;
        }
    }
    args.reductionRank = reductionRank;
    args.reductionDomainSize = reductionDomainSize;

    if(reductionRank > 0)
    {
        args.reductionDomainStride[reductionRank - 1] = 1;
        for(long long i = reductionRank - 2; i >= 0; --i)
        {
            args.reductionDomainStride[i]
                = args.reductionDomainStride[i + 1] * reductionDomainShape[i + 1];
        }
    }

    launchKernel(
        kernel.function(),
        std::accumulate(outputDims.begin(), outputDims.end(), int64_t{1}, std::multiplies<>()),
        &args,
        sizeof(args));
}

} // namespace hipdnn_gpu_ref
