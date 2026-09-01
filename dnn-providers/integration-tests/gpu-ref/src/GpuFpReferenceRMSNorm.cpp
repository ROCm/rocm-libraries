// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn-gpu-ref/detail/GpuRefHelpers.hpp"
#include <hipdnn-gpu-ref/GpuFpReferenceRMSNorm.hpp>

#include <hipdnn-gpu-ref/detail/GpuRefHipError.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefRMSNormArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function, int64_t gridSize, void* argsPtr, size_t argsSize)
{
    // Check the device limits for grid size
    detail::assertValidGridSize(
        gridSize, 1, 1, static_cast<int64_t>(GpuFpReferenceRMSNorm::BLOCK_SIZE), 1, 1);

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
                                                  GpuFpReferenceRMSNorm::BLOCK_SIZE,
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

void GpuFpReferenceRMSNorm::launchFprop(const void* inputPtr,
                                        const std::vector<int64_t>& inputDims,
                                        const std::vector<int64_t>& inputStrides,
                                        const void* scalePtr,
                                        const std::vector<int64_t>& scaleDims,
                                        void* outputPtr,
                                        const std::vector<std::string>& defines,
                                        void* invRmsPtr,
                                        const void* biasPtr,
                                        double epsilon)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefRMSNormFwd.cpp", defines, "RMSNormFwdRef");

    const auto& normalizeDim = getNormalizeDim(inputDims, scaleDims);
    const auto& stride = getStride(inputDims, inputStrides, normalizeDim);
    const auto& outerSize = getOuterSize(inputDims, normalizeDim, stride);
    const auto& innerSize = getInnerSize(inputDims, normalizeDim);

    RMSNormFwdArgs args{};
    args.input = inputPtr;
    args.scale = scalePtr;
    args.bias = biasPtr;
    args.output = outputPtr;
    args.invRms = invRmsPtr;
    args.innerSize = static_cast<long long>(innerSize);
    args.stride = static_cast<long long>(stride);
    args.eps = epsilon;

    launchKernel(kernel.function(), outerSize * stride, &args, sizeof(args));
}

void GpuFpReferenceRMSNorm::launchDgrad(const void* gradOutputPtr,
                                        const void* inputPtr,
                                        const std::vector<int64_t>& inputDims,
                                        const std::vector<int64_t>& inputStrides,
                                        const void* scalePtr,
                                        const std::vector<int64_t>& scaleDims,
                                        const void* invRmsPtr,
                                        void* gradInputPtr,
                                        const std::vector<std::string>& defines)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefRMSNormBwd.cpp", defines, "RMSNormBwdDataRef");

    const auto& normalizeDim = getNormalizeDim(inputDims, scaleDims);
    const auto& stride = getStride(inputDims, inputStrides, normalizeDim);
    const auto& outerSize = getOuterSize(inputDims, normalizeDim, stride);
    const auto& innerSize = getInnerSize(inputDims, normalizeDim);

    RMSNormBwdArgs args{};
    args.gradOutput = gradOutputPtr;
    args.input = inputPtr;
    args.scale = scalePtr;
    args.invRms = invRmsPtr;
    args.gradInput = gradInputPtr;
    args.gradScale = nullptr;
    args.gradBias = nullptr;
    args.innerSize = static_cast<long long>(innerSize);
    args.outerSize = static_cast<long long>(outerSize);
    args.stride = static_cast<long long>(stride);

    launchKernel(kernel.function(), outerSize * stride, &args, sizeof(args));
}

void GpuFpReferenceRMSNorm::launchWgrad(const void* gradOutputPtr,
                                        const void* inputPtr,
                                        const std::vector<int64_t>& inputDims,
                                        const std::vector<int64_t>& inputStrides,
                                        const std::vector<int64_t>& scaleDims,
                                        const void* invRmsPtr,
                                        void* gradScalePtr,
                                        const std::vector<std::string>& defines,
                                        void* gradBiasPtr)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel
        = compiler.getOrCompile("GpuRefRMSNormBwd.cpp", defines, "RMSNormBwdWeightBiasRef");

    const auto& normalizeDim = getNormalizeDim(inputDims, scaleDims);
    const auto& stride = getStride(inputDims, inputStrides, normalizeDim);
    const auto& outerSize = getOuterSize(inputDims, normalizeDim, stride);
    const auto& innerSize = getInnerSize(inputDims, normalizeDim);

    RMSNormBwdArgs args{};
    args.gradOutput = gradOutputPtr;
    args.input = inputPtr;
    args.scale = nullptr;
    args.invRms = invRmsPtr;
    args.gradInput = nullptr;
    args.gradScale = gradScalePtr;
    args.gradBias = gradBiasPtr;
    args.innerSize = static_cast<long long>(innerSize);
    args.outerSize = static_cast<long long>(outerSize);
    args.stride = static_cast<long long>(stride);

    launchKernel(kernel.function(), (innerSize + BLOCK_SIZE - 1) / BLOCK_SIZE, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
