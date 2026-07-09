// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn_gpu_ref/GpuFpReferenceRMSNorm.hpp>

#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefRMSNormArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function, int64_t outerSize, void* argsPtr, size_t argsSize)
{
    if(outerSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
    {
        throw std::runtime_error("Grid size exceeds hipModuleLaunchKernel limit");
    }

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(outerSize),
                                                  1,
                                                  1,
                                                  GpuFpReferenceRMSNorm::blockSize,
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
                                        const void* scalePtr,
                                        const std::vector<int64_t>& scaleDims,
                                        void* outputPtr,
                                        const std::vector<std::string>& defines,
                                        const void* biasPtr,
                                        void* invRmsPtr,
                                        double epsilon)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    const auto& kernel = compiler.getOrCompile("GpuRefRMSNormFwd.cpp", defines, "RMSNormFwdRef");

    const auto& normalizeDim = getNormalizeDim(inputDims, scaleDims);
    const auto& stride = getStride(inputDims, normalizeDim);
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

} // namespace hipdnn_gpu_ref
