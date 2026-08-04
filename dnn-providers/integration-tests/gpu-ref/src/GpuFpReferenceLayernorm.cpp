// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hip/amd_detail/amd_hip_runtime.h>
#include <hipdnn-gpu-ref/GpuFpReferenceLayernorm.hpp>

#include "hipdnn-gpu-ref/detail/GpuRefHelpers.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefHipError.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp"

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefLayernormArgs.h> // NOLINT(misc-include-cleaner)

void launchKernel(hipFunction_t function,
                  int64_t outerSize,
                  int64_t stride,
                  int64_t localSize,
                  void* argsPtr,
                  size_t argsSize)
{
    const int64_t xlocalsize = localSize;
    const int64_t xgridsize = outerSize * stride;
    const int64_t ylocalsize = 1;
    const int64_t ygridsize = 1;
    const int64_t zlocalsize = 1;
    const int64_t zgridsize = 1;

    // Check the device limits for grid size
    detail::assertValidGridSize(xgridsize, ygridsize, zgridsize);

    // NOLINTNEXTLINE(modernize-avoid-c-arrays)
    void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                      argsPtr,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE,
                      &argsSize,
                      HIP_LAUNCH_PARAM_END};

    detail::throwOnHipError(hipModuleLaunchKernel(function,
                                                  static_cast<unsigned int>(xgridsize),
                                                  static_cast<unsigned int>(ygridsize),
                                                  static_cast<unsigned int>(zgridsize),
                                                  static_cast<unsigned int>(xlocalsize),
                                                  static_cast<unsigned int>(ylocalsize),
                                                  static_cast<unsigned int>(zlocalsize),
                                                  0,
                                                  nullptr,
                                                  nullptr,
                                                  config),
                            "hipModuleLaunchKernel failed");

    detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

} // namespace

void GpuFpReferenceLayernorm::launchFprop(const void* xPtr,
                                          const std::vector<int64_t>& xDims,
                                          const std::vector<int64_t>& xStrides,
                                          const void* scalePtr,
                                          const void* biasPtr,
                                          void* yPtr,
                                          void* meanPtr,
                                          void* rstdPtr,
                                          const int64_t normalizedDimCount,
                                          const std::vector<std::string>& defines,
                                          const int64_t localSize,
                                          const double epsilon)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefLayernormFwd.cpp", defines, "LayernormFwdRef");

    LayernormFwdArgs args{};
    args.x = xPtr;
    args.scale = scalePtr;
    args.bias = biasPtr;
    args.y = yPtr;
    args.mean = meanPtr;
    args.rstd = rstdPtr;
    args.epsilon = epsilon;

    int64_t outerSize;
    int64_t innerSize;
    int64_t stride;
    detail::getLayernormDimensions(
        outerSize, innerSize, stride, xDims, xStrides, normalizedDimCount);

    launchKernel(kernel.function(), outerSize, stride, localSize, &args, sizeof(args));
}

void GpuFpReferenceLayernorm::launchBprop(const void* dyPtr,
                                          const std::vector<int64_t>& dyDims,
                                          const std::vector<int64_t>& dyStrides,
                                          const void* xPtr,
                                          const void* scalePtr,
                                          void* dxPtr,
                                          void* dscalePtr,
                                          void* dbiasPtr,
                                          const void* meanPtr,
                                          const void* rstdPtr,
                                          void* workspace,
                                          const int64_t normalizedDimCount,
                                          const std::vector<std::string>& defines,
                                          const int64_t localSize,
                                          const double epsilon)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefLayernormBwd.cpp", defines, "LayernormBwdRef");
    auto& kernelWeights
        = compiler.getOrCompile("GpuRefLayernormBwd.cpp", defines, "LayernormBwdWeightsRef");

    LayernormBwdArgs args{};
    args.dy = dyPtr;
    args.x = xPtr;
    args.scale = scalePtr;
    args.dx = dxPtr;
    args.mean = meanPtr;
    args.rstd = rstdPtr;
    args.epsilon = epsilon;
    args.workspace = workspace;

    LayernormBwdWeightsArgs argsWeights{};
    argsWeights.dy = dyPtr;
    argsWeights.x = xPtr;
    argsWeights.dscale = dscalePtr;
    argsWeights.dbias = dbiasPtr;
    argsWeights.mean = meanPtr;
    argsWeights.rstd = rstdPtr;
    argsWeights.workspace = workspace;

    int64_t outerSize;
    int64_t innerSize;
    int64_t stride;
    detail::getLayernormDimensions(
        outerSize, innerSize, stride, dyDims, dyStrides, normalizedDimCount);

    launchKernel(kernel.function(), outerSize, stride, localSize, &args, sizeof(args));
    launchKernel(
        kernelWeights.function(), outerSize, stride, localSize, &argsWeights, sizeof(argsWeights));
}

} // namespace hipdnn_gpu_ref
