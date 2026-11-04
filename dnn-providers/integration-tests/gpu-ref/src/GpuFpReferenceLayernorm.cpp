// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn-gpu-ref/GpuFpReferenceLayernorm.hpp>

#include "hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp"
#include "hipdnn-gpu-ref/detail/GpuRefLaunch.hpp"

namespace hipdnn_gpu_ref
{

namespace
{

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefLayernormArgs.h> // NOLINT(misc-include-cleaner)

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

    detail::launchKernel1d(kernel.function(), outerSize * stride, localSize, &args, sizeof(args));
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

    detail::launchKernel1d(kernel.function(), outerSize * stride, localSize, &args, sizeof(args));
    detail::launchKernel1d(
        kernelWeights.function(), outerSize * stride, localSize, &argsWeights, sizeof(argsWeights));
}

} // namespace hipdnn_gpu_ref
