// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_gpu_ref/GpuFpReferenceConvolution.hpp>

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
#include <GpuRefConvArgs.h> // NOLINT(misc-include-cleaner)

Strides3 toStrides3(const std::vector<int64_t>& strides)
{
    Strides3 result{};
    for(size_t i = 0; i < 3 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

Strides4 toStrides4(const std::vector<int64_t>& strides)
{
    Strides4 result{};
    for(size_t i = 0; i < 4 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

Strides5 toStrides5(const std::vector<int64_t>& strides)
{
    Strides5 result{};
    for(size_t i = 0; i < 5 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

void launchKernel(hipFunction_t function, int64_t totalElements, void* argsPtr, size_t argsSize)
{
    const int64_t blockSize = 256;
    auto gridSize = (totalElements + blockSize - 1) / blockSize;

    if(gridSize > static_cast<int64_t>(std::numeric_limits<unsigned int>::max()))
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
                                                  static_cast<unsigned int>(gridSize),
                                                  1,
                                                  1,
                                                  static_cast<unsigned int>(blockSize),
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

// --- 1D kernel launcher ---

void GpuFpReferenceConvolution::launchFprop1d(const void* xPtr,
                                              const void* wPtr,
                                              void* yPtr,
                                              const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& wDims,
                                              const std::vector<int64_t>& yDims,
                                              const std::vector<int64_t>& xTensorStrides,
                                              const std::vector<int64_t>& wTensorStrides,
                                              const std::vector<int64_t>& yTensorStrides,
                                              const std::vector<int64_t>& convStrides,
                                              const std::vector<int64_t>& dilations,
                                              const std::vector<int64_t>& padding,
                                              const std::vector<std::string>& defines,
                                              double alpha,
                                              double beta)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef1d");

    auto nGroups = xDims[1] / wDims[1];

    ConvFwdArgs1d args{};
    args.x = xPtr;
    args.w = wPtr;
    args.y = yPtr;
    args.xStr = toStrides3(xTensorStrides);
    args.wStr = toStrides3(wTensorStrides);
    args.yStr = toStrides3(yTensorStrides);
    args.N = static_cast<long long>(xDims[0]);
    args.C = static_cast<long long>(xDims[1]);
    args.Wi = static_cast<long long>(xDims[2]);
    args.K = static_cast<long long>(wDims[0]);
    args.Wo = static_cast<long long>(yDims[2]);
    args.Kw = static_cast<long long>(wDims[2]);
    args.strideW = static_cast<long long>(convStrides[0]);
    args.dilW = static_cast<long long>(dilations[0]);
    args.padW = static_cast<long long>(padding[0]);
    args.groups = static_cast<long long>(nGroups);
    args.alpha = alpha;
    args.beta = beta;

    auto totalElements = xDims[0] * wDims[0] * yDims[2];
    launchKernel(kernel.function(), totalElements, &args, sizeof(args));
}

// --- 2D kernel launcher ---

void GpuFpReferenceConvolution::launchFprop2d(const void* xPtr,
                                              const void* wPtr,
                                              void* yPtr,
                                              const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& wDims,
                                              const std::vector<int64_t>& yDims,
                                              const std::vector<int64_t>& xTensorStrides,
                                              const std::vector<int64_t>& wTensorStrides,
                                              const std::vector<int64_t>& yTensorStrides,
                                              const std::vector<int64_t>& convStrides,
                                              const std::vector<int64_t>& dilations,
                                              const std::vector<int64_t>& padding,
                                              const std::vector<std::string>& defines,
                                              double alpha,
                                              double beta)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef2d");

    auto nGroups = xDims[1] / wDims[1];

    ConvFwdArgs2d args{};
    args.x = xPtr;
    args.w = wPtr;
    args.y = yPtr;
    args.xStr = toStrides4(xTensorStrides);
    args.wStr = toStrides4(wTensorStrides);
    args.yStr = toStrides4(yTensorStrides);
    args.N = static_cast<long long>(xDims[0]);
    args.C = static_cast<long long>(xDims[1]);
    args.Hi = static_cast<long long>(xDims[2]);
    args.Wi = static_cast<long long>(xDims[3]);
    args.K = static_cast<long long>(wDims[0]);
    args.Ho = static_cast<long long>(yDims[2]);
    args.Wo = static_cast<long long>(yDims[3]);
    args.Kh = static_cast<long long>(wDims[2]);
    args.Kw = static_cast<long long>(wDims[3]);
    args.strideH = static_cast<long long>(convStrides[0]);
    args.strideW = static_cast<long long>(convStrides[1]);
    args.dilH = static_cast<long long>(dilations[0]);
    args.dilW = static_cast<long long>(dilations[1]);
    args.padH = static_cast<long long>(padding[0]);
    args.padW = static_cast<long long>(padding[1]);
    args.groups = static_cast<long long>(nGroups);
    args.alpha = alpha;
    args.beta = beta;

    auto totalElements = xDims[0] * wDims[0] * yDims[2] * yDims[3];
    launchKernel(kernel.function(), totalElements, &args, sizeof(args));
}

// --- 3D kernel launcher ---

void GpuFpReferenceConvolution::launchFprop3d(const void* xPtr,
                                              const void* wPtr,
                                              void* yPtr,
                                              const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& wDims,
                                              const std::vector<int64_t>& yDims,
                                              const std::vector<int64_t>& xTensorStrides,
                                              const std::vector<int64_t>& wTensorStrides,
                                              const std::vector<int64_t>& yTensorStrides,
                                              const std::vector<int64_t>& convStrides,
                                              const std::vector<int64_t>& dilations,
                                              const std::vector<int64_t>& padding,
                                              const std::vector<std::string>& defines,
                                              double alpha,
                                              double beta)
{
    auto& compiler = detail::GpuRefKernelCompiler::instance();
    auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef3d");

    auto nGroups = xDims[1] / wDims[1];

    ConvFwdArgs3d args{};
    args.x = xPtr;
    args.w = wPtr;
    args.y = yPtr;
    args.xStr = toStrides5(xTensorStrides);
    args.wStr = toStrides5(wTensorStrides);
    args.yStr = toStrides5(yTensorStrides);
    args.N = static_cast<long long>(xDims[0]);
    args.C = static_cast<long long>(xDims[1]);
    args.Di = static_cast<long long>(xDims[2]);
    args.Hi = static_cast<long long>(xDims[3]);
    args.Wi = static_cast<long long>(xDims[4]);
    args.K = static_cast<long long>(wDims[0]);
    args.Do = static_cast<long long>(yDims[2]);
    args.Ho = static_cast<long long>(yDims[3]);
    args.Wo = static_cast<long long>(yDims[4]);
    args.Kd = static_cast<long long>(wDims[2]);
    args.Kh = static_cast<long long>(wDims[3]);
    args.Kw = static_cast<long long>(wDims[4]);
    args.strideD = static_cast<long long>(convStrides[0]);
    args.strideH = static_cast<long long>(convStrides[1]);
    args.strideW = static_cast<long long>(convStrides[2]);
    args.dilD = static_cast<long long>(dilations[0]);
    args.dilH = static_cast<long long>(dilations[1]);
    args.dilW = static_cast<long long>(dilations[2]);
    args.padD = static_cast<long long>(padding[0]);
    args.padH = static_cast<long long>(padding[1]);
    args.padW = static_cast<long long>(padding[2]);
    args.groups = static_cast<long long>(nGroups);
    args.alpha = alpha;
    args.beta = beta;

    auto totalElements = xDims[0] * wDims[0] * yDims[2] * yDims[3] * yDims[4];
    launchKernel(kernel.function(), totalElements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
