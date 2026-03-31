// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefHipError.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>

#include <array>
#include <cstdint>
#include <hip/hip_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

// --- HipRTC type name mapping ---

template <typename T>
struct HipRtcTypeName;

template <>
struct HipRtcTypeName<float>
{
    static constexpr const char* VALUE = "float";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::half>
{
    static constexpr const char* VALUE = "_Float16";
};

template <>
struct HipRtcTypeName<hipdnn_data_sdk::types::bfloat16>
{
    static constexpr const char* VALUE = "unsigned short";
};

template <>
struct HipRtcTypeName<int8_t>
{
    static constexpr const char* VALUE = "signed char";
};

template <>
struct HipRtcTypeName<int32_t>
{
    static constexpr const char* VALUE = "int";
};

template <>
struct HipRtcTypeName<double>
{
    static constexpr const char* VALUE = "double";
};

// Shared argument and stride structs — single definition used by both host and device (HipRTC).
#include <GpuRefConvArgs.h> // NOLINT(misc-include-cleaner)

// --- Helpers ---

template <typename XDataType, typename WDataType, typename YDataType, typename ComputeDataType>
inline std::vector<std::string> buildConvDefines(bool useTf32 = false)
{
    std::vector<std::string> defines;
    defines.emplace_back(std::string("-DX_TYPE=") + HipRtcTypeName<XDataType>::VALUE);
    defines.emplace_back(std::string("-DW_TYPE=") + HipRtcTypeName<WDataType>::VALUE);
    defines.emplace_back(std::string("-DY_TYPE=") + HipRtcTypeName<YDataType>::VALUE);
    defines.emplace_back(std::string("-DCOMPUTE_TYPE=") + HipRtcTypeName<ComputeDataType>::VALUE);
    if(useTf32)
    {
        defines.emplace_back("-DUSE_TF32");
    }
    return defines;
}

inline Strides3 toStrides3(const std::vector<int64_t>& strides)
{
    Strides3 result{};
    for(size_t i = 0; i < 3 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

inline Strides4 toStrides4(const std::vector<int64_t>& strides)
{
    Strides4 result{};
    for(size_t i = 0; i < 4 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

inline Strides5 toStrides5(const std::vector<int64_t>& strides)
{
    Strides5 result{};
    for(size_t i = 0; i < 5 && i < strides.size(); ++i)
    {
        result.s[i] = static_cast<long long>(strides[i]);
    }
    return result;
}

inline void
    launchKernel(hipFunction_t function, int64_t totalElements, void* argsPtr, size_t argsSize)
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

    throwOnHipError(hipModuleLaunchKernel(function,
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

    throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");
}

} // namespace detail

class GpuFpReferenceConvolution
{
public:
    // --- Forward convolution (fprop) ---
    // x and w are non-const because MigratableMemory::deviceData() triggers
    // lazy host-to-device synchronization, which mutates internal state.

    // Overload for uniform padding
    template <class XDataType,
              class WDataType = XDataType,
              class YDataType = XDataType,
              class ComputeDataType = double>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding,
                      double alpha = 1.0,
                      double beta = 0.0,
                      bool useTf32 = false)
    {
        fprop<XDataType, WDataType, YDataType, ComputeDataType>(
            x, w, y, convStrides, dilations, padding, padding, alpha, beta, useTf32);
    }

    template <class XDataType,
              class WDataType = XDataType,
              class YDataType = XDataType,
              class ComputeDataType = double>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& convStrides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding,
                      double alpha = 1.0,
                      double beta = 0.0,
                      bool useTf32 = false)
    {
        validateInput(x, w, y, convStrides, dilations, prePadding, postPadding);

        const auto nDims = x.dims().size();
        auto defines
            = detail::buildConvDefines<XDataType, WDataType, YDataType, ComputeDataType>(useTf32);

        auto* xPtr = x.memory().deviceData();
        auto* wPtr = w.memory().deviceData();
        auto* yPtr = y.memory().deviceData();

        // Only prePadding is passed to the kernel. Post-padding is implicitly
        // handled by the output tensor dimensions and the kernel's bounds checks.
        if(nDims == 3)
        {
            launchFprop1d(xPtr,
                          wPtr,
                          yPtr,
                          x.dims(),
                          w.dims(),
                          y.dims(),
                          x.strides(),
                          w.strides(),
                          y.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 4)
        {
            launchFprop2d(xPtr,
                          wPtr,
                          yPtr,
                          x.dims(),
                          w.dims(),
                          y.dims(),
                          x.strides(),
                          w.strides(),
                          y.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else if(nDims == 5)
        {
            launchFprop3d(xPtr,
                          wPtr,
                          yPtr,
                          x.dims(),
                          w.dims(),
                          y.dims(),
                          x.strides(),
                          w.strides(),
                          y.strides(),
                          convStrides,
                          dilations,
                          prePadding,
                          defines,
                          alpha,
                          beta);
        }
        else
        {
            throw std::invalid_argument("Unsupported number of dimensions: "
                                        + std::to_string(nDims));
        }

        y.memory().markDeviceModified();
    }

private:
    // --- Validation ---

    template <typename T1, typename T2, typename T3>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<T1>& x,
                              const hipdnn_data_sdk::utilities::TensorBase<T2>& w,
                              const hipdnn_data_sdk::utilities::TensorBase<T3>& y,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding)
    {
        const auto nDims = x.dims().size();

        // GPU kernels only support 1D (3), 2D (4), and 3D (5) convolutions
        if(nDims != 3 && nDims != 4 && nDims != 5)
        {
            throw std::invalid_argument("Input tensor must have 3 dimensions (1D conv), "
                                        "4 dimensions (2D conv), or 5 dimensions (3D conv)");
        }

        hipdnn_test_sdk::utilities::validateConvolutionParams(
            x, w, y, strides, dilations, prePadding, postPadding);
    }

    // --- 1D kernel launcher ---

    static void launchFprop1d(const void* xPtr,
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
        auto& compiler = hipdnn_gpu_ref::detail::GpuRefKernelCompiler::instance();
        auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef1d");

        auto nGroups = xDims[1] / wDims[1];

        detail::ConvFwdArgs1d args{};
        args.x = xPtr;
        args.w = wPtr;
        args.y = yPtr;
        args.xStr = detail::toStrides3(xTensorStrides);
        args.wStr = detail::toStrides3(wTensorStrides);
        args.yStr = detail::toStrides3(yTensorStrides);
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
        detail::launchKernel(kernel.function(), totalElements, &args, sizeof(args));
    }

    // --- 2D kernel launcher ---

    static void launchFprop2d(const void* xPtr,
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
        auto& compiler = hipdnn_gpu_ref::detail::GpuRefKernelCompiler::instance();
        auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef2d");

        auto nGroups = xDims[1] / wDims[1];

        detail::ConvFwdArgs2d args{};
        args.x = xPtr;
        args.w = wPtr;
        args.y = yPtr;
        args.xStr = detail::toStrides4(xTensorStrides);
        args.wStr = detail::toStrides4(wTensorStrides);
        args.yStr = detail::toStrides4(yTensorStrides);
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
        detail::launchKernel(kernel.function(), totalElements, &args, sizeof(args));
    }

    // --- 3D kernel launcher ---

    static void launchFprop3d(const void* xPtr,
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
        auto& compiler = hipdnn_gpu_ref::detail::GpuRefKernelCompiler::instance();
        auto& kernel = compiler.getOrCompile("GpuRefConvFwd.cpp", defines, "convFwdRef3d");

        auto nGroups = xDims[1] / wDims[1];

        detail::ConvFwdArgs3d args{};
        args.x = xPtr;
        args.w = wPtr;
        args.y = yPtr;
        args.xStr = detail::toStrides5(xTensorStrides);
        args.wStr = detail::toStrides5(wTensorStrides);
        args.yStr = detail::toStrides5(yTensorStrides);
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
        detail::launchKernel(kernel.function(), totalElements, &args, sizeof(args));
    }
};

} // namespace hipdnn_gpu_ref
