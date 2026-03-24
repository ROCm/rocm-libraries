// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_gpu_ref/detail/GpuRefKernelCompiler.hpp>

#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_gpu_ref
{

namespace detail
{

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

inline void throwOnHipError(hipError_t err, const char* msg)
{
    if(err != hipSuccess)
    {
        throw std::runtime_error(std::string(msg) + ": " + hipGetErrorString(err));
    }
}

} // namespace detail

class GpuFpReferenceConvolution
{
public:
    // Overload for uniform padding
    template <class XType, class WType, class YType, class ComputeType = float>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YType>& y,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding)
    {
        fprop<XType, WType, YType, ComputeType>(x, w, y, strides, dilations, padding, padding);
    }

    template <class XType, class WType, class YType, class ComputeType = float>
    static void fprop(hipdnn_data_sdk::utilities::TensorBase<XType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<WType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YType>& y,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
    {
        validateInput(x, w, y, strides, dilations, prePadding, postPadding);

        const auto& xDims = x.dims();
        const auto& wDims = w.dims();
        const auto& yDims = y.dims();

        auto nBatch = xDims[0];
        auto nInputChannels = xDims[1];
        auto hIn = xDims[2];
        auto wIn = xDims[3];
        auto nOutputChannels = wDims[0];
        auto kH = wDims[2];
        auto kW = wDims[3];
        auto hOut = yDims[2];
        auto wOut = yDims[3];

        auto channelsPerGroup = wDims[1];
        auto nGroups = nInputChannels / channelsPerGroup;

        // Get or compile the kernel for this data type
        const std::string typeName = detail::HipRtcTypeName<XType>::VALUE;
        auto& compiler = hipdnn_gpu_ref::detail::GpuRefKernelCompiler::instance();
        auto& kernel = compiler.getOrCompile(typeName, "convFwdRef");

        // Ensure data is on device (non-const: deviceData() may trigger host->device copy)
        auto* xPtr = x.memory().deviceData();
        auto* wPtr = w.memory().deviceData();
        auto* yPtr = y.memory().deviceData();

        // Calculate grid dimensions
        auto totalElements = nBatch * nOutputChannels * hOut * wOut;
        const int64_t blockSize = 256;
        auto gridSize = (totalElements + blockSize - 1) / blockSize;

        // Build kernel arguments — must match the convFwdRef signature exactly
        // NOLINTBEGIN(misc-non-private-member-variables-in-classes)
        struct KernelArgs
        {
            const void* xData;
            const void* wData;
            void* yData;
            long long nBatch;
            long long nInputChannels;
            long long hIn;
            long long wIn;
            long long nOutputChannels;
            long long hOut;
            long long wOut;
            long long kH;
            long long kW;
            long long strideH;
            long long strideW;
            long long dilationH;
            long long dilationW;
            long long padH;
            long long padW;
            long long nGroups;
        };
        // NOLINTEND(misc-non-private-member-variables-in-classes)

        KernelArgs args;
        args.xData = xPtr;
        args.wData = wPtr;
        args.yData = yPtr;
        args.nBatch = static_cast<long long>(nBatch);
        args.nInputChannels = static_cast<long long>(nInputChannels);
        args.hIn = static_cast<long long>(hIn);
        args.wIn = static_cast<long long>(wIn);
        args.nOutputChannels = static_cast<long long>(nOutputChannels);
        args.hOut = static_cast<long long>(hOut);
        args.wOut = static_cast<long long>(wOut);
        args.kH = static_cast<long long>(kH);
        args.kW = static_cast<long long>(kW);
        args.strideH = static_cast<long long>(strides[0]);
        args.strideW = static_cast<long long>(strides[1]);
        args.dilationH = static_cast<long long>(dilations[0]);
        args.dilationW = static_cast<long long>(dilations[1]);
        args.padH = static_cast<long long>(prePadding[0]);
        args.padW = static_cast<long long>(prePadding[1]);
        args.nGroups = static_cast<long long>(nGroups);

        size_t argsSize = sizeof(args);
        // NOLINTNEXTLINE(modernize-avoid-c-arrays)
        void* config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                          &args,
                          HIP_LAUNCH_PARAM_BUFFER_SIZE,
                          &argsSize,
                          HIP_LAUNCH_PARAM_END};

        detail::throwOnHipError(
            hipModuleLaunchKernel(kernel.function(),
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
            "hipModuleLaunchKernel failed for convFwdRef");

        detail::throwOnHipError(hipDeviceSynchronize(), "hipDeviceSynchronize failed");

        y.memory().markDeviceModified();
    }

private:
    template <typename T1, typename T2, typename T3>
    static void validateInput(const hipdnn_data_sdk::utilities::TensorBase<T1>& x,
                              const hipdnn_data_sdk::utilities::TensorBase<T2>& w,
                              const hipdnn_data_sdk::utilities::TensorBase<T3>& y,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding)
    {
        if(x.dims().size() != 4)
        {
            throw std::invalid_argument("Input tensor must have exactly 4 dimensions (NCHW)");
        }

        if(w.dims().size() != 4)
        {
            throw std::invalid_argument("Weight tensor must have exactly 4 dimensions");
        }

        if(y.dims().size() != 4)
        {
            throw std::invalid_argument("Output tensor must have exactly 4 dimensions (NCHW)");
        }

        if(strides.size() != 2)
        {
            throw std::invalid_argument("Strides must have exactly 2 elements for 2D convolution");
        }

        if(dilations.size() != 2)
        {
            throw std::invalid_argument(
                "Dilations must have exactly 2 elements for 2D convolution");
        }

        if(prePadding.size() != 2)
        {
            throw std::invalid_argument(
                "PrePadding must have exactly 2 elements for 2D convolution");
        }

        if(postPadding.size() != 2)
        {
            throw std::invalid_argument(
                "PostPadding must have exactly 2 elements for 2D convolution");
        }

        const auto& xDims = x.dims();
        const auto& wDims = w.dims();
        const auto& yDims = y.dims();

        for(int i = 0; i < 2; ++i)
        {
            auto idx = static_cast<size_t>(i);

            if(strides[idx] <= 0)
            {
                throw std::invalid_argument("Stride values must be positive");
            }

            if(dilations[idx] <= 0)
            {
                throw std::invalid_argument("Dilation values must be positive");
            }

            if(prePadding[idx] < 0)
            {
                throw std::invalid_argument("PrePadding values must be non-negative");
            }

            if(postPadding[idx] < 0)
            {
                throw std::invalid_argument("PostPadding values must be non-negative");
            }

            const int64_t xDim = xDims[idx + 2];
            const int64_t kernelDim = wDims[idx + 2];
            const int64_t yDim = yDims[idx + 2];

            const int64_t kernelSize = (dilations[idx] * (kernelDim - 1)) + 1;
            const int64_t expectedOutputDim
                = ((xDim + prePadding[idx] + postPadding[idx] - kernelSize) / strides[idx]) + 1;

            if(expectedOutputDim != yDim)
            {
                throw std::invalid_argument(
                    "Output dimension " + std::to_string(yDim) + " at spatial dimension "
                    + std::to_string(i) + " does not match expected dimension "
                    + std::to_string(expectedOutputDim));
            }
        }
    }
};

} // namespace hipdnn_gpu_ref
