// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefHelpers.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>
#include <hipdnn_data_sdk/utilities/MigratableMemory.hpp>

#include <cstring>

namespace hipdnn_gpu_ref
{
namespace
{
#include <GpuRefBlockScaleArgs.h> // NOLINT(misc-include-cleaner)
}

std::optional<std::string>
    GpuFpReferenceBlockScale::checkQuantize(const std::vector<int64_t>& xDims,
                                            const std::vector<int64_t>& yDims,
                                            const std::vector<int64_t>& scaleDims,
                                            int32_t blockSize,
                                            std::optional<int64_t> axis)
{
    if(xDims.empty() || xDims.size() > MAX_TENSOR_RANK || xDims != yDims
       || scaleDims.size() != xDims.size())
    {
        return "BlockScaleQuantize requires matching non-empty IO dimensions of rank at most 5.";
    }
    if(blockSize <= 0)
    {
        return "BlockScaleQuantize requires a positive block size.";
    }
    const auto targetAxis = axis.value_or(static_cast<int64_t>(xDims.size()) - 1);
    if(targetAxis < 0 || targetAxis >= static_cast<int64_t>(xDims.size()))
    {
        return "BlockScaleQuantize axis is out of bounds.";
    }
    auto expectedScaleDims = xDims;
    expectedScaleDims[static_cast<size_t>(targetAxis)]
        = (xDims[static_cast<size_t>(targetAxis)] + blockSize - 1) / blockSize;
    if(scaleDims != expectedScaleDims)
    {
        return "BlockScaleQuantize scale dimensions do not match the input.";
    }
    return std::nullopt;
}

void GpuFpReferenceBlockScale::validateQuantize(const std::vector<int64_t>& xDims,
                                                const std::vector<int64_t>& yDims,
                                                const std::vector<int64_t>& scaleDims,
                                                int32_t blockSize,
                                                std::optional<int64_t> axis)
{
    if(const auto error = checkQuantize(xDims, yDims, scaleDims, blockSize, axis))
    {
        throw std::invalid_argument(*error);
    }
}

std::optional<std::string>
    GpuFpReferenceBlockScale::checkDequantize(const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& yDims,
                                              const std::vector<int64_t>& scaleDims,
                                              const std::vector<int32_t>& blockSize)
{
    if(xDims.empty() || xDims.size() > MAX_TENSOR_RANK || xDims != yDims
       || scaleDims.size() != xDims.size() || blockSize.size() > xDims.size())
    {
        return "BlockScaleDequantize requires matching non-empty IO dimensions of rank at most 5.";
    }
    std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
    const auto offset = xDims.size() - blockSize.size();
    for(size_t i = 0; i < blockSize.size(); ++i)
    {
        if(blockSize[i] <= 0)
        {
            return "BlockScaleDequantize requires positive block sizes.";
        }
        effectiveBlockSize[offset + i] = blockSize[i];
    }
    for(size_t dim = 0; dim < xDims.size(); ++dim)
    {
        if(scaleDims[dim] != (xDims[dim] + effectiveBlockSize[dim] - 1) / effectiveBlockSize[dim])
        {
            return "BlockScaleDequantize scale dimensions do not match the input.";
        }
    }
    return std::nullopt;
}

std::vector<int64_t>
    GpuFpReferenceBlockScale::validateDequantize(const std::vector<int64_t>& xDims,
                                                 const std::vector<int64_t>& yDims,
                                                 const std::vector<int64_t>& scaleDims,
                                                 const std::vector<int32_t>& blockSize)
{
    if(const auto error = checkDequantize(xDims, yDims, scaleDims, blockSize))
    {
        throw std::invalid_argument(*error);
    }
    std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
    const auto offset = xDims.size() - blockSize.size();
    for(size_t i = 0; i < blockSize.size(); ++i)
    {
        effectiveBlockSize[offset + i] = blockSize[i];
    }
    return effectiveBlockSize;
}

void GpuFpReferenceBlockScale::validateStrides(const std::vector<int64_t>& dims,
                                               const std::vector<int64_t>& strides,
                                               const char* tensorName)
{
    if(strides.size() != dims.size())
    {
        throw std::invalid_argument(std::string("BlockScale ") + tensorName
                                    + " strides must match its rank.");
    }
}

void GpuFpReferenceBlockScale::launchQuantize(const void* x,
                                              const std::vector<int64_t>& xDims,
                                              const std::vector<int64_t>& xStrides,
                                              void* y,
                                              const std::vector<int64_t>& yStrides,
                                              void* scale,
                                              const std::vector<int64_t>& scaleDims,
                                              const std::vector<int64_t>& scaleStrides,
                                              int32_t blockSize,
                                              int64_t axis,
                                              double outputLowest,
                                              double outputMax,
                                              const std::vector<std::string>& defines)
{
    const auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefBlockScaleQuantize.cpp", defines, "BlockScaleQuantizeRef");
    BlockScaleQuantizeArgs args{};
    args.x = x;
    args.y = y;
    args.scale = scale;
    args.scaleElements = detail::elementCount(scaleDims);
    args.nDim = static_cast<int>(xDims.size());
    args.axis = static_cast<int>(axis);
    args.blockSize = blockSize;
    args.outputLowest = outputLowest;
    args.outputMax = outputMax;
    hipdnn_data_sdk::utilities::MigratableMemory<int> invalidInput(1);
    invalidInput.hostData()[0] = 0;
    args.invalidInput = static_cast<int*>(invalidInput.deviceData());
    std::memcpy(args.xDims, xDims.data(), xDims.size() * sizeof(int64_t));
    std::memcpy(args.xStrides, xStrides.data(), xStrides.size() * sizeof(int64_t));
    std::memcpy(args.yStrides, yStrides.data(), yStrides.size() * sizeof(int64_t));
    std::memcpy(args.scaleDims, scaleDims.data(), scaleDims.size() * sizeof(int64_t));
    std::memcpy(args.scaleStrides, scaleStrides.data(), scaleStrides.size() * sizeof(int64_t));
    detail::launchKernelForElements(kernel.function(), args.scaleElements, &args, sizeof(args));
    invalidInput.markDeviceModified();
    if(invalidInput.hostData()[0] != 0)
    {
        throw std::runtime_error("BlockScaleQuantize: input tensor contains a NaN or Inf value.");
    }
}

void GpuFpReferenceBlockScale::launchDequantize(const void* x,
                                                const std::vector<int64_t>& xDims,
                                                const std::vector<int64_t>& xStrides,
                                                const void* scale,
                                                const std::vector<int64_t>& scaleStrides,
                                                void* y,
                                                const std::vector<int64_t>& yStrides,
                                                const std::vector<int64_t>& effectiveBlockSize,
                                                bool isNegativeScale,
                                                const std::vector<std::string>& defines)
{
    const auto& kernel = detail::GpuRefKernelCompiler::instance().getOrCompile(
        "GpuRefBlockScaleDequantize.cpp", defines, "BlockScaleDequantizeRef");
    BlockScaleDequantizeArgs args{};
    args.x = x;
    args.scale = scale;
    args.y = y;
    args.elements = detail::elementCount(xDims);
    args.nDim = static_cast<int>(xDims.size());
    args.isNegativeScale = isNegativeScale ? 1 : 0;
    std::memcpy(args.xDims, xDims.data(), xDims.size() * sizeof(int64_t));
    std::memcpy(args.xStrides, xStrides.data(), xStrides.size() * sizeof(int64_t));
    std::memcpy(args.yStrides, yStrides.data(), yStrides.size() * sizeof(int64_t));
    std::memcpy(args.scaleStrides, scaleStrides.data(), scaleStrides.size() * sizeof(int64_t));
    std::memcpy(args.effectiveBlockSize,
                effectiveBlockSize.data(),
                effectiveBlockSize.size() * sizeof(int64_t));
    detail::launchKernelForElements(kernel.function(), args.elements, &args, sizeof(args));
}

} // namespace hipdnn_gpu_ref
