// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipdnn-gpu-ref/GpuFpReferenceBlockScale.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefKernelCompiler.hpp>
#include <hipdnn-gpu-ref/detail/GpuRefLaunch.hpp>

#include <cstring>

namespace hipdnn_gpu_ref
{
namespace
{
#include <GpuRefBlockScaleArgs.h> // NOLINT(misc-include-cleaner)

int64_t elementCount(const std::vector<int64_t>& dims)
{
    return std::accumulate(dims.begin(), dims.end(), int64_t{1}, std::multiplies<>());
}
}

void GpuFpReferenceBlockScale::validateQuantize(const std::vector<int64_t>& xDims,
                                                const std::vector<int64_t>& yDims,
                                                const std::vector<int64_t>& scaleDims,
                                                int32_t blockSize,
                                                std::optional<int64_t> axis)
{
    if(xDims.empty() || xDims.size() > 5 || xDims != yDims || scaleDims.size() != xDims.size())
    {
        throw std::invalid_argument(
            "BlockScaleQuantize requires matching non-empty IO dimensions of rank at most 5.");
    }
    if(blockSize <= 0)
    {
        throw std::invalid_argument("BlockScaleQuantize requires a positive block size.");
    }
    const auto targetAxis = axis.value_or(static_cast<int64_t>(xDims.size()) - 1);
    if(targetAxis < 0 || targetAxis >= static_cast<int64_t>(xDims.size()))
    {
        throw std::out_of_range("BlockScaleQuantize axis is out of bounds.");
    }
    auto expectedScaleDims = xDims;
    expectedScaleDims[static_cast<size_t>(targetAxis)]
        = (xDims[static_cast<size_t>(targetAxis)] + blockSize - 1) / blockSize;
    if(scaleDims != expectedScaleDims)
    {
        throw std::invalid_argument("BlockScaleQuantize scale dimensions do not match the input.");
    }
}

std::vector<int64_t>
    GpuFpReferenceBlockScale::validateDequantize(const std::vector<int64_t>& xDims,
                                                 const std::vector<int64_t>& yDims,
                                                 const std::vector<int64_t>& scaleDims,
                                                 const std::vector<int32_t>& blockSize)
{
    if(xDims.empty() || xDims.size() > 5 || xDims != yDims || scaleDims.size() != xDims.size()
       || blockSize.size() > xDims.size())
    {
        throw std::invalid_argument(
            "BlockScaleDequantize requires matching non-empty IO dimensions of rank at most 5.");
    }
    std::vector<int64_t> effectiveBlockSize(xDims.size(), 1);
    const auto offset = xDims.size() - blockSize.size();
    for(size_t i = 0; i < blockSize.size(); ++i)
    {
        if(blockSize[i] <= 0)
        {
            throw std::invalid_argument("BlockScaleDequantize requires positive block sizes.");
        }
        effectiveBlockSize[offset + i] = blockSize[i];
    }
    for(size_t dim = 0; dim < xDims.size(); ++dim)
    {
        if(scaleDims[dim] != (xDims[dim] + effectiveBlockSize[dim] - 1) / effectiveBlockSize[dim])
        {
            throw std::invalid_argument(
                "BlockScaleDequantize scale dimensions do not match the input.");
        }
    }
    return effectiveBlockSize;
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
    args.scaleElements = elementCount(scaleDims);
    args.nDim = static_cast<int>(xDims.size());
    args.axis = static_cast<int>(axis);
    args.blockSize = blockSize;
    args.outputLowest = outputLowest;
    args.outputMax = outputMax;
    std::memcpy(args.xDims, xDims.data(), xDims.size() * sizeof(int64_t));
    std::memcpy(args.xStrides, xStrides.data(), xStrides.size() * sizeof(int64_t));
    std::memcpy(args.yStrides, yStrides.data(), yStrides.size() * sizeof(int64_t));
    std::memcpy(args.scaleDims, scaleDims.data(), scaleDims.size() * sizeof(int64_t));
    std::memcpy(args.scaleStrides, scaleStrides.data(), scaleStrides.size() * sizeof(int64_t));
    detail::launchKernel1d(kernel.function(),
                           (args.scaleElements + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           BLOCK_SIZE,
                           &args,
                           sizeof(args));
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
    args.elements = elementCount(xDims);
    args.nDim = static_cast<int>(xDims.size());
    args.isNegativeScale = isNegativeScale ? 1 : 0;
    std::memcpy(args.xDims, xDims.data(), xDims.size() * sizeof(int64_t));
    std::memcpy(args.xStrides, xStrides.data(), xStrides.size() * sizeof(int64_t));
    std::memcpy(args.yStrides, yStrides.data(), yStrides.size() * sizeof(int64_t));
    std::memcpy(args.scaleStrides, scaleStrides.data(), scaleStrides.size() * sizeof(int64_t));
    std::memcpy(args.effectiveBlockSize,
                effectiveBlockSize.data(),
                effectiveBlockSize.size() * sizeof(int64_t));
    detail::launchKernel1d(kernel.function(),
                           (args.elements + BLOCK_SIZE - 1) / BLOCK_SIZE,
                           BLOCK_SIZE,
                           &args,
                           sizeof(args));
}

} // namespace hipdnn_gpu_ref
