// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/ConvolutionValidation.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceConvolution
{
public:
    // Check if this CPU implementation supports the given node configuration
    static bool isApplicable(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        bool validNode = (node.attributes_type() == NodeAttributes::ConvolutionFwdAttributes
                          || node.attributes_type() == NodeAttributes::ConvolutionBwdAttributes);

        if(node.attributes_type() == NodeAttributes::ConvolutionBwdAttributes)
        {
            auto convAttr = node.attributes_as_ConvolutionBwdAttributes();
            validNode &= convAttr->conv_mode() == ConvMode::CROSS_CORRELATION;
        }

        if(node.attributes_type() == NodeAttributes::ConvolutionFwdAttributes)
        {
            auto convAttr = node.attributes_as_ConvolutionFwdAttributes();
            validNode &= convAttr->conv_mode() == ConvMode::CROSS_CORRELATION;
        }

        return validNode;
    }

    // Overload for uniform padding
    template <class XDataType, class WDataType, class YDataType, class ComputeDataType = float>
    static void fprop(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      const hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding)
    {
        fprop(x, w, y, strides, dilations, padding, padding);
    }

    template <class XDataType, class WDataType, class YDataType, class ComputeDataType = float>
    static void fprop(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      const hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      hipdnn_data_sdk::utilities::TensorBase<YDataType>& y,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
    {
        validateInput(x, w, y, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NC[spatial...] format for x/y, [G*K][C][spatial...] for w
        const auto& xDims = x.dims();
        const auto& wDims = w.dims();
        const auto& yDims = y.dims();

        const int64_t nBatch = xDims[0];
        const int64_t nInputChannels = xDims[1];
        const int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(xDims.size()) - 2;
        std::vector<int64_t> xSpatialDims(xDims.begin() + 2, xDims.end());
        std::vector<int64_t> kernelSpatialDims(wDims.begin() + 2, wDims.end());
        std::vector<int64_t> ySpatialDims(yDims.begin() + 2, yDims.end());

        // Calculate groups from x/w channel relationship
        const int64_t nGroups = nInputChannels / channelsPerGroup;
        const int64_t yChannelsPerGroup = totalOutputChannels / nGroups;

        // Raw pointers and strides are hoisted once. The inner loops never call
        // getHostValue/setHostValue: each of those costs a virtual memory() call plus a
        // stride reduction over a freshly allocated index vector.
        const XDataType* xBase = x.memory().hostData();
        const WDataType* wBase = w.memory().hostData();
        YDataType* yBase = y.memory().hostData();

        const auto& xStrides = x.strides();
        const auto& wStrides = w.strides();
        const auto& yStrides = y.strides();

        // This lambda computes a single element of the y tensor
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            const int64_t gIdx = indices[0]; // group index
            const int64_t nIdx = indices[1]; // batch index
            const int64_t kIdx = indices[2]; // y channel within group

            // Add 3 because [gIdx, nIdx, kIdx] are the first 3 elements
            const int64_t* ySpatialIndices = indices.data() + 3;

            // Which kernel taps hit the logical x tensor depends only on the y spatial
            // position, so resolve the whole window once instead of per channel.
            thread_local hipdnn_test_sdk::detail::ConvolutionWindow window;
            window.build(static_cast<size_t>(nSpatialDims),
                         kernelSpatialDims.data(),
                         wStrides.data() + 2,
                         xStrides.data() + 2,
                         [&](size_t dim, int64_t kernelIndex) {
                             const int64_t xIndex = (ySpatialIndices[dim] * strides[dim])
                                                    + (kernelIndex * dilations[dim])
                                                    - prePadding[dim];

                             // In either case, this position does not exist in the logical x tensor.
                             // 1.  (y_idx * stride) + (kernel_idx * dilation) - prePadding < 0
                             //  => (y_idx * stride) + (kernel_idx * dilation) < prePadding
                             // 2.  (y_idx * stride) + (kernel_idx * dilation) - prePadding >= x_dim
                             //  => (y_idx * stride) + (kernel_idx * dilation) >= x_dim + prePadding
                             // It is implicit in Case 2 that the position could be in the postPadding region.
                             return (xIndex < 0 || xIndex >= xSpatialDims[dim]) ? -1 : xIndex;
                         });

            // Weight dims: [yChannels, xChannels/groupCount, ...]
            // Thus, we index via flattened y channel index and group-offset x channel index (c).
            const int64_t yChannel = (gIdx * yChannelsPerGroup) + kIdx;
            const WDataType* wFilter = wBase + (yChannel * wStrides[0]);

            // Input dims: [n, xChannel, ...], indexed via the global x channel index.
            const XDataType* xBatch
                = xBase + (nIdx * xStrides[0]) + (gIdx * channelsPerGroup * xStrides[1]);

            auto accumulator = static_cast<ComputeDataType>(0.0f);

            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                const XDataType* xChannel = xBatch + (c * xStrides[1]);
                const WDataType* wChannel = wFilter + (c * wStrides[1]);

                for(const auto& tap : window.taps())
                {
                    accumulator = accumulator
                                  + (static_cast<ComputeDataType>(xChannel[tap.sourceOffset])
                                     * static_cast<ComputeDataType>(wChannel[tap.windowOffset]));
                }
            }

            int64_t yOffset = (nIdx * yStrides[0]) + (yChannel * yStrides[1]);
            for(int64_t dim = 0; dim < nSpatialDims; ++dim)
            {
                const auto dimIdx = static_cast<size_t>(dim);
                yOffset += ySpatialIndices[dimIdx] * yStrides[dimIdx + 2];
            }

            yBase[yOffset] = static_cast<YDataType>(accumulator);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, yChannelsPerGroup};
        parallelDims.insert(parallelDims.end(), ySpatialDims.begin(), ySpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        y.memory().markHostModified();
    }

    // Overload for uniform padding
    template <class DxDataType, class WDataType, class DyDataType, class ComputeDataType = float>
    static void dgrad(hipdnn_data_sdk::utilities::TensorBase<DxDataType>& gradX,
                      const hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding)
    {
        dgrad(gradX, w, gradY, strides, dilations, padding, padding);
    }

    template <class DxDataType, class WDataType, class DyDataType, class ComputeDataType = float>
    static void dgrad(hipdnn_data_sdk::utilities::TensorBase<DxDataType>& gradX,
                      const hipdnn_data_sdk::utilities::TensorBase<WDataType>& w,
                      const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
    {
        validateInput(gradX, w, gradY, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NC[spatial...] format for x/y, [G*K][C][spatial...] for w
        const auto& xDims = gradX.dims();
        const auto& wDims = w.dims();
        const auto& yDims = gradY.dims();

        const int64_t nBatch = xDims[0];
        const int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(xDims.size()) - 2;
        std::vector<int64_t> xSpatialDims(xDims.begin() + 2, xDims.end());
        std::vector<int64_t> kernelSpatialDims(wDims.begin() + 2, wDims.end());
        std::vector<int64_t> ySpatialDims(yDims.begin() + 2, yDims.end());

        // Calculate groups from x/w channel relationship
        const int64_t nInputChannels = xDims[1];
        const int64_t nGroups = nInputChannels / channelsPerGroup; // G
        const int64_t yChannelsPerGroup = totalOutputChannels / nGroups; // K

        // Raw pointers and strides are hoisted once; see fprop.
        DxDataType* gradXBase = gradX.memory().hostData();
        const WDataType* wBase = w.memory().hostData();
        const DyDataType* gradYBase = gradY.memory().hostData();

        const auto& xStrides = gradX.strides();
        const auto& wStrides = w.strides();
        const auto& yStrides = gradY.strides();

        // This lambda computes a single element of the x gradient tensor (dx)
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            const int64_t gIdx = indices[0]; // group index
            const int64_t nIdx = indices[1]; // batch index
            const int64_t cIdx = indices[2]; // channel index within group

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            const int64_t* xSpatialIndices = indices.data() + 3;

            // Which kernel taps have a contributing y gradient depends only on the x
            // spatial position, so resolve the whole window once instead of per y channel.
            thread_local hipdnn_test_sdk::detail::ConvolutionWindow window;
            window.build(static_cast<size_t>(nSpatialDims),
                         kernelSpatialDims.data(),
                         wStrides.data() + 2,
                         yStrides.data() + 2,
                         [&](size_t dim, int64_t kernelIndex) -> int64_t {
                             const int64_t tmp = xSpatialIndices[dim] + prePadding[dim]
                                                 - (kernelIndex * dilations[dim]);

                             // Check if the current x position could have contributed to an y element. If the
                             // remainder is non-zero, this combination is not aligned with the stride, so it's not a valid
                             // mapping from the forward pass.
                             if(tmp % strides[dim] != 0)
                             {
                                 return -1;
                             }

                             // Check if position does not exist in the logical y tensor.
                             // 1.  (x_idx + prePadding - (kernel_idx * dilation)) / stride < 0
                             //  => numerator < 0 => sampling from a location before the y tensor
                             // 2.  (x_idx + prePadding - (kernel_idx * dilation)) / stride >= y_dim
                             //  => x_idx + prePadding >= (y_dim * stride) + (kernel_idx * dilation) => beyond the y tensor
                             const int64_t yIndex = tmp / strides[dim];
                             return (yIndex < 0 || yIndex >= ySpatialDims[dim]) ? -1 : yIndex;
                         });

            const int64_t gradYBatch
                = (nIdx * yStrides[0]) + (gIdx * yChannelsPerGroup * yStrides[1]);
            const int64_t wFilter = (gIdx * yChannelsPerGroup * wStrides[0]) + (cIdx * wStrides[1]);

            auto vAcc = static_cast<ComputeDataType>(0.0f);

            for(const auto& tap : window.taps())
            {
                // Iterate over each y channel in the group, as they all contribute to the x gradient
                for(int64_t k = 0; k < yChannelsPerGroup; ++k)
                {
                    const DyDataType vOut
                        = gradYBase[gradYBatch + (k * yStrides[1]) + tap.sourceOffset];
                    const WDataType vWei = wBase[wFilter + (k * wStrides[0]) + tap.windowOffset];

                    vAcc = vAcc
                           + (static_cast<ComputeDataType>(vOut)
                              * static_cast<ComputeDataType>(vWei));
                }
            }

            const int64_t xChannelIdx = (gIdx * channelsPerGroup) + cIdx;
            int64_t gradXOffset = (nIdx * xStrides[0]) + (xChannelIdx * xStrides[1]);
            for(int64_t dim = 0; dim < nSpatialDims; ++dim)
            {
                const auto dimIdx = static_cast<size_t>(dim);
                gradXOffset += xSpatialIndices[dimIdx] * xStrides[dimIdx + 2];
            }

            gradXBase[gradXOffset] = static_cast<DxDataType>(vAcc);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), xSpatialDims.begin(), xSpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        gradX.memory().markHostModified();
    }

    template <class XDataType, class DwDataType, class DyDataType, class ComputeDataType = float>
    static void wgrad(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<DwDataType>& gradW,
                      const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& padding)
    {
        wgrad(x, gradW, gradY, strides, dilations, padding, padding);
    }

    template <class XDataType, class DwDataType, class DyDataType, class ComputeDataType = float>
    static void wgrad(const hipdnn_data_sdk::utilities::TensorBase<XDataType>& x,
                      hipdnn_data_sdk::utilities::TensorBase<DwDataType>& gradW,
                      const hipdnn_data_sdk::utilities::TensorBase<DyDataType>& gradY,
                      const std::vector<int64_t>& strides,
                      const std::vector<int64_t>& dilations,
                      const std::vector<int64_t>& prePadding,
                      const std::vector<int64_t>& postPadding)
    {
        validateInput(x, gradW, gradY, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NCHW format for x/y, [G*K][C][Y][X] for w (4D flattened)
        const auto& xDims = x.dims();
        const auto& wDims = gradW.dims();
        const auto& yDims = gradY.dims();

        int64_t nBatch = yDims[0];

        int64_t nSpatialDims = static_cast<int64_t>(xDims.size()) - 2;
        std::vector<int64_t> xSpatialDims(xDims.begin() + 2, xDims.end());
        std::vector<int64_t> kernelSpatialDims(wDims.begin() + 2, wDims.end());
        std::vector<int64_t> ySpatialDims(yDims.begin() + 2, yDims.end());

        const int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        // Calculate groups from x/w channel relationship
        const int64_t nInputChannels = xDims[1];
        const int64_t nGroups = nInputChannels / channelsPerGroup; // G
        const int64_t yChannelsPerGroup = totalOutputChannels / nGroups; // K

        // Raw pointers and strides are hoisted once; see fprop.
        const XDataType* xBase = x.memory().hostData();
        DwDataType* gradWBase = gradW.memory().hostData();
        const DyDataType* gradYBase = gradY.memory().hostData();

        const auto& xStrides = x.strides();
        const auto& wStrides = gradW.strides();
        const auto& yStrides = gradY.strides();

        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            const int64_t gIdx = indices[0];
            const int64_t kIdx = indices[1];
            const int64_t cIdx = indices[2];

            // Add 3 because [gIdx, kIdx, cIdx] are the first 3 elements
            const int64_t* kernelSpatialIndices = indices.data() + 3;

            // Which y gradient positions sample a real x element depends only on the
            // kernel spatial position, so resolve the whole window once instead of per batch.
            thread_local hipdnn_test_sdk::detail::ConvolutionWindow window;
            window.build(static_cast<size_t>(nSpatialDims),
                         ySpatialDims.data(),
                         yStrides.data() + 2,
                         xStrides.data() + 2,
                         [&](size_t dim, int64_t yIndex) {
                             const int64_t xIndex = (yIndex * strides[dim])
                                                    + (kernelSpatialIndices[dim] * dilations[dim])
                                                    - prePadding[dim];

                             return (xIndex < 0 || xIndex >= xSpatialDims[dim]) ? -1 : xIndex;
                         });

            const int64_t yChannelIdx = (gIdx * yChannelsPerGroup) + kIdx;
            const int64_t xChannelIdx = (gIdx * channelsPerGroup) + cIdx;
            const DyDataType* gradYChannel = gradYBase + (yChannelIdx * yStrides[1]);
            const XDataType* xChannel = xBase + (xChannelIdx * xStrides[1]);

            auto vAcc = static_cast<ComputeDataType>(0.0f);

            for(const auto& tap : window.taps())
            {
                for(int64_t n = 0; n < nBatch; ++n)
                {
                    const DyDataType vOut = gradYChannel[(n * yStrides[0]) + tap.windowOffset];
                    const XDataType vIn = xChannel[(n * xStrides[0]) + tap.sourceOffset];

                    vAcc = vAcc
                           + (static_cast<ComputeDataType>(vOut)
                              * static_cast<ComputeDataType>(vIn));
                }
            }

            int64_t gradWOffset = (yChannelIdx * wStrides[0]) + (cIdx * wStrides[1]);
            for(int64_t dim = 0; dim < nSpatialDims; ++dim)
            {
                const auto dimIdx = static_cast<size_t>(dim);
                gradWOffset += kernelSpatialIndices[dimIdx] * wStrides[dimIdx + 2];
            }

            gradWBase[gradWOffset] = static_cast<DwDataType>(vAcc);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, yChannelsPerGroup, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), kernelSpatialDims.begin(), kernelSpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::detail::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        gradW.memory().markHostModified();
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
        if(x.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Input tensor must have at least 3 dimensions (N, C, spatial...)");
        }

        // The kernels address memory through hoisted base pointers and strides, which is
        // the dense layout only; a ragged tensor rebases every batch at its own offset.
        static constexpr auto PREFIX = "CpuFpReferenceConvolution: ";
        hipdnn_test_sdk::detail::validateNoRaggedTensor(x, PREFIX, "x");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(w, PREFIX, "w");
        hipdnn_test_sdk::detail::validateNoRaggedTensor(y, PREFIX, "y");

        hipdnn_test_sdk::utilities::validateConvolutionParams(
            x, w, y, strides, dilations, prePadding, postPadding);
    }
};

} // namespace hipdnn_test_sdk::utilities
