// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceUtilities.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

class CpuFpReferenceConvolution
{
public:
    // Check if this CPU implementation supports the given node configuration
    static bool isApplicable(const hipdnn_data_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_data_sdk::data_objects;

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

        int64_t nBatch = xDims[0];
        int64_t nInputChannels = xDims[1];
        int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(xDims.size()) - 2;
        std::vector<int64_t> xSpatialDims(xDims.begin() + 2, xDims.end());
        std::vector<int64_t> kernelSpatialDims(wDims.begin() + 2, wDims.end());
        std::vector<int64_t> ySpatialDims(yDims.begin() + 2, yDims.end());

        // Calculate groups from x/w channel relationship
        int64_t nGroups = nInputChannels / channelsPerGroup;
        int64_t yChannelsPerGroup = totalOutputChannels / nGroups;

        // This lambda computes a single element of the y tensor
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto kIdx = indices[2]; // y channel within group

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> ySpatialIndices(indices.begin() + 3, indices.end());

            auto accumulator = static_cast<ComputeDataType>(0.0f);
            int64_t baseInputChannel = gIdx * channelsPerGroup;

            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                int64_t xChannel = baseInputChannel + c;

                // Iterate kernel spatial positions
                hipdnn_data_sdk::utilities::iterateAlongDimensions(
                    kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                        std::vector<int64_t> xSpatialIndices(static_cast<size_t>(nSpatialDims));
                        bool validPosition = true;

                        // For each spatial dimension, calculate the corresponding x index
                        for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                        {
                            auto dimIdx = static_cast<size_t>(dim);
                            xSpatialIndices[dimIdx]
                                = (ySpatialIndices[dimIdx] * strides[dimIdx])
                                  + (kernelSpatialIndices[dimIdx] * dilations[dimIdx])
                                  - prePadding[dimIdx];

                            // In either case, this position does not exist in the logical x tensor.
                            // 1.  (y_idx * stride) + (kernel_idx * dilation) - prePadding < 0
                            //  => (y_idx * stride) + (kernel_idx * dilation) < prePadding
                            // 2.  (y_idx * stride) + (kernel_idx * dilation) - prePadding >= x_dim
                            //  => (y_idx * stride) + (kernel_idx * dilation) >= x_dim + prePadding
                            // It is implicit in Case 2 that the position could be in the postPadding region.
                            if(xSpatialIndices[dimIdx] < 0
                               || xSpatialIndices[dimIdx] >= xSpatialDims[dimIdx])
                            {
                                validPosition = false;
                                break;
                            }
                        }

                        if(validPosition)
                        {
                            // Input dims: [n, xChannel, ...]
                            // Thus, we index via global x channel index.
                            auto xFullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                                nIdx, xChannel, xSpatialIndices);

                            // Weight dims: [yChannels,
                            // xChannels/groupCount, ...] Thus, we index via flattened y channel index and
                            // group-offset x channel index (c).
                            int64_t wIdx = (gIdx * yChannelsPerGroup) + kIdx;
                            auto wFullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                                wIdx, c, kernelSpatialIndices);

                            XDataType xVal = x.getHostValue(xFullIndices);
                            WDataType wVal = w.getHostValue(wFullIndices);

                            accumulator = accumulator
                                          + (static_cast<ComputeDataType>(xVal)
                                             * static_cast<ComputeDataType>(wVal));
                        }
                    });
            }

            int64_t yChannel = (gIdx * yChannelsPerGroup) + kIdx;
            auto yFullIndices
                = hipdnn_data_sdk::utilities::buildTensorIndices(nIdx, yChannel, ySpatialIndices);

            y.setHostValue(static_cast<YDataType>(accumulator), yFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, yChannelsPerGroup};
        parallelDims.insert(parallelDims.end(), ySpatialDims.begin(), ySpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
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

        int64_t nBatch = xDims[0];
        int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(xDims.size()) - 2;
        std::vector<int64_t> xSpatialDims(xDims.begin() + 2, xDims.end());
        std::vector<int64_t> kernelSpatialDims(wDims.begin() + 2, wDims.end());
        std::vector<int64_t> ySpatialDims(yDims.begin() + 2, yDims.end());

        // Calculate groups from x/w channel relationship
        int64_t nInputChannels = xDims[1];
        int64_t nGroups = nInputChannels / channelsPerGroup; // G
        int64_t yChannelsPerGroup = totalOutputChannels / nGroups; // K

        // This lambda computes a single element of the x gradient tensor (dx)
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto cIdx = indices[2]; // channel index within group

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> xSpatialIndices(indices.begin() + 3, indices.end());

            auto vAcc = static_cast<ComputeDataType>(0.0f);

            // Iterate over each spatial position of the kernel for contributing y gradients
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                    std::vector<int64_t> ySpatialIndices(static_cast<size_t>(nSpatialDims));
                    bool validPosition = true;

                    // For each spatial dimension, calculate the corresponding y gradient index
                    for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                    {
                        auto dimIdx = static_cast<size_t>(dim);
                        int64_t tmp = xSpatialIndices[dimIdx] + prePadding[dimIdx]
                                      - (kernelSpatialIndices[dimIdx] * dilations[dimIdx]);

                        // Check if the current x position could have contributed to an y element. If the
                        // remainder is non-zero, this combination is not aligned with the stride, so it's not a valid
                        // mapping from the forward pass.
                        if(tmp % strides[dimIdx] != 0)
                        {
                            validPosition = false;
                            break;
                        }

                        ySpatialIndices[dimIdx] = tmp / strides[dimIdx];

                        // Check if position does not exist in the logical y tensor.
                        // 1.  (x_idx + prePadding - (kernel_idx * dilation)) / stride < 0
                        //  => numerator < 0 => sampling from a location before the y tensor
                        // 2.  (x_idx + prePadding - (kernel_idx * dilation)) / stride >= y_dim
                        //  => x_idx + prePadding >= (y_dim * stride) + (kernel_idx * dilation) => beyond the y tensor
                        if(ySpatialIndices[dimIdx] < 0
                           || ySpatialIndices[dimIdx] >= ySpatialDims[dimIdx])
                        {
                            validPosition = false;
                            break;
                        }
                    }

                    if(validPosition)
                    {
                        // Iterate over each y channel in the group, as they all contribute to the x gradient
                        for(int64_t k = 0; k < yChannelsPerGroup; ++k)
                        {
                            auto yChannelIdx = (gIdx * yChannelsPerGroup) + k;

                            auto gradOutputFullIndices
                                = hipdnn_data_sdk::utilities::buildTensorIndices(
                                    nIdx, yChannelIdx, ySpatialIndices);

                            auto wBatchIdx = yChannelIdx;
                            auto wChannelIdx = cIdx;
                            auto wFullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                                wBatchIdx, wChannelIdx, kernelSpatialIndices);

                            DyDataType vOut = gradY.getHostValue(gradOutputFullIndices);
                            WDataType vWei = w.getHostValue(wFullIndices);

                            vAcc = vAcc
                                   + (static_cast<ComputeDataType>(vOut)
                                      * static_cast<ComputeDataType>(vWei));
                        }
                    }
                });

            int64_t xChannelIdx = (gIdx * channelsPerGroup) + cIdx;
            auto gradInputFullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                nIdx, xChannelIdx, xSpatialIndices);

            gradX.setHostValue(static_cast<DxDataType>(vAcc), gradInputFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), xSpatialDims.begin(), xSpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
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

        int64_t totalOutputChannels = wDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = wDims[1]; // C

        // Calculate groups from x/w channel relationship
        int64_t nInputChannels = xDims[1];
        int64_t nGroups = nInputChannels / channelsPerGroup; // G
        int64_t yChannelsPerGroup = totalOutputChannels / nGroups; // K

        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0];
            auto kIdx = indices[1];
            auto cIdx = indices[2];

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> kernelSpatialIndices(indices.begin() + 3, indices.end());

            auto vAcc = static_cast<ComputeDataType>(0.0f);

            // Iterate over each spatial position of the kernel for contributing y gradients
            hipdnn_data_sdk::utilities::iterateAlongDimensions(
                ySpatialDims, [&](const std::vector<int64_t>& ySpatialIndices) {
                    std::vector<int64_t> xSpatialIndices(static_cast<size_t>(nSpatialDims));

                    bool validPosition = true;

                    // For each spatial dimension, calculate the corresponding y gradient index
                    for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                    {
                        auto dimIdx = static_cast<size_t>(dim);

                        int64_t tmp = (ySpatialIndices[dimIdx] * strides[dimIdx])
                                      + (kernelSpatialIndices[dimIdx] * dilations[dimIdx])
                                      - prePadding[dimIdx];

                        xSpatialIndices[dimIdx] = tmp;

                        if(xSpatialIndices[dimIdx] < 0
                           || xSpatialIndices[dimIdx] >= xSpatialDims[dimIdx])
                        {
                            validPosition = false;
                            break;
                        }
                    }

                    if(validPosition)
                    {
                        for(int64_t n = 0; n < nBatch; ++n)
                        {
                            auto yChannelIdx = (gIdx * yChannelsPerGroup) + kIdx;

                            auto gradOutputFullIndices
                                = hipdnn_data_sdk::utilities::buildTensorIndices(
                                    n, yChannelIdx, ySpatialIndices);

                            auto xChannelIdx = (gIdx * channelsPerGroup) + cIdx;

                            auto xFullIndices = hipdnn_data_sdk::utilities::buildTensorIndices(
                                n, xChannelIdx, xSpatialIndices);

                            DyDataType vOut = gradY.getHostValue(gradOutputFullIndices);

                            XDataType vIn = x.getHostValue(xFullIndices);

                            vAcc = vAcc
                                   + (static_cast<ComputeDataType>(vOut)
                                      * static_cast<ComputeDataType>(vIn));
                        }
                    }
                });

            auto wN = (gIdx * yChannelsPerGroup) + kIdx;
            auto wFullIndices
                = hipdnn_data_sdk::utilities::buildTensorIndices(wN, cIdx, kernelSpatialIndices);

            gradW.setHostValue(static_cast<DwDataType>(vAcc), wFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, yChannelsPerGroup, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), kernelSpatialDims.begin(), kernelSpatialDims.end());

        auto parallelFunc
            = hipdnn_test_sdk::utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
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
        // Input validation
        if(x.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Input tensor must have at least 3 dimensions (N, C, spatial...)");
        }

        if(y.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Output tensor must have at least 3 dimensions (N, C, spatial...)");
        }

        if(w.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Weight tensor must have at least 3 dimensions ([G*K], C, spatial...)");
        }

        // Check that all tensors have same number of dimensions
        if(x.dims().size() != y.dims().size() || x.dims().size() != w.dims().size())
        {
            throw std::invalid_argument(
                "Input, y, and w tensors must have the same number of dimensions");
        }

        int64_t nSpatialDims = static_cast<int64_t>(x.dims().size()) - 2;

        if(strides.size() != static_cast<size_t>(nSpatialDims))
        {
            throw std::invalid_argument("Strides must have exactly " + std::to_string(nSpatialDims)
                                        + " elements for " + std::to_string(nSpatialDims)
                                        + "D spatial convolution");
        }

        if(dilations.size() != static_cast<size_t>(nSpatialDims))
        {
            throw std::invalid_argument("Dilations must have exactly "
                                        + std::to_string(nSpatialDims) + " elements for "
                                        + std::to_string(nSpatialDims) + "D spatial convolution");
        }

        if(prePadding.size() != static_cast<size_t>(nSpatialDims))
        {
            throw std::invalid_argument("PrePadding must have exactly "
                                        + std::to_string(nSpatialDims) + " elements for "
                                        + std::to_string(nSpatialDims) + "D spatial convolution");
        }

        if(postPadding.size() != static_cast<size_t>(nSpatialDims))
        {
            throw std::invalid_argument("PostPadding must have exactly "
                                        + std::to_string(nSpatialDims) + " elements for "
                                        + std::to_string(nSpatialDims) + "D spatial convolution");
        }

        const auto& xDims = x.dims();
        const auto& wDims = w.dims();
        const auto& yDims = y.dims();

        for(int64_t i = 0; i < nSpatialDims; ++i)
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

            // Validate that y dimensions are correct given the padding
            // Some of this validation could probably be consolidated into the sdk and removed from frontend nodes
            int64_t xDim = xDims[idx + 2];
            int64_t kernelDim = wDims[idx + 2];
            int64_t yDim = yDims[idx + 2];

            int64_t kernelSize = (dilations[idx] * (kernelDim - 1)) + 1;
            int64_t expectedOutputDim
                = ((xDim + prePadding[idx] + postPadding[idx] - kernelSize) / strides[idx]) + 1;

            if(expectedOutputDim != yDim)
            {
                throw std::invalid_argument(
                    "Output dimension " + std::to_string(yDim) + " at spatial dimension "
                    + std::to_string(i) + " does not match expected dimension "
                    + std::to_string(expectedOutputDim) + " given the x parameters.");
            }
        }
    }
};

} // namespace hipdnn_test_sdk::utilities
