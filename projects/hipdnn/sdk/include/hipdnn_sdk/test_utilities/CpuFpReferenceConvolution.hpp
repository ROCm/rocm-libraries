// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <thread>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

template <class InputDataType, class AccumulatorType>
class CpuFpReferenceConvolutionImpl
{
public:
    // Check if this CPU implementation supports the given node configuration
    static bool isApplicable(const hipdnn_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_sdk::data_objects;

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
    static void convFwdInference(const TensorBase<InputDataType>& input,
                                 const TensorBase<InputDataType>& weight,
                                 TensorBase<InputDataType>& output,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations,
                                 const std::vector<int64_t>& padding)
    {
        convFwdInference(input, weight, output, strides, dilations, padding, padding);
    }

    static void convFwdInference(const TensorBase<InputDataType>& input,
                                 const TensorBase<InputDataType>& weight,
                                 TensorBase<InputDataType>& output,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations,
                                 const std::vector<int64_t>& prePadding,
                                 const std::vector<int64_t>& postPadding)
    {
        validateInput(input, weight, output, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NC[spatial...] format for input/output, [G*K][C][spatial...] for weight
        const auto& inputDims = input.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = output.dims();

        int64_t nBatch = inputDims[0];
        int64_t nInputChannels = inputDims[1];
        int64_t totalOutputChannels = weightDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = weightDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(inputDims.size()) - 2;
        std::vector<int64_t> inputSpatialDims(inputDims.begin() + 2, inputDims.end());
        std::vector<int64_t> kernelSpatialDims(weightDims.begin() + 2, weightDims.end());
        std::vector<int64_t> outputSpatialDims(outputDims.begin() + 2, outputDims.end());

        // Calculate groups from input/weight channel relationship
        int64_t nGroups = nInputChannels / channelsPerGroup;
        int64_t outputChannelsPerGroup = totalOutputChannels / nGroups;

        // This lambda computes a single element of the output tensor
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto kIdx = indices[2]; // output channel within group

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> outputSpatialIndices(indices.begin() + 3, indices.end());

            auto accumulator = static_cast<AccumulatorType>(0.0f);
            int64_t baseInputChannel = gIdx * channelsPerGroup;

            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                int64_t inputChannel = baseInputChannel + c;

                // Iterate kernel spatial positions
                iterateAlongDimensions(
                    kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                        std::vector<int64_t> inputSpatialIndices(static_cast<size_t>(nSpatialDims));
                        bool validPosition = true;

                        // For each spatial dimension, calculate the corresponding input index
                        for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                        {
                            auto dimIdx = static_cast<size_t>(dim);
                            inputSpatialIndices[dimIdx]
                                = (outputSpatialIndices[dimIdx] * strides[dimIdx])
                                  + (kernelSpatialIndices[dimIdx] * dilations[dimIdx])
                                  - prePadding[dimIdx];

                            // In either case, this position does not exist in the logical input tensor.
                            // 1.  (output_idx * stride) + (kernel_idx * dilation) - prePadding < 0
                            //  => (output_idx * stride) + (kernel_idx * dilation) < prePadding
                            // 2.  (output_idx * stride) + (kernel_idx * dilation) - prePadding >= input_dim
                            //  => (output_idx * stride) + (kernel_idx * dilation) >= input_dim + prePadding
                            // It is implicit in Case 2 that the position could be in the postPadding region.
                            if(inputSpatialIndices[dimIdx] < 0
                               || inputSpatialIndices[dimIdx] >= inputSpatialDims[dimIdx])
                            {
                                validPosition = false;
                                break;
                            }
                        }

                        if(validPosition)
                        {
                            // Input dims: [n, inputChannel, ...]
                            // Thus, we index via global input channel index.
                            auto inputFullIndices
                                = buildTensorIndices(nIdx, inputChannel, inputSpatialIndices);

                            // Weight dims: [outputChannels,
                            // inputChannels/groupCount, ...] Thus, we index via flattened output channel index and
                            // group-offset input channel index (c).
                            int64_t weightIdx = (gIdx * outputChannelsPerGroup) + kIdx;
                            auto weightFullIndices
                                = buildTensorIndices(weightIdx, c, kernelSpatialIndices);

                            InputDataType inputVal = input.getHostValue(inputFullIndices);
                            InputDataType weightVal = weight.getHostValue(weightFullIndices);

                            accumulator = accumulator
                                          + (static_cast<AccumulatorType>(inputVal)
                                             * static_cast<AccumulatorType>(weightVal));
                        }
                    });
            }

            int64_t outputChannel = (gIdx * outputChannelsPerGroup) + kIdx;
            auto outputFullIndices = buildTensorIndices(nIdx, outputChannel, outputSpatialIndices);

            output.setHostValue(static_cast<InputDataType>(accumulator), outputFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, outputChannelsPerGroup};
        parallelDims.insert(parallelDims.end(), outputSpatialDims.begin(), outputSpatialDims.end());

        auto parallelFunc
            = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        output.memory().markHostModified();
    }

    // Overload for uniform padding
    static void convBwdData(TensorBase<InputDataType>& gradInput,
                            const TensorBase<InputDataType>& weight,
                            const TensorBase<InputDataType>& gradOutput,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& padding)
    {
        convBwdData(gradInput, weight, gradOutput, strides, dilations, padding, padding);
    }

    static void convBwdData(TensorBase<InputDataType>& gradInput,
                            const TensorBase<InputDataType>& weight,
                            const TensorBase<InputDataType>& gradOutput,
                            const std::vector<int64_t>& strides,
                            const std::vector<int64_t>& dilations,
                            const std::vector<int64_t>& prePadding,
                            const std::vector<int64_t>& postPadding)
    {
        validateInput(gradInput, weight, gradOutput, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NC[spatial...] format for input/output, [G*K][C][spatial...] for weight
        const auto& inputDims = gradInput.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = gradOutput.dims();

        int64_t nBatch = inputDims[0];
        int64_t totalOutputChannels = weightDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = weightDims[1]; // C

        int64_t nSpatialDims = static_cast<int64_t>(inputDims.size()) - 2;
        std::vector<int64_t> inputSpatialDims(inputDims.begin() + 2, inputDims.end());
        std::vector<int64_t> kernelSpatialDims(weightDims.begin() + 2, weightDims.end());
        std::vector<int64_t> outputSpatialDims(outputDims.begin() + 2, outputDims.end());

        // Calculate groups from input/weight channel relationship
        int64_t nInputChannels = inputDims[1];
        int64_t nGroups = nInputChannels / channelsPerGroup; // G
        int64_t outputChannelsPerGroup = totalOutputChannels / nGroups; // K

        // This lambda computes a single element of the input gradient tensor (dx)
        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto cIdx = indices[2]; // channel index within group

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> inputSpatialIndices(indices.begin() + 3, indices.end());

            auto vAcc = static_cast<AccumulatorType>(0.0f);

            // Iterate over each spatial position of the kernel for contributing output gradients
            iterateAlongDimensions(
                kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                    std::vector<int64_t> outputSpatialIndices(static_cast<size_t>(nSpatialDims));
                    bool validPosition = true;

                    // For each spatial dimension, calculate the corresponding output gradient index
                    for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                    {
                        auto dimIdx = static_cast<size_t>(dim);
                        int64_t tmp = inputSpatialIndices[dimIdx] + prePadding[dimIdx]
                                      - (kernelSpatialIndices[dimIdx] * dilations[dimIdx]);

                        // Check if the current input position could have contributed to an output element. If the
                        // remainder is non-zero, this combination is not aligned with the stride, so it's not a valid
                        // mapping from the forward pass.
                        if(tmp % strides[dimIdx] != 0)
                        {
                            validPosition = false;
                            break;
                        }

                        outputSpatialIndices[dimIdx] = tmp / strides[dimIdx];

                        // Check if position does not exist in the logical output tensor.
                        // 1.  (input_idx + prePadding - (kernel_idx * dilation)) / stride < 0
                        //  => numerator < 0 => sampling from a location before the output tensor
                        // 2.  (input_idx + prePadding - (kernel_idx * dilation)) / stride >= output_dim
                        //  => input_idx + prePadding >= (output_dim * stride) + (kernel_idx * dilation) => beyond the output tensor
                        if(outputSpatialIndices[dimIdx] < 0
                           || outputSpatialIndices[dimIdx] >= outputSpatialDims[dimIdx])
                        {
                            validPosition = false;
                            break;
                        }
                    }

                    if(validPosition)
                    {
                        // Iterate over each output channel in the group, as they all contribute to the input gradient
                        for(int64_t k = 0; k < outputChannelsPerGroup; ++k)
                        {
                            auto outputChannelIdx = (gIdx * outputChannelsPerGroup) + k;

                            auto gradOutputFullIndices
                                = buildTensorIndices(nIdx, outputChannelIdx, outputSpatialIndices);

                            auto weightBatchIdx = outputChannelIdx;
                            auto weightChannelIdx = cIdx;
                            auto weightFullIndices = buildTensorIndices(
                                weightBatchIdx, weightChannelIdx, kernelSpatialIndices);

                            InputDataType vOut = gradOutput.getHostValue(gradOutputFullIndices);
                            InputDataType vWei = weight.getHostValue(weightFullIndices);

                            vAcc = vAcc
                                   + (static_cast<AccumulatorType>(vOut)
                                      * static_cast<AccumulatorType>(vWei));
                        }
                    }
                });

            int64_t inputChannelIdx = (gIdx * channelsPerGroup) + cIdx;
            auto gradInputFullIndices
                = buildTensorIndices(nIdx, inputChannelIdx, inputSpatialIndices);

            gradInput.setHostValue(static_cast<InputDataType>(vAcc), gradInputFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, nBatch, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), inputSpatialDims.begin(), inputSpatialDims.end());

        auto parallelFunc
            = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        gradInput.memory().markHostModified();
    }

    static void convBwdWeight(const TensorBase<InputDataType>& input,
                              TensorBase<InputDataType>& gradWeight,
                              const TensorBase<InputDataType>& gradOutput,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& padding)
    {
        convBwdWeight(input, gradWeight, gradOutput, strides, dilations, padding, padding);
    }

    static void convBwdWeight(const TensorBase<InputDataType>& input,
                              TensorBase<InputDataType>& gradWeight,
                              const TensorBase<InputDataType>& gradOutput,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding)
    {
        validateInput(input, gradWeight, gradOutput, strides, dilations, prePadding, postPadding);

        // Extract dimensions - NCHW format for input/output, [G*K][C][Y][X] for weight (4D flattened)
        const auto& inputDims = input.dims();
        const auto& weightDims = gradWeight.dims();
        const auto& outputDims = gradOutput.dims();

        int64_t nBatch = outputDims[0];

        int64_t nSpatialDims = static_cast<int64_t>(inputDims.size()) - 2;
        std::vector<int64_t> inputSpatialDims(inputDims.begin() + 2, inputDims.end());
        std::vector<int64_t> kernelSpatialDims(weightDims.begin() + 2, weightDims.end());
        std::vector<int64_t> outputSpatialDims(outputDims.begin() + 2, outputDims.end());

        int64_t totalOutputChannels = weightDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = weightDims[1]; // C

        // Calculate groups from input/weight channel relationship
        int64_t nInputChannels = inputDims[1];
        int64_t nGroups = nInputChannels / channelsPerGroup; // G
        int64_t outputChannelsPerGroup = totalOutputChannels / nGroups; // K

        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0];
            auto kIdx = indices[1];
            auto cIdx = indices[2];

            // Add 3 because [gIdx, nIdx, cIdx] are the first 3 elements
            std::vector<int64_t> kernelSpatialIndices(indices.begin() + 3, indices.end());

            auto vAcc = static_cast<AccumulatorType>(0.0f);

            // Iterate over each spatial position of the kernel for contributing output gradients
            iterateAlongDimensions(
                outputSpatialDims, [&](const std::vector<int64_t>& outputSpatialIndices) {
                    std::vector<int64_t> inputSpatialIndices(static_cast<size_t>(nSpatialDims));

                    bool validPosition = true;

                    // For each spatial dimension, calculate the corresponding output gradient index
                    for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                    {
                        auto dimIdx = static_cast<size_t>(dim);

                        int64_t tmp = (outputSpatialIndices[dimIdx] * strides[dimIdx])
                                      + (kernelSpatialIndices[dimIdx] * dilations[dimIdx])
                                      - prePadding[dimIdx];

                        inputSpatialIndices[dimIdx] = tmp;

                        if(inputSpatialIndices[dimIdx] < 0
                           || inputSpatialIndices[dimIdx] >= inputSpatialDims[dimIdx])
                        {
                            validPosition = false;
                            break;
                        }
                    }

                    if(validPosition)
                    {
                        for(int64_t n = 0; n < nBatch; ++n)
                        {
                            auto outputChannelIdx = (gIdx * outputChannelsPerGroup) + kIdx;

                            auto gradOutputFullIndices
                                = buildTensorIndices(n, outputChannelIdx, outputSpatialIndices);

                            auto inputChannelIdx = (gIdx * channelsPerGroup) + cIdx;

                            auto inputFullIndices
                                = buildTensorIndices(n, inputChannelIdx, inputSpatialIndices);

                            InputDataType vOut = gradOutput.getHostValue(gradOutputFullIndices);

                            InputDataType vIn = input.getHostValue(inputFullIndices);

                            vAcc = vAcc
                                   + (static_cast<AccumulatorType>(vOut)
                                      * static_cast<AccumulatorType>(vIn));
                        }
                    }
                });

            auto weightN = (gIdx * outputChannelsPerGroup) + kIdx;
            auto weightFullIndices = buildTensorIndices(weightN, cIdx, kernelSpatialIndices);

            gradWeight.setHostValue(static_cast<InputDataType>(vAcc), weightFullIndices);
        };

        // Build dimensions for parallel iteration
        std::vector<int64_t> parallelDims = {nGroups, outputChannelsPerGroup, channelsPerGroup};
        parallelDims.insert(parallelDims.end(), kernelSpatialDims.begin(), kernelSpatialDims.end());

        auto parallelFunc
            = hipdnn_sdk::test_utilities::makeParallelTensorFunctor(convolutionFunc, parallelDims);
        parallelFunc(std::thread::hardware_concurrency());

        gradWeight.memory().markHostModified();
    }

private:
    static void validateInput(const TensorBase<InputDataType>& input,
                              const TensorBase<InputDataType>& weight,
                              const TensorBase<InputDataType>& output,
                              const std::vector<int64_t>& strides,
                              const std::vector<int64_t>& dilations,
                              const std::vector<int64_t>& prePadding,
                              const std::vector<int64_t>& postPadding)
    {
        // Input validation
        if(input.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Input tensor must have at least 3 dimensions (N, C, spatial...)");
        }

        if(output.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Output tensor must have at least 3 dimensions (N, C, spatial...)");
        }

        if(weight.dims().size() < 3)
        {
            throw std::invalid_argument(
                "Weight tensor must have at least 3 dimensions ([G*K], C, spatial...)");
        }

        // Check that all tensors have same number of dimensions
        if(input.dims().size() != output.dims().size()
           || input.dims().size() != weight.dims().size())
        {
            throw std::invalid_argument(
                "Input, output, and weight tensors must have the same number of dimensions");
        }

        int64_t nSpatialDims = static_cast<int64_t>(input.dims().size()) - 2;

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

        const auto& inputDims = input.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = output.dims();

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

            // Validate that output dimensions are correct given the padding
            // Some of this validation could probably be consolidated into the sdk and removed from frontend nodes
            int64_t inputDim = inputDims[idx + 2];
            int64_t kernelDim = weightDims[idx + 2];
            int64_t outputDim = outputDims[idx + 2];

            int64_t kernelSize = (dilations[idx] * (kernelDim - 1)) + 1;
            int64_t expectedOutputDim
                = ((inputDim + prePadding[idx] + postPadding[idx] - kernelSize) / strides[idx]) + 1;

            if(expectedOutputDim != outputDim)
            {
                throw std::invalid_argument(
                    "Output dimension " + std::to_string(outputDim) + " at spatial dimension "
                    + std::to_string(i) + " does not match expected dimension "
                    + std::to_string(expectedOutputDim) + " given the input parameters.");
            }
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
