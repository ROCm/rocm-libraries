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

    // Backward compatibility
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

        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto kIdx = indices[2]; // output channel within group

            std::vector<int64_t> outputSpatialIndices(indices.begin() + 3, indices.end());

            auto accumulator = static_cast<AccumulatorType>(0);
            int64_t baseInputChannel = gIdx * channelsPerGroup;

            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                int64_t inputChannel = baseInputChannel + c;

                // Iterate kernel spatial positions
                iterateSpatialPositions(
                    kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                        std::vector<int64_t> inputSpatialIndices(static_cast<size_t>(nSpatialDims));
                        bool validPosition = true;

                        for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                        {
                            auto dimIdx = static_cast<size_t>(dim);
                            inputSpatialIndices[dimIdx]
                                = (outputSpatialIndices[dimIdx] * strides[dimIdx])
                                  + (kernelSpatialIndices[dimIdx] * dilations[dimIdx])
                                  - prePadding[dimIdx];

                            if(inputSpatialIndices[dimIdx] < 0
                               || inputSpatialIndices[dimIdx] >= inputSpatialDims[dimIdx])
                            {
                                validPosition = false;
                                break;
                            }
                        }

                        if(validPosition)
                        {
                            auto inputFullIndices
                                = buildTensorIndices(nIdx, inputChannel, inputSpatialIndices);

                            int64_t weightIdx = (gIdx * outputChannelsPerGroup) + kIdx;
                            auto weightFullIndices
                                = buildTensorIndices(weightIdx, c, kernelSpatialIndices);

                            InputDataType inputVal = input.getHostValue(inputFullIndices);
                            InputDataType weightVal = weight.getHostValue(weightFullIndices);

                            accumulator += static_cast<AccumulatorType>(inputVal)
                                           * static_cast<AccumulatorType>(weightVal);
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

    // Backward compatibility
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

        int64_t nBatch = outputDims[0];
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

        auto convolutionFunc = [&](const std::vector<int64_t>& indices) {
            auto gIdx = indices[0]; // group index
            auto nIdx = indices[1]; // batch index
            auto cIdx = indices[2]; // channel index within group

            std::vector<int64_t> inputSpatialIndices(indices.begin() + 3, indices.end());

            AccumulatorType vAcc = 0;

            iterateSpatialPositions(
                kernelSpatialDims, [&](const std::vector<int64_t>& kernelSpatialIndices) {
                    std::vector<int64_t> outputSpatialIndices(static_cast<size_t>(nSpatialDims));
                    bool validPosition = true;

                    for(int64_t dim = 0; dim < nSpatialDims; ++dim)
                    {
                        auto dimIdx = static_cast<size_t>(dim);
                        int64_t tmp = inputSpatialIndices[dimIdx] + prePadding[dimIdx]
                                      - (kernelSpatialIndices[dimIdx] * dilations[dimIdx]);

                        if(tmp % strides[dimIdx] != 0)
                        {
                            validPosition = false;
                            break;
                        }

                        outputSpatialIndices[dimIdx] = tmp / strides[dimIdx];

                        if(outputSpatialIndices[dimIdx] < 0
                           || outputSpatialIndices[dimIdx] >= outputSpatialDims[dimIdx])
                        {
                            validPosition = false;
                            break;
                        }
                    }

                    if(validPosition)
                    {
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

                            vAcc += static_cast<AccumulatorType>(vOut)
                                    * static_cast<AccumulatorType>(vWei);
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

private:
    static std::vector<int64_t> buildTensorIndices(int64_t batchIdx,
                                                   int64_t channelIdx,
                                                   const std::vector<int64_t>& spatialIndices)
    {
        std::vector<int64_t> fullIndices = {batchIdx, channelIdx};
        fullIndices.insert(fullIndices.end(), spatialIndices.begin(), spatialIndices.end());
        return fullIndices;
    }

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
        }
    }

    // Helper function to iterate over spatial positions
    static void
        iterateSpatialPositions(const std::vector<int64_t>& spatialDims,
                                const std::function<void(const std::vector<int64_t>&)>& func)
    {
        if(spatialDims.empty())
        {
            func({});
            return;
        }

        int64_t totalElements = 1;
        for(auto dim : spatialDims)
        {
            totalElements *= dim;
        }

        std::vector<int64_t> indices(spatialDims.size(), 0);

        for(int64_t iter = 0; iter < totalElements; ++iter)
        {
            func(indices);

            for(int dim = static_cast<int>(spatialDims.size()) - 1; dim >= 0; --dim)
            {
                auto dimIdx = static_cast<size_t>(dim);
                indices[dimIdx]++;

                if(indices[dimIdx] < spatialDims[dimIdx])
                {
                    break;
                }

                indices[dimIdx] = 0;
            }
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
