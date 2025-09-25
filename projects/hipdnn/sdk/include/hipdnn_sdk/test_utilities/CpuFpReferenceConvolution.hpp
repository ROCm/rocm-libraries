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

        if(node.attributes_type() != NodeAttributes::ConvolutionFwdAttributes)
        {
            return false;
        }

        return true;
    }

    static void convFwdInference(const TensorBase<InputDataType>& input,
                                 const TensorBase<InputDataType>& weight,
                                 TensorBase<InputDataType>& output,
                                 const std::vector<int64_t>& strides,
                                 const std::vector<int64_t>& dilations,
                                 const std::vector<int64_t>& padding)
    {
        // Input validation
        if(input.dims().size() != 4)
        {
            throw std::invalid_argument("Input tensor must be 4D (NCHW format)");
        }

        if(weight.dims().size() != 4)
        {
            throw std::invalid_argument("Weight tensor must be 4D ([G*K][C][Y][X] format)");
        }

        if(strides.size() != 2)
        {
            throw std::invalid_argument("Strides must have exactly 2 elements [H, W]");
        }

        if(dilations.size() != 2)
        {
            throw std::invalid_argument("Dilations must have exactly 2 elements [H, W]");
        }

        if(padding.size() != 2)
        {
            throw std::invalid_argument("Padding must have exactly 2 elements [H, W]");
        }

        if(strides[0] <= 0 || strides[1] <= 0)
        {
            throw std::invalid_argument("Stride values must be positive");
        }

        if(dilations[0] <= 0 || dilations[1] <= 0)
        {
            throw std::invalid_argument("Dilation values must be positive");
        }

        if(padding[0] < 0 || padding[1] < 0)
        {
            throw std::invalid_argument("Padding values must be non-negative");
        }

        // Extract dimensions - NCHW format for input/output, [G*K][C][Y][X] for weight (4D flattened)
        const auto& inputDims = input.dims();
        const auto& weightDims = weight.dims();
        const auto& outputDims = output.dims();

        int64_t nBatch = inputDims[0];
        int64_t nInputChannels = inputDims[1];
        int64_t inputHeight = inputDims[2];
        int64_t inputWidth = inputDims[3];

        int64_t totalOutputChannels = weightDims[0]; // G * K (flattened)
        int64_t channelsPerGroup = weightDims[1]; // C
        int64_t kernelHeight = weightDims[2]; // Y
        int64_t kernelWidth = weightDims[3]; // X

        int64_t outputHeight = outputDims[2];
        int64_t outputWidth = outputDims[3];

        // Calculate groups from input/weight channel relationship
        int64_t nGroups = nInputChannels / channelsPerGroup;
        int64_t outputChannelsPerGroup = totalOutputChannels / nGroups;

        // Extract convolution parameters
        int64_t strideH = strides[0];
        int64_t strideW = strides[1];
        int64_t dilationH = dilations[0];
        int64_t dilationW = dilations[1];
        int64_t padH = padding[0];
        int64_t padW = padding[1];

        auto convolutionFunc = [&](auto g, auto n, auto k, auto ho, auto wo) {
            AccumulatorType accumulator = static_cast<AccumulatorType>(0);

            int64_t gIdx = static_cast<int64_t>(g);
            int64_t nIdx = static_cast<int64_t>(n);
            int64_t kIdx = static_cast<int64_t>(k);
            int64_t hoIdx = static_cast<int64_t>(ho);
            int64_t woIdx = static_cast<int64_t>(wo);

            int64_t baseInputChannel = gIdx * channelsPerGroup;

            for(int64_t c = 0; c < channelsPerGroup; ++c)
            {
                int64_t inputChannel = baseInputChannel + c;

                for(int64_t y = 0; y < kernelHeight; ++y)
                {
                    int64_t hi = hoIdx * strideH + y * dilationH - padH;

                    for(int64_t x = 0; x < kernelWidth; ++x)
                    {
                        int64_t wi = woIdx * strideW + x * dilationW - padW;

                        if(hi >= 0 && hi < inputHeight && wi >= 0 && wi < inputWidth)
                        {
                            InputDataType inputVal = input.getHostValue(nIdx, inputChannel, hi, wi);

                            int64_t weightIdx = gIdx * outputChannelsPerGroup + kIdx;
                            InputDataType weightVal = weight.getHostValue(weightIdx, c, y, x);

                            accumulator += static_cast<AccumulatorType>(inputVal)
                                           * static_cast<AccumulatorType>(weightVal);
                        }
                    }
                }
            }

            int64_t outputChannel = gIdx * outputChannelsPerGroup + kIdx;
            output.setHostValue(
                static_cast<InputDataType>(accumulator), nIdx, outputChannel, hoIdx, woIdx);
        };

        hipdnn_sdk::test_utilities::makeParallelTensorFunctor(
            convolutionFunc, nGroups, nBatch, outputChannelsPerGroup, outputHeight, outputWidth)(
            std::thread::hardware_concurrency());

        output.memory().markHostModified();
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
