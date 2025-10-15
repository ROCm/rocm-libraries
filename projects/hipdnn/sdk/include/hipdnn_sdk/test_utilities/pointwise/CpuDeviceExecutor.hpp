// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class OutputType, class... InputTypes>
class CpuDeviceExecutor
{
public:
    template <typename Op, typename InputType>
    void executeUnary(const TensorBase<InputType>& input, TensorBase<OutputType>& output, Op op)
    {
        const auto& inputDims = input.dims();
        const auto& outputDims = output.dims();

        // Validate that input and output dimensions are compatible
        if(!areDimensionsBroadcastCompatible(inputDims, outputDims))
        {
            throw std::runtime_error("Input dimensions are not broadcast compatible with output");
        }

        // Use output dimensions as the broadcast shape
        const auto& broadcastShape = outputDims;

        auto func = [&](const std::vector<int64_t>& indices) {
            // Get broadcasted index for input
            auto inputIndices = getBroadcastableIndex(indices, inputDims);

            // Get value from input tensor and apply operation
            auto inputValue = input.getHostValue(inputIndices);

            // Apply operation and set output
            auto result = op(inputValue);
            output.setHostValue(static_cast<OutputType>(result), indices);
        };

        auto parallelFunc = makeParallelTensorFunctor(func, broadcastShape);
        parallelFunc();
    }

    template <typename Op, typename Input1Type, typename Input2Type>
    void executeBinaryBroadcast(const TensorBase<Input1Type>& input1,
                                const TensorBase<Input2Type>& input2,
                                TensorBase<OutputType>& output,
                                Op op)
    {
        const auto& input1Dims = input1.dims();
        const auto& input2Dims = input2.dims();
        const auto& outputDims = output.dims();

        // Validate broadcast compatibility using existing utility function
        if(!areDimensionsBroadcastCompatible(input1Dims, outputDims))
        {
            throw std::runtime_error("Input1 dimensions are not broadcast compatible with output");
        }

        if(!areDimensionsBroadcastCompatible(input2Dims, outputDims))
        {
            throw std::runtime_error("Input2 dimensions are not broadcast compatible with output");
        }

        // Use output dimensions as the broadcast shape
        const auto& broadcastShape = outputDims;

        auto func = [&](const std::vector<int64_t>& indices) {
            // Get broadcasted indices for each input
            auto input1Indices = getBroadcastableIndex(indices, input1Dims);
            auto input2Indices = getBroadcastableIndex(indices, input2Dims);

            // Get values from input tensors and apply operation
            auto input1Value = input1.getHostValue(input1Indices);
            auto input2Value = input2.getHostValue(input2Indices);

            // Apply operation and set output
            auto result = op(input1Value, input2Value);
            output.setHostValue(static_cast<OutputType>(result), indices);
        };

        auto parallelFunc = makeParallelTensorFunctor(func, broadcastShape);
        parallelFunc();
    }

    void markOutputModified(TensorBase<OutputType>& output)
    {
        output.memory().markHostModified();
    }

private:
    static std::vector<int64_t> getBroadcastableIndex(const std::vector<int64_t>& broadcastIndex,
                                                      const std::vector<int64_t>& tensorDims)
    {
        if(broadcastIndex.size() < tensorDims.size())
        {
            throw std::runtime_error("Broadcast index has fewer dimensions than tensor");
        }

        std::vector<int64_t> broadcastedIndex(tensorDims.size());

        size_t dimOffset = broadcastIndex.size() - tensorDims.size();

        for(size_t i = 0; i < tensorDims.size(); ++i)
        {
            size_t broadcastDimIdx = dimOffset + i;
            broadcastedIndex[i] = (tensorDims[i] == 1) ? 0 : broadcastIndex[broadcastDimIdx];
        }

        return broadcastedIndex;
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
