// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <vector>

namespace hipdnn_sdk
{
namespace test_utilities
{

using namespace hipdnn_sdk::utilities;

template <class DataType, class ExecutionPolicy>
class ReferencePointwiseBase
{
public:
    static bool isApplicable(const hipdnn_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_sdk::data_objects;

        if(node.attributes_type() != NodeAttributes::PointwiseAttributes)
        {
            return false;
        }

        return true;
    }

    static void pointwiseForward(const std::vector<const TensorBase<DataType>*>& inputs,
                                 TensorBase<DataType>& output,
                                 hipdnn_sdk::data_objects::PointwiseMode operation)
    {
        if(inputs.empty())
        {
            throw std::runtime_error("Pointwise operation requires at least one input tensor.");
        }

        validateTensorDimensions(inputs, output);

        ExecutionPolicy policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinary(inputs, output, pointwise::Add{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinary(inputs, output, pointwise::Subtract{});
            break;
        default:
            throw std::runtime_error("Unsupported pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

private:
    static void validateTensorDimensions(const std::vector<const TensorBase<DataType>*>& inputs,
                                         const TensorBase<DataType>& output)
    {
        if(inputs.empty())
        {
            throw std::runtime_error("No input tensors provided.");
        }

        const auto& outputDims = output.dims();

        for(size_t i = 0; i < inputs.size(); ++i)
        {
            if(inputs[i] == nullptr)
            {
                throw std::runtime_error("Input tensor " + std::to_string(i) + " is null.");
            }

            const auto& inputDims = inputs[i]->dims();
            if(inputDims != outputDims)
            {
                throw std::runtime_error("Input tensor " + std::to_string(i)
                                         + " dimensions do not match output tensor dimensions.");
            }
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
