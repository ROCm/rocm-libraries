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

template <class DataType, class DeviceExecutor>
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

        const auto* pointwise_attrs = node.attributes_as_PointwiseAttributes();
        if(pointwise_attrs == nullptr)
        {
            return false;
        }

        PointwiseMode operation = pointwise_attrs->operation();
        if(!canExecuteOperation(operation, pointwise_attrs))
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

        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinaryBroadcast(inputs, output, pointwise::Add{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinaryBroadcast(inputs, output, pointwise::Subtract{});
            break;
        default:
            throw std::runtime_error("Unsupported pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

private:
    static bool canExecuteOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                    const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
    {
        using namespace hipdnn_sdk::data_objects;

        if(attrs == nullptr)
        {
            return false;
        }

        if(attrs->in_0_tensor_uid() == 0 || attrs->out_0_tensor_uid() == 0)
        {
            return false;
        }

        // Use the same switch statement logic as pointwiseForward
        // This is our single source of truth for supported operations
        switch(operation)
        {
        case PointwiseMode::ADD:
        case PointwiseMode::SUB:
            // Binary operations require second input
            return (attrs->in_1_tensor_uid() != 0);
        default:
            // Any operation not in pointwiseForward is unsupported
            return false;
        }

        // Note: We don't validate tensor dimensions, data types, or broadcasting compatibility here
        // because our N-dimensional broadcasting implementation handles these dynamically at runtime
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
