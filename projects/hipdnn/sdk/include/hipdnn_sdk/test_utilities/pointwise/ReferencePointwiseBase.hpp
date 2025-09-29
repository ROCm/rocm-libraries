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

template <class DeviceExecutor, class OutputType, class... InputTypes>
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

        const auto* pointwiseAttrs = node.attributes_as_PointwiseAttributes();
        if(pointwiseAttrs == nullptr)
        {
            return false;
        }

        if(!canExecuteOperation(pointwiseAttrs))
        {
            return false;
        }

        return true;
    }

    template <typename... Tensors>
    static void pointwiseForward(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 Tensors&&... inputs)
    {
        static_assert(sizeof...(Tensors) >= 1, "Need at least one input tensor");
        static_assert(sizeof...(Tensors) == 2, "Currently only binary operations are supported");

        auto inputArgs = std::forward_as_tuple(inputs...);

        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinaryBroadcast(
                std::get<0>(inputArgs), std::get<1>(inputArgs), output, pointwise::Add{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinaryBroadcast(
                std::get<0>(inputArgs), std::get<1>(inputArgs), output, pointwise::Subtract{});
            break;
        default:
            throw std::runtime_error("Unsupported pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

private:
    static bool canExecuteOperation(const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
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

        PointwiseMode operation = attrs->operation();
        switch(operation)
        {
        case PointwiseMode::ADD:
        case PointwiseMode::SUB:
            // Binary operations require second input
            // Check if nullable field is set and has a non-zero value
            return (attrs->in_1_tensor_uid() && *attrs->in_1_tensor_uid() != 0);
        default:
            // Any operation not in pointwiseForward is unsupported
            return false;
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
