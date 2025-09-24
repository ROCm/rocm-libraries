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

template <class OutputType, class DeviceExecutor, class... InputTypes>
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

    // Variadic template interface for unary, binary, and ternary operations
    template<typename... Tensors>
    static void pointwiseForward(hipdnn_sdk::data_objects::PointwiseMode operation,
                                Tensors&&... tensors_and_output)
    {
        static_assert(sizeof...(Tensors) >= 2, "Need at least one input and one output tensor");
        static_assert(sizeof...(Tensors) == 3, "Currently only binary operations are supported");
        
        auto args = std::forward_as_tuple(tensors_and_output...);
        
        DeviceExecutor policy;
        
        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinaryBroadcast(std::get<0>(args), std::get<1>(args), std::get<2>(args), pointwise::Add{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinaryBroadcast(std::get<0>(args), std::get<1>(args), std::get<2>(args), pointwise::Subtract{});
            break;
        default:
            throw std::runtime_error("Unsupported pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(std::get<2>(args));
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
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
