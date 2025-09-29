// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <stdexcept>
#include <tuple>

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

    template <typename... Args>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 Args&&... args)
    {
        static_assert(sizeof...(Args) >= 1, "Need at least one input tensor");

        auto inputArgs = std::forward_as_tuple(args...);
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            if constexpr(sizeof...(Args) == 2)
            {
                policy.executeBinaryBroadcast(
                    std::get<0>(inputArgs), std::get<1>(inputArgs), output, pointwise::Add{});
            }
            else
            {
                throw std::runtime_error("Binary operations require exactly 2 input tensors");
            }
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            if constexpr(sizeof...(Args) == 2)
            {
                policy.executeBinaryBroadcast(
                    std::get<0>(inputArgs), std::get<1>(inputArgs), output, pointwise::Subtract{});
            }
            else
            {
                throw std::runtime_error("Binary operations require exactly 2 input tensors");
            }
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD:
            if constexpr(sizeof...(Args) == 4)
            {
                policy.executeUnary(std::get<0>(inputArgs),
                                    output,
                                    pointwise::ReluForward{std::get<1>(inputArgs),
                                                           std::get<2>(inputArgs),
                                                           std::get<3>(inputArgs)});
            }
            else
            {
                policy.executeUnary(std::get<0>(inputArgs), output, pointwise::ReluForward{});
            }
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD:
            if constexpr(sizeof...(Args) == 4)
            {
                policy.executeUnary(std::get<0>(inputArgs),
                                    output,
                                    pointwise::ParameterizedReluBackward{std::get<1>(inputArgs),
                                                                         std::get<2>(inputArgs),
                                                                         std::get<3>(inputArgs)});
            }
            else if constexpr(sizeof...(Args) == 1)
            {
                policy.executeUnary(std::get<0>(inputArgs), output, pointwise::ReluBackward{});
            }
            else
            {
                throw std::runtime_error(
                    "RELU_BWD requires either 1 input tensor (default parameters) or 4 arguments "
                    "(input + lowerClip + upperClip + lowerSlope)");
            }
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::SigmoidForward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::SigmoidBackward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::TanhForward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::TanhBackward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::ABS:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::AbsoluteValue{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::NEG:
            policy.executeUnary(std::get<0>(inputArgs), output, pointwise::Negation{});
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
        case PointwiseMode::RELU_FWD:
        case PointwiseMode::RELU_BWD:
        case PointwiseMode::SIGMOID_FWD:
        case PointwiseMode::SIGMOID_BWD:
        case PointwiseMode::TANH_FWD:
        case PointwiseMode::TANH_BWD:
        case PointwiseMode::ABS:
        case PointwiseMode::NEG:
            return true;
        default:
            return false;
        }
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
