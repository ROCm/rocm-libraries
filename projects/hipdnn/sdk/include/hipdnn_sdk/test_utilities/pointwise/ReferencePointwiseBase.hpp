// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
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

    // Unary operations
    template <typename InputType>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<InputType>& input)
    {
        executeUnaryOperation(operation, output, input);
    }

    template <typename InputType, typename ParamType>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<InputType>& input,
                                 const ParamType lowerClip,
                                 const ParamType upperClip,
                                 const ParamType lowerSlope)
    {
        static_assert(IS_VALID_TENSOR_TYPE_V<ParamType>,
                      "ParamType must be a valid tensor type for scalar parameters");
        executeParameterizedUnaryOperation(
            operation, output, input, lowerClip, upperClip, lowerSlope);
    }

    // Binary operations
    template <typename Input1Type, typename Input2Type>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<Input1Type>& input1,
                                 const TensorBase<Input2Type>& input2)
    {
        executeBinaryOperation(operation, output, input1, input2);
    }

private:
    template <typename InputType>
    static void executeUnaryOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                      TensorBase<OutputType>& output,
                                      const TensorBase<InputType>& input)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD:
            policy.executeUnary(input, output, pointwise::ReluForward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD:
            policy.executeUnary(input, output, pointwise::ReluBackward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
            policy.executeUnary(input, output, pointwise::SigmoidForward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
            policy.executeUnary(input, output, pointwise::SigmoidBackward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD:
            policy.executeUnary(input, output, pointwise::TanhForward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD:
            policy.executeUnary(input, output, pointwise::TanhBackward{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::ABS:
            policy.executeUnary(input, output, pointwise::AbsoluteValue{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::NEG:
            policy.executeUnary(input, output, pointwise::Negation{});
            break;
        default:
            throw std::runtime_error("Unsupported unary pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename InputType, typename ParamType>
    static void
        executeParameterizedUnaryOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                           TensorBase<OutputType>& output,
                                           const TensorBase<InputType>& input,
                                           const ParamType lowerClip,
                                           const ParamType upperClip,
                                           const ParamType lowerSlope)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD:
            policy.executeUnary(
                input, output, pointwise::ReluForward{lowerClip, upperClip, lowerSlope});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD:
            policy.executeUnary(
                input,
                output,
                pointwise::ParameterizedReluBackward{lowerClip, upperClip, lowerSlope});
            break;
        default:
            throw std::runtime_error("Unsupported parameterized pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename Input1Type, typename Input2Type>
    static void executeBinaryOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                       TensorBase<OutputType>& output,
                                       const TensorBase<Input1Type>& input1,
                                       const TensorBase<Input2Type>& input2)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Add{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Subtract{});
            break;
        default:
            throw std::runtime_error("Unsupported binary pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

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
