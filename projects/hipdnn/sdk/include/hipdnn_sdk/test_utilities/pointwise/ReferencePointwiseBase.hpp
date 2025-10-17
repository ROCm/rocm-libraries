// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_sdk/test_utilities/CpuFpReferenceUtilities.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>
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
    template <typename InputType, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<InputType>& input)
    {
        executeUnaryOperation<InputType, ComputeType>(operation, output, input);
    }

    template <typename InputType, typename ParamType, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<InputType>& input,
                                 const ParamType lowerClip,
                                 const ParamType upperClip,
                                 const ParamType lowerSlope)
    {
        static_assert(IS_VALID_TENSOR_TYPE_V<ParamType>,
                      "ParamType must be a valid tensor type for scalar parameters");
        executeParameterizedUnaryOperation<InputType, ParamType, ComputeType>(
            operation, output, input, lowerClip, upperClip, lowerSlope);
    }

    // Binary operations
    template <typename Input1Type, typename Input2Type, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<Input1Type>& input1,
                                 const TensorBase<Input2Type>& input2)
    {
        executeBinaryOperation<Input1Type, Input2Type, ComputeType>(
            operation, output, input1, input2);
    }

    // Parameterized binary operations
    template <typename Input1Type,
              typename Input2Type,
              typename ParamType,
              typename ComputeType = double>
    static void pointwiseCompute(hipdnn_sdk::data_objects::PointwiseMode operation,
                                 TensorBase<OutputType>& output,
                                 const TensorBase<Input1Type>& input1,
                                 const TensorBase<Input2Type>& input2,
                                 const ParamType lowerClip,
                                 const ParamType upperClip,
                                 const ParamType lowerSlope)
    {
        static_assert(IS_VALID_TENSOR_TYPE_V<ParamType>,
                      "ParamType must be a valid tensor type for scalar parameters");
        executeParameterizedBinaryOperation<Input1Type, Input2Type, ParamType, ComputeType>(
            operation, output, input1, input2, lowerClip, upperClip, lowerSlope);
    }

private:
    template <typename InputType, typename ComputeType>
    static void executeUnaryOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                      TensorBase<OutputType>& output,
                                      const TensorBase<InputType>& input)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD:
            policy.executeUnary(input, output, pointwise::ReluForward<ComputeType>{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
            policy.executeUnary(input, output, pointwise::SigmoidForward<ComputeType>{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD:
            policy.executeUnary(input, output, pointwise::TanhForward<ComputeType>{});
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

    template <typename InputType, typename ParamType, typename ComputeType>
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
                input,
                output,
                pointwise::ReluForward<ComputeType>{static_cast<ComputeType>(lowerClip),
                                                    static_cast<ComputeType>(upperClip),
                                                    static_cast<ComputeType>(lowerSlope)});
            break;
        default:
            throw std::runtime_error("Unsupported parameterized pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename Input1Type, typename Input2Type, typename ComputeType>
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
        case hipdnn_sdk::data_objects::PointwiseMode::MUL:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Multiply{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD:
            policy.executeBinaryBroadcast(
                input1, input2, output, pointwise::ReluBackward<ComputeType>{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
            policy.executeBinaryBroadcast(
                input1, input2, output, pointwise::SigmoidBackward<ComputeType>{});
            break;
        case hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD:
            policy.executeBinaryBroadcast(
                input1, input2, output, pointwise::TanhBackward<ComputeType>{});
            break;
        default:
            throw std::runtime_error("Unsupported binary pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename Input1Type, typename Input2Type, typename ParamType, typename ComputeType>
    static void
        executeParameterizedBinaryOperation(hipdnn_sdk::data_objects::PointwiseMode operation,
                                            TensorBase<OutputType>& output,
                                            const TensorBase<Input1Type>& input1,
                                            const TensorBase<Input2Type>& input2,
                                            const ParamType lowerClip,
                                            const ParamType upperClip,
                                            const ParamType lowerSlope)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD:
            policy.executeBinaryBroadcast(input1,
                                          input2,
                                          output,
                                          pointwise::ParameterizedReluBackward<ComputeType>{
                                              static_cast<ComputeType>(lowerClip),
                                              static_cast<ComputeType>(upperClip),
                                              static_cast<ComputeType>(lowerSlope)});
            break;
        default:
            throw std::runtime_error("Unsupported parameterized binary pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    static bool canExecuteUnaryOperation(const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               !attrs->in_1_tensor_uid() && // Must NOT be set
               !attrs->in_2_tensor_uid() && // Must NOT be set
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool
        canExecuteBinaryOperation(const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               attrs->in_1_tensor_uid() && // Must be set
               *attrs->in_1_tensor_uid() != 0 && // Must be non-zero
               !attrs->in_2_tensor_uid() && // Must NOT be set
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool
        canExecuteTernaryOperation(const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               attrs->in_1_tensor_uid() && // Must be set
               *attrs->in_1_tensor_uid() != 0 && // Must be non-zero
               attrs->in_2_tensor_uid() && // Must be set
               *attrs->in_2_tensor_uid() != 0 && // Must be non-zero
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool canExecuteOperation(const hipdnn_sdk::data_objects::PointwiseAttributes* attrs)
    {
        using namespace hipdnn_sdk::data_objects;

        if(attrs == nullptr)
        {
            return false;
        }

        PointwiseMode operation = attrs->operation();

        if(hipdnn_sdk::utilities::isImplementedUnaryPointwiseMode(operation))
        {
            return canExecuteUnaryOperation(attrs);
        }
        if(hipdnn_sdk::utilities::isImplementedBinaryPointwiseMode(operation))
        {
            return canExecuteBinaryOperation(attrs);
        }
        if(hipdnn_sdk::utilities::isImplementedTernaryPointwiseMode(operation))
        {
            return canExecuteTernaryOperation(attrs);
        }

        return false;
    }
};

} // namespace test_utilities
} // namespace hipdnn_sdk
