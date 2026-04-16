// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/pointwise/PointwiseOperationFunctors.hpp>
#include <stdexcept>
#include <tuple>

namespace hipdnn_test_sdk::utilities
{

template <class DeviceExecutor, class OutputType, class... InputTypes>
class ReferencePointwiseBase
{
public:
    static bool isApplicable(const hipdnn_flatbuffers_sdk::data_objects::Node& node)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

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
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 const hipdnn_data_sdk::utilities::TensorBase<InputType>& input)
    {
        executeUnaryOperation<InputType, ComputeType>(operation, output, input);
    }

    template <typename InputType, typename ParamType, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 const hipdnn_data_sdk::utilities::TensorBase<InputType>& input,
                                 const ParamType lowerClip,
                                 const ParamType upperClip,
                                 const ParamType lowerSlope,
                                 const ParamType swishBeta = ParamType{1})
    {
        static_assert(hipdnn_test_sdk::detail::IS_VALID_TENSOR_TYPE_V<ParamType>,
                      "ParamType must be a valid tensor type for scalar parameters");
        executeParameterizedUnaryOperation<InputType, ParamType, ComputeType>(
            operation, output, input, lowerClip, upperClip, lowerSlope, swishBeta);
    }

    // Binary operations
    template <typename Input1Type, typename Input2Type, typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 const hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
                                 const hipdnn_data_sdk::utilities::TensorBase<Input2Type>& input2)
    {
        executeBinaryOperation<Input1Type, Input2Type, ComputeType>(
            operation, output, input1, input2);
    }

    // Parameterized binary operations
    template <typename Input1Type,
              typename Input2Type,
              typename ParamType,
              typename ComputeType = double>
    static void pointwiseCompute(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                                 hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                                 const hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
                                 const hipdnn_data_sdk::utilities::TensorBase<Input2Type>& input2,
                                 const ParamType lowerClip,
                                 const ParamType upperClip,
                                 const ParamType lowerSlope)
    {
        static_assert(hipdnn_test_sdk::detail::IS_VALID_TENSOR_TYPE_V<ParamType>,
                      "ParamType must be a valid tensor type for scalar parameters");
        executeParameterizedBinaryOperation<Input1Type, Input2Type, ParamType, ComputeType>(
            operation, output, input1, input2, lowerClip, upperClip, lowerSlope);
    }

private:
    template <typename InputType, typename ComputeType>
    static void
        executeUnaryOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                              hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                              const hipdnn_data_sdk::utilities::TensorBase<InputType>& input)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
            policy.executeUnary(input, output, pointwise::ReluForward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD:
            policy.executeUnary(input, output, pointwise::SigmoidForward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD:
            policy.executeUnary(input, output, pointwise::TanhForward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS:
            policy.executeUnary(input, output, pointwise::AbsoluteValue{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG:
            policy.executeUnary(input, output, pointwise::Negation{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::IDENTITY:
            policy.executeUnary(input, output, pointwise::Identity{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD:
            policy.executeUnary(input, output, pointwise::GeluForward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD:
            policy.executeUnary(input, output, pointwise::GeluApproxTanhForward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD:
            policy.executeUnary(input, output, pointwise::SwishForward<ComputeType>{});
            break;
        default:
            throw std::runtime_error("Unsupported unary pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename InputType, typename ParamType, typename ComputeType>
    static void executeParameterizedUnaryOperation(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
        hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
        const hipdnn_data_sdk::utilities::TensorBase<InputType>& input,
        const ParamType lowerClip,
        const ParamType upperClip,
        const ParamType lowerSlope,
        const ParamType swishBeta)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD:
            policy.executeUnary(
                input,
                output,
                pointwise::ReluForward<ComputeType>{static_cast<ComputeType>(lowerClip),
                                                    static_cast<ComputeType>(upperClip),
                                                    static_cast<ComputeType>(lowerSlope)});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD:
            policy.executeUnary(
                input,
                output,
                pointwise::SwishForward<ComputeType>{static_cast<ComputeType>(swishBeta)});
            break;
        default:
            throw std::runtime_error("Unsupported parameterized pointwise operation: "
                                     + std::to_string(static_cast<int>(operation)));
        }

        policy.markOutputModified(output);
    }

    template <typename Input1Type, typename Input2Type, typename ComputeType>
    static void
        executeBinaryOperation(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
                               hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
                               const hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
                               const hipdnn_data_sdk::utilities::TensorBase<Input2Type>& input2)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Add{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Subtract{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL:
            policy.executeBinaryBroadcast(input1, input2, output, pointwise::Multiply{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
            policy.executeBinaryBroadcast(
                input1, input2, output, pointwise::ReluBackward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD:
            policy.executeBinaryBroadcast(
                input1, input2, output, pointwise::SigmoidBackward<ComputeType>{});
            break;
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD:
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
    static void executeParameterizedBinaryOperation(
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation,
        hipdnn_data_sdk::utilities::TensorBase<OutputType>& output,
        const hipdnn_data_sdk::utilities::TensorBase<Input1Type>& input1,
        const hipdnn_data_sdk::utilities::TensorBase<Input2Type>& input2,
        const ParamType lowerClip,
        const ParamType upperClip,
        const ParamType lowerSlope)
    {
        DeviceExecutor policy;

        switch(operation)
        {
        case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD:
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

    static bool canExecuteUnaryOperation(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               !attrs->in_1_tensor_uid() && // Must NOT be set
               !attrs->in_2_tensor_uid() && // Must NOT be set
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool canExecuteBinaryOperation(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               attrs->in_1_tensor_uid() && // Must be set
               *attrs->in_1_tensor_uid() != 0 && // Must be non-zero
               !attrs->in_2_tensor_uid() && // Must NOT be set
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool canExecuteTernaryOperation(
        const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs)
    {
        return attrs->in_0_tensor_uid() != 0 && // Required: first input
               attrs->in_1_tensor_uid() && // Must be set
               *attrs->in_1_tensor_uid() != 0 && // Must be non-zero
               attrs->in_2_tensor_uid() && // Must be set
               *attrs->in_2_tensor_uid() != 0 && // Must be non-zero
               attrs->out_0_tensor_uid() != 0; // Required: output
    }

    static bool
        canExecuteOperation(const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes* attrs)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;

        if(attrs == nullptr)
        {
            return false;
        }

        const PointwiseMode operation = attrs->operation();
        const auto dataSdkOperation
            = static_cast<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode>(operation);

        if(hipdnn_flatbuffers_sdk::utilities::isImplementedUnaryPointwiseMode(dataSdkOperation))
        {
            return canExecuteUnaryOperation(attrs);
        }
        if(hipdnn_flatbuffers_sdk::utilities::isImplementedBinaryPointwiseMode(dataSdkOperation))
        {
            return canExecuteBinaryOperation(attrs);
        }
        if(hipdnn_flatbuffers_sdk::utilities::isImplementedTernaryPointwiseMode(dataSdkOperation))
        {
            return canExecuteTernaryOperation(attrs);
        }

        return false;
    }
};

} // namespace hipdnn_test_sdk::utilities
