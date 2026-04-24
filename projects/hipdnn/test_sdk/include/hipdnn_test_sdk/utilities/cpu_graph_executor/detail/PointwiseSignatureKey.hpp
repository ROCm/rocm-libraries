// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwisePlan.hpp>
#include <ostream>

namespace hipdnn_test_sdk::detail
{

struct PointwiseSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation;
    hipdnn_flatbuffers_sdk::data_objects::DataType inputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType input1DataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET; // For binary ops

    PointwiseSignatureKey() = default;
    constexpr PointwiseSignatureKey(hipdnn_flatbuffers_sdk::data_objects::PointwiseMode op,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType input,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType compute,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                    hipdnn_flatbuffers_sdk::data_objects::DataType input1
                                    = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        : operation(op)
        , inputDataType(input)
        , computeDataType(compute)
        , outputDataType(output)
        , input1DataType(input1)
    {
    }

    PointwiseSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap)
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to PointwiseAttributes");
        }

        operation = nodeAttributes->operation();

        // Get input tensor (always present)
        auto input0TensorAttr = tensorMap.at(nodeAttributes->in_0_tensor_uid());
        if(input0TensorAttr == nullptr)
        {
            throw std::runtime_error("Input tensor attributes could not be found in the map");
        }
        inputDataType = input0TensorAttr->data_type();

        // Get compute data type from node
        computeDataType = node.compute_data_type();

        // Get output tensor (always present)
        auto outputTensorAttr = tensorMap.at(nodeAttributes->out_0_tensor_uid());
        if(outputTensorAttr == nullptr)
        {
            throw std::runtime_error("Output tensor attributes could not be found in the map");
        }
        outputDataType = outputTensorAttr->data_type();

        // Get second input tensor if this is a binary operation
        if(hipdnn_flatbuffers_sdk::utilities::isBinaryPointwiseMode(operation))
        {
            if(nodeAttributes->in_1_tensor_uid().has_value())
            {
                auto input1TensorAttr = tensorMap.at(nodeAttributes->in_1_tensor_uid().value());
                if(input1TensorAttr == nullptr)
                {
                    throw std::runtime_error(
                        "Second input tensor attributes could not be found in the map");
                }
                input1DataType = input1TensorAttr->data_type();
            }
            else
            {
                throw std::runtime_error("Binary operation missing second input tensor");
            }
        }
    }

    std::size_t operator()(const PointwiseSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(operation)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(inputDataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 16)
               ^ (static_cast<std::size_t>(static_cast<int>(input1DataType)) << 20);
    }

    bool operator==(const PointwiseSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && operation == other.operation
               && inputDataType == other.inputDataType && computeDataType == other.computeDataType
               && outputDataType == other.outputDataType && input1DataType == other.input1DataType;
    }

    static std::unordered_map<PointwiseSignatureKey,
                              std::unique_ptr<IGraphNodePlanBuilder>,
                              PointwiseSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<PointwiseSignatureKey,
                           std::unique_ptr<IGraphNodePlanBuilder>,
                           PointwiseSignatureKey>
            map;
        // Add plan builders for implemented unary operations (input/compute/output)
        addUnaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addUnaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addUnaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addUnaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addUnaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                             hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);

        // Add plan builders for implemented binary operations (input0/input1/compute/output)
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                              hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        return map;
    }

private:
    template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addUnaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        // Add all implemented unary operations
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_FWD,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_FWD,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ABS,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::NEG,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_FWD,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<
            hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::GELU_APPROX_TANH_FWD,
            InputDataTypeEnum,
            ComputeDataTypeEnum,
            OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD,
                            InputDataTypeEnum,
                            ComputeDataTypeEnum,
                            OutputDataTypeEnum>(map);
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType Input0DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType Input1DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addBinaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                         std::unique_ptr<IGraphNodePlanBuilder>,
                                                         PointwiseSignatureKey>& map)
    {
        // Add all implemented binary operations
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SUB,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::MUL,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_BWD,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SIGMOID_BWD,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::TANH_BWD,
                             Input0DataTypeEnum,
                             Input1DataTypeEnum,
                             ComputeDataTypeEnum,
                             OutputDataTypeEnum>(map);
    }

    template <hipdnn_flatbuffers_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addUnaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                       std::unique_ptr<IGraphNodePlanBuilder>,
                                                       PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(
            ModeEnum, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<InputDataTypeEnum,
                                                    InputDataTypeEnum,
                                                    ComputeDataTypeEnum,
                                                    OutputDataTypeEnum>>();
    }

    template <hipdnn_flatbuffers_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType Input0DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType Input1DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addBinaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(ModeEnum,
                                  Input0DataTypeEnum,
                                  ComputeDataTypeEnum,
                                  OutputDataTypeEnum,
                                  Input1DataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<Input0DataTypeEnum,
                                                    Input1DataTypeEnum,
                                                    ComputeDataTypeEnum,
                                                    OutputDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const PointwiseSignatureKey& key)
{
    if(key.input1DataType != hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
    {
        // Binary operation
        os << "Pointwise(op=" << key.operation << ", in0=" << key.inputDataType
           << ", in1=" << key.input1DataType << ", compute=" << key.computeDataType
           << ", out=" << key.outputDataType << ")";
    }
    else
    {
        // Unary operation
        os << "Pointwise(op=" << key.operation << ", in=" << key.inputDataType
           << ", compute=" << key.computeDataType << ", out=" << key.outputDataType << ")";
    }
    return os;
}

} // namespace hipdnn_test_sdk::detail
