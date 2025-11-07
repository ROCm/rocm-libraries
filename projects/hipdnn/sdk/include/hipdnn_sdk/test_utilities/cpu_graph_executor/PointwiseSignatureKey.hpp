// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwisePlan.hpp>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>
#include <hipdnn_sdk/plugin/PluginFlatbufferTypeHelpers.hpp>

namespace hipdnn_sdk::test_utilities
{

struct PointwiseSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes;
    hipdnn_sdk::data_objects::PointwiseMode operation;
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType computeDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;
    hipdnn_sdk::data_objects::DataType input1DataType
        = hipdnn_sdk::data_objects::DataType::UNSET; // For binary ops

    PointwiseSignatureKey() = default;
    constexpr PointwiseSignatureKey(hipdnn_sdk::data_objects::PointwiseMode op,
                                    hipdnn_sdk::data_objects::DataType input,
                                    hipdnn_sdk::data_objects::DataType compute,
                                    hipdnn_sdk::data_objects::DataType output,
                                    hipdnn_sdk::data_objects::DataType input1
                                    = hipdnn_sdk::data_objects::DataType::UNSET)
        : operation(op)
        , inputDataType(input)
        , computeDataType(compute)
        , outputDataType(output)
        , input1DataType(input1)
    {
    }

    PointwiseSignatureKey(
        const hipdnn_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t, const hipdnn_sdk::data_objects::TensorAttributes*>&
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
        if(hipdnn_sdk::utilities::isBinaryPointwiseMode(operation))
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

        // Add plan builders for implemented unary operations
        // FLOAT input/compute/output
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        // HALF input, FLOAT compute, HALF output
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::HALF,
                             hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::HALF>(map);
        // BFLOAT16 input, FLOAT compute, BFLOAT16 output
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                             hipdnn_sdk::data_objects::DataType::FLOAT,
                             hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        // Add plan builders for implemented binary operations
        // HALF input, FLOAT compute, HALF output
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::HALF,
                              hipdnn_sdk::data_objects::DataType::FLOAT,
                              hipdnn_sdk::data_objects::DataType::HALF>(map);
        // BFLOAT16 input, FLOAT compute, BFLOAT16 output
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_sdk::data_objects::DataType::FLOAT,
                              hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        // Add plan builders for implemented binary operations
        // HALF input, FLOAT compute, HALF output
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::HALF,
                              hipdnn_sdk::data_objects::DataType::FLOAT,
                              hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        // BFLOAT16 input, FLOAT compute, BFLOAT16 output
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::BFLOAT16,
                              hipdnn_sdk::data_objects::DataType::FLOAT,
                              hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        

        return map;
    }

private:
    template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addUnaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        // Add all implemented unary operations
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(
            map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::ABS, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::NEG, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
    }

    template <hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addBinaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                         std::unique_ptr<IGraphNodePlanBuilder>,
                                                         PointwiseSignatureKey>& map)
    {
        // Add all implemented binary operations
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::ADD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SUB, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::MUL, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(
            map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum>(map);
    }

    template <hipdnn_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addUnaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                       std::unique_ptr<IGraphNodePlanBuilder>,
                                                       PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(ModeEnum, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<OutputDataTypeEnum>>();
    }

    template <hipdnn_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_sdk::data_objects::DataType ComputeDataTypeEnum,
              hipdnn_sdk::data_objects::DataType OutputDataTypeEnum>
    static void addBinaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(ModeEnum, InputDataTypeEnum, ComputeDataTypeEnum, OutputDataTypeEnum, InputDataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<OutputDataTypeEnum>>();
    }
};
}
