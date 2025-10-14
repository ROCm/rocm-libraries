// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/PointwisePlan.hpp>
#include <hipdnn_sdk/utilities/PointwiseValidation.hpp>

namespace hipdnn_sdk::test_utilities
{

struct PointwiseSignatureKey
{
    const hipdnn_sdk::data_objects::NodeAttributes nodeType
        = hipdnn_sdk::data_objects::NodeAttributes::PointwiseAttributes;
    hipdnn_sdk::data_objects::PointwiseMode operation;
    hipdnn_sdk::data_objects::DataType inputDataType;
    hipdnn_sdk::data_objects::DataType outputDataType;
    hipdnn_sdk::data_objects::DataType input1DataType
        = hipdnn_sdk::data_objects::DataType::UNSET; // For binary ops

    PointwiseSignatureKey() = default;
    constexpr PointwiseSignatureKey(hipdnn_sdk::data_objects::PointwiseMode op,
                                    hipdnn_sdk::data_objects::DataType input,
                                    hipdnn_sdk::data_objects::DataType output,
                                    hipdnn_sdk::data_objects::DataType input1
                                    = hipdnn_sdk::data_objects::DataType::UNSET)
        : operation(op)
        , inputDataType(input)
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
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(input1DataType)) << 16);
    }

    bool operator==(const PointwiseSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && operation == other.operation
               && inputDataType == other.inputDataType && outputDataType == other.outputDataType
               && input1DataType == other.input1DataType;
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
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::HALF>(map);
        addUnaryPlanBuilders<hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        // Add plan builders for implemented binary operations
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::FLOAT>(map);
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::HALF>(map);
        addBinaryPlanBuilders<hipdnn_sdk::data_objects::DataType::BFLOAT16>(map);

        return map;
    }

private:
    template <hipdnn_sdk::data_objects::DataType DataTypeEnum>
    static void addUnaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        // Add all implemented unary operations
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::RELU_FWD, DataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_FWD, DataTypeEnum>(
            map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::TANH_FWD, DataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::ABS, DataTypeEnum>(map);
        addUnaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::NEG, DataTypeEnum>(map);
    }

    template <hipdnn_sdk::data_objects::DataType DataTypeEnum>
    static void addBinaryPlanBuilders(std::unordered_map<PointwiseSignatureKey,
                                                         std::unique_ptr<IGraphNodePlanBuilder>,
                                                         PointwiseSignatureKey>& map)
    {
        // Add all implemented binary operations
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::ADD, DataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SUB, DataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::MUL, DataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::RELU_BWD, DataTypeEnum>(map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::SIGMOID_BWD, DataTypeEnum>(
            map);
        addBinaryPlanBuilder<hipdnn_sdk::data_objects::PointwiseMode::TANH_BWD, DataTypeEnum>(map);
    }

    template <hipdnn_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_sdk::data_objects::DataType DataTypeEnum>
    static void addUnaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                       std::unique_ptr<IGraphNodePlanBuilder>,
                                                       PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(ModeEnum, DataTypeEnum, DataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<DataTypeEnum>>();
    }

    template <hipdnn_sdk::data_objects::PointwiseMode ModeEnum,
              hipdnn_sdk::data_objects::DataType DataTypeEnum>
    static void addBinaryPlanBuilder(std::unordered_map<PointwiseSignatureKey,
                                                        std::unique_ptr<IGraphNodePlanBuilder>,
                                                        PointwiseSignatureKey>& map)
    {
        map[PointwiseSignatureKey(ModeEnum, DataTypeEnum, DataTypeEnum, DataTypeEnum)]
            = std::make_unique<PointwisePlanBuilder<DataTypeEnum>>();
    }
};

}
