// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include "GpuPointwisePlan.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuPointwiseSignatureKey
{
    const hipdnn_flatbuffers_sdk::data_objects::NodeAttributes nodeType{
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes};
    hipdnn_flatbuffers_sdk::data_objects::DataType input0DataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType input1DataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType outputDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType{
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET};

    GpuPointwiseSignatureKey() = default;
    constexpr GpuPointwiseSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType input0,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType input1,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : input0DataType(input0)
        , input1DataType(input1)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    constexpr GpuPointwiseSignatureKey(hipdnn_flatbuffers_sdk::data_objects::DataType input,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType output,
                                       hipdnn_flatbuffers_sdk::data_objects::DataType compute)
        : input0DataType(input)
        , outputDataType(output)
        , computeDataType(compute)
    {
    }

    GpuPointwiseSignatureKey(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap,
        const hipdnn_flatbuffers_sdk::data_objects::DataType computeType)
    {
        const auto* nodeAttributes = node.attributes_as_PointwiseAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes could not be cast to PointwiseAttributes");
        }

        auto in0TensorAttr = tensorMap.at(nodeAttributes->in_0_tensor_uid());
        auto in1TensorAttr = nodeAttributes->in_1_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->in_1_tensor_uid().value())
                                 : nullptr;
        auto outTensorAttr = tensorMap.at(nodeAttributes->out_0_tensor_uid());
        if(in0TensorAttr == nullptr || outTensorAttr == nullptr)
        {
            throw std::runtime_error(
                "One or more required tensor attributes could not be found in the map, "
                "failed to construct key");
        }

        input0DataType = in0TensorAttr->data_type();
        if(in1TensorAttr != nullptr)
        {
            input1DataType = in1TensorAttr->data_type();
        }
        computeDataType = computeType;
        outputDataType = outTensorAttr->data_type();
    }

    std::size_t operator()(const GpuPointwiseSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType))
               ^ (static_cast<std::size_t>(static_cast<int>(input0DataType)) << 4)
               ^ (static_cast<std::size_t>(static_cast<int>(input1DataType)) << 8)
               ^ (static_cast<std::size_t>(static_cast<int>(outputDataType)) << 12)
               ^ (static_cast<std::size_t>(static_cast<int>(computeDataType)) << 16);
    }

    bool operator==(const GpuPointwiseSignatureKey& other) const noexcept
    {
        return nodeType == other.nodeType && input0DataType == other.input0DataType
               && input1DataType == other.input1DataType && outputDataType == other.outputDataType
               && computeDataType == other.computeDataType;
    }

    static std::unordered_map<GpuPointwiseSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuPointwiseSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuPointwiseSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuPointwiseSignatureKey>
            map;

        // Binary plans: Input0, Input1, Output, Compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);

        // Unary plans: Input0, Output, Compute
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        addPlanBuilder<hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
                       hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT>(map);
        return map;
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType Input0DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType Input1DataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuPointwiseSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuPointwiseSignatureKey>& map)
    {
        map[GpuPointwiseSignatureKey(
            Input0DataTypeEnum, Input1DataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<GpuPointwiseBinaryPlanBuilder<Input0DataTypeEnum,
                                                             Input1DataTypeEnum,
                                                             OutputDataTypeEnum,
                                                             ComputeDataTypeEnum>>();
    }

    template <hipdnn_flatbuffers_sdk::data_objects::DataType InputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType OutputDataTypeEnum,
              hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
    static void addPlanBuilder(std::unordered_map<GpuPointwiseSignatureKey,
                                                  std::unique_ptr<IGpuGraphNodePlanBuilder>,
                                                  GpuPointwiseSignatureKey>& map)
    {
        map[GpuPointwiseSignatureKey(InputDataTypeEnum, OutputDataTypeEnum, ComputeDataTypeEnum)]
            = std::make_unique<GpuPointwiseUnaryPlanBuilder<InputDataTypeEnum,
                                                            OutputDataTypeEnum,
                                                            ComputeDataTypeEnum>>();
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuPointwiseSignatureKey& key)
{
    os << "GpuPointwise(in0=" << key.input0DataType << ", in1=" << key.input1DataType
       << ", out=" << key.outputDataType << ", compute=" << key.computeDataType << ")";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
