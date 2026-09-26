// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/Uuid.hpp>

#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// The three packs of the multi-pack engine. Everything but `operationMatcher` is
/// deliberately identical: sharing by id is what the three-pack topology exists to show.
inline constexpr PackSymbols POINTWISE_ADD{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.add_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

inline constexpr PackSymbols POINTWISE_MUL{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.mul_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

inline constexpr PackSymbols POINTWISE_SUB{"hipkernel:Pointwise",
                                           "hipkernel.pointwise.graph_match",
                                           "hipkernel.pointwise.sub_match",
                                           "hipkernel.pointwise.kernel_match",
                                           "hipkernel.pointwise.score",
                                           "hipkernel.pointwise.dispatch",
                                           "pointwise.input_a.uid",
                                           "pointwise.input_b.uid",
                                           "pointwise.output.uid"};

/// Tensor uids the builders below use, in argument order.
constexpr int64_t INPUT_A_UID = 1;
constexpr int64_t INPUT_B_UID = 2;
constexpr int64_t OUTPUT_UID = 3;
/// A real third operand, added when `includeThirdOperand` is set.
constexpr int64_t INPUT_C_UID = 4;
/// Uid named by `in_1_tensor_uid` unless `danglingInputBUid` overrides it; never inserted.
constexpr int64_t DEFAULT_DANGLING_UID = 999;

/**
 * @brief Builds a single-node binary-pointwise-add graph, parameterized on everything
 *        this pack's matchers gate.
 *
 * @param includeThirdOperand Adds a real third tensor (`INPUT_C_UID`), producing a
 *        ternary op.
 * @param danglingInputBUid When set, `in_1_tensor_uid` names this value instead of
 *        `INPUT_B_UID`, and no tensor is inserted for it.
 * @param omitStrides Builds every tensor with no strides vector at all: applicability
 *        runs before anything has validated a caller-supplied graph.
 */
inline flatbuffers::FlatBufferBuilder buildPointwiseGraph(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode operation
    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::ADD,
    hipdnn_flatbuffers_sdk::data_objects::DataType dataType
    = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
    const std::vector<int64_t>& dims = {1, 1, 1, 1},
    std::optional<hipdnn_flatbuffers_sdk::utilities::UuidBytes> graphId = std::nullopt,
    bool binary = true,
    const std::optional<std::vector<int64_t>>& explicitStrides = std::nullopt,
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> inputBDataType = std::nullopt,
    bool includeThirdOperand = false,
    std::optional<int64_t> danglingInputBUid = std::nullopt,
    bool inputAVirtual = false,
    bool inputAIsRuntimePassByValue = false,
    bool outputVirtual = false,
    bool omitStrides = false)
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    // An explicit set describes a view into a larger buffer.
    const std::vector<int64_t> strides
        = explicitStrides.has_value() ? *explicitStrides : std::vector<int64_t>(dims.size(), 1);
    // Null, not empty: the field is omitted entirely, so strides() returns nullptr.
    const std::vector<int64_t>* const stridesPtr = omitStrides ? nullptr : &strides;
    const auto resolvedInputBDataType = inputBDataType.value_or(dataType);

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                 INPUT_A_UID,
                                                                 nullptr,
                                                                 dataType,
                                                                 stridesPtr,
                                                                 &dims,
                                                                 inputAVirtual,
                                                                 data_objects::TensorValue::NONE,
                                                                 0,
                                                                 inputAIsRuntimePassByValue));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, INPUT_B_UID, nullptr, resolvedInputBDataType, stridesPtr, &dims, false));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, OUTPUT_UID, nullptr, dataType, stridesPtr, &dims, outputVirtual));
    if(includeThirdOperand)
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder, INPUT_C_UID, nullptr, dataType, stridesPtr, &dims, false));
    }

    data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
    attributesBuilder.add_operation(operation);
    attributesBuilder.add_in_0_tensor_uid(INPUT_A_UID);
    if(binary)
    {
        attributesBuilder.add_in_1_tensor_uid(danglingInputBUid.value_or(INPUT_B_UID));
    }
    if(includeThirdOperand)
    {
        attributesBuilder.add_in_2_tensor_uid(INPUT_C_UID);
    }
    attributesBuilder.add_out_0_tensor_uid(OUTPUT_UID);
    auto attributes = attributesBuilder.Finish();

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    nodes.push_back(
        data_objects::CreateNodeDirect(builder,
                                       "pointwise",
                                       dataType,
                                       data_objects::NodeAttributes::PointwiseAttributes,
                                       attributes.Union()));

    auto name = builder.CreateString("pointwise_add_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);

    // Held for the duration of the GraphBuilder: add_id stores a pointer to it.
    data_objects::Uuid uuid{};
    if(graphId.has_value())
    {
        uuid = hipdnn_flatbuffers_sdk::utilities::toFlatbufferUuid(*graphId);
        graphBuilder.add_id(&uuid);
    }
    builder.Finish(graphBuilder.Finish());

    return builder;
}

/// @brief A graph with two pointwise nodes, which no prebuilt single-op kernel serves.
inline flatbuffers::FlatBufferBuilder buildTwoNodePointwiseGraph()
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 1, 1, 1};
    const std::vector<int64_t> strides = {1, 1, 1, 1};
    constexpr int64_t INTERMEDIATE_UID = 4;

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto uid : {INPUT_A_UID, INPUT_B_UID, OUTPUT_UID, INTERMEDIATE_UID})
    {
        tensors.push_back(data_objects::CreateTensorAttributesDirect(builder,
                                                                     uid,
                                                                     nullptr,
                                                                     data_objects::DataType::FLOAT,
                                                                     &strides,
                                                                     &dims,
                                                                     uid == INTERMEDIATE_UID));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(const auto& [in0, in1, out] : {std::tuple{INPUT_A_UID, INPUT_B_UID, INTERMEDIATE_UID},
                                       std::tuple{INTERMEDIATE_UID, INPUT_B_UID, OUTPUT_UID}})
    {
        data_objects::PointwiseAttributesBuilder attributesBuilder(builder);
        attributesBuilder.add_operation(data_objects::PointwiseMode::ADD);
        attributesBuilder.add_in_0_tensor_uid(in0);
        attributesBuilder.add_in_1_tensor_uid(in1);
        attributesBuilder.add_out_0_tensor_uid(out);
        auto attributes = attributesBuilder.Finish();

        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "pointwise",
                                           data_objects::DataType::FLOAT,
                                           data_objects::NodeAttributes::PointwiseAttributes,
                                           attributes.Union()));
    }

    builder.Finish(data_objects::CreateGraphDirect(builder,
                                                   "two_node_pointwise",
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   data_objects::DataType::FLOAT,
                                                   &tensors,
                                                   &nodes));

    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
