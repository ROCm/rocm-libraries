// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/IngestorPackTestSupport.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_moe_routing. One pack, so `operationMatcher` is empty:
/// `MoeRoutingAttributes` is its own union arm. Input A is topk_ids, input B
/// first_token_offset (an output, the only two-uid slot left), output token_index.
inline constexpr PackSymbols FLYDSL_MOE_ROUTING{"hipkernel:flydsl_moe_routing",
                                                "hipkernel.flydsl_moe_routing.graph_match",
                                                "",
                                                "hipkernel.flydsl_moe_routing.kernel_match",
                                                "hipkernel.flydsl_moe_routing.score",
                                                "hipkernel.flydsl_moe_routing.dispatch",
                                                "flydsl_moe_routing.topk_ids.uid",
                                                "flydsl_moe_routing.first_token_offset.uid",
                                                "flydsl_moe_routing.token_index.uid"};

/// What the graph match binds beside those three uids.
constexpr const char* FLYDSL_MOE_ROUTING_TOKEN_KS_TOKEN = "flydsl_moe_routing.token_ks.uid";
constexpr const char* FLYDSL_MOE_ROUTING_TOKENS_TOKEN = "flydsl_moe_routing.tokens";
constexpr const char* FLYDSL_MOE_ROUTING_TOP_K_TOKEN = "flydsl_moe_routing.top_k";
constexpr const char* FLYDSL_MOE_ROUTING_EXPERTS_TOKEN = "flydsl_moe_routing.num_experts";
constexpr const char* FLYDSL_MOE_ROUTING_OFFSET_IS_I64_TOKEN = "flydsl_moe_routing.offset_is_i64";

/// The fields the MoE routing KMD declares.
constexpr const char* FLYDSL_MOE_ROUTING_DTYPE_FIELD = "dtype";
constexpr const char* FLYDSL_MOE_ROUTING_E_MAX_FIELD = "e_max";
constexpr const char* FLYDSL_MOE_ROUTING_PREFIX_SYMBOL_FIELD = "prefix_symbol";
constexpr const char* FLYDSL_MOE_ROUTING_SCATTER_SYMBOL_FIELD = "scatter_symbol";

/// Tensor uids buildFlydslMoeRoutingGraph() uses.
constexpr int64_t FLYDSL_MOE_ROUTING_TOPK_IDS_UID = 1;
constexpr int64_t FLYDSL_MOE_ROUTING_OFFSETS_UID = 2;
constexpr int64_t FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID = 3;
constexpr int64_t FLYDSL_MOE_ROUTING_TOKEN_KS_UID = 4;

/// Everything buildFlydslMoeRoutingGraph() varies. The defaults are a Qwen3.8 MoE layer at
/// a short prefill: 16 tokens, top-10 of 512 experts, INT32 offsets.
struct MoeRoutingGraphSpec
{
    int64_t tokens = 16;
    int64_t topK = 10;
    int64_t experts = 512;
    hipdnn_flatbuffers_sdk::data_objects::DataType offsetDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;

    /// Per-tensor overrides by uid; unset means the op's dtype, shape and packed strides.
    std::map<int64_t, hipdnn_flatbuffers_sdk::data_objects::DataType> dataTypes;
    std::map<int64_t, std::vector<int64_t>> dims;
    std::map<int64_t, std::vector<int64_t>> strides;
    /// Declared base alignment by uid; unset means the schema's 16.
    std::map<int64_t, int64_t> alignment;
    std::set<int64_t> virtualTensors;
    /// The uid the node names for token_index, to alias it with another operand.
    std::optional<int64_t> tokenIndexUid;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

/// A single-node MoE routing graph built from @p spec.
inline flatbuffers::FlatBufferBuilder buildFlydslMoeRoutingGraph(const MoeRoutingGraphSpec& spec
                                                                 = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

    const int64_t pairs = spec.tokens * spec.topK;
    const std::map<int64_t, std::vector<int64_t>> opDims{
        {FLYDSL_MOE_ROUTING_TOPK_IDS_UID, {spec.tokens, spec.topK}},
        {FLYDSL_MOE_ROUTING_OFFSETS_UID, {spec.experts + 1, 1, 1}},
        {FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID, {1, pairs, 1}},
        {FLYDSL_MOE_ROUTING_TOKEN_KS_UID, {1, pairs, 1}}};
    const std::map<int64_t, const char*> names{
        {FLYDSL_MOE_ROUTING_TOPK_IDS_UID, "topk_ids"},
        {FLYDSL_MOE_ROUTING_OFFSETS_UID, "first_token_offset"},
        {FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID, "token_index"},
        {FLYDSL_MOE_ROUTING_TOKEN_KS_UID, "token_ks"}};

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    for(const auto& [uid, defaultDims] : opDims)
    {
        const auto dimsIt = spec.dims.find(uid);
        const auto dims = dimsIt == spec.dims.end() ? defaultDims : dimsIt->second;
        const auto stridesIt = spec.strides.find(uid);
        const auto strides
            = stridesIt == spec.strides.end() ? flydslPackedStrides(dims) : stridesIt->second;
        const auto typeIt = spec.dataTypes.find(uid);
        auto type = data_objects::DataType::INT32;
        if(typeIt != spec.dataTypes.end())
        {
            type = typeIt->second;
        }
        else if(uid == FLYDSL_MOE_ROUTING_OFFSETS_UID)
        {
            type = spec.offsetDataType;
        }
        const auto declared = spec.alignment.find(uid);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            uid,
            names.at(uid),
            type,
            &strides,
            &dims,
            spec.virtualTensors.count(uid) != 0,
            data_objects::TensorValue::NONE,
            0,
            false,
            flatbuffers::nullopt,
            declared == spec.alignment.end() ? 16 : declared->second));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreateMoeRoutingAttributes(
            builder,
            FLYDSL_MOE_ROUTING_TOPK_IDS_UID,
            FLYDSL_MOE_ROUTING_OFFSETS_UID,
            spec.tokenIndexUid.value_or(FLYDSL_MOE_ROUTING_TOKEN_INDEX_UID),
            FLYDSL_MOE_ROUTING_TOKEN_KS_UID,
            spec.experts);
        nodes.push_back(
            data_objects::CreateNodeDirect(builder,
                                           "moe_routing",
                                           data_objects::DataType::INT32,
                                           data_objects::NodeAttributes::MoeRoutingAttributes,
                                           attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_moe_routing_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);

    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
