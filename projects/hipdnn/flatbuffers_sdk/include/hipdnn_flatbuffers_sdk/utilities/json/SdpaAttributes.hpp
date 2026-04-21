// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(DiagonalAlignment,
                             {{DiagonalAlignment::TOP_LEFT, "TOP_LEFT"},
                              {DiagonalAlignment::BOTTOM_RIGHT, "BOTTOM_RIGHT"}})

NLOHMANN_JSON_SERIALIZE_ENUM(AttentionImplementation,
                             {{AttentionImplementation::AUTO, "AUTO"},
                              {AttentionImplementation::COMPOSITE, "COMPOSITE"},
                              {AttentionImplementation::UNIFIED, "UNIFIED"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& sdpaJson, const SdpaAttributes& sdpa)
{
    auto& inputs = sdpaJson["inputs"] = {};
    auto& outputs = sdpaJson["outputs"] = {};
    auto& attributes = sdpaJson["attributes"] = {};

    // Required input tensor UIDs
    inputs["q_tensor_uid"] = sdpa.q_tensor_uid();
    inputs["k_tensor_uid"] = sdpa.k_tensor_uid();
    inputs["v_tensor_uid"] = sdpa.v_tensor_uid();

    // Optional input tensor UIDs
    inputs["attn_mask_tensor_uid"] = sdpa.attn_mask_tensor_uid();
    inputs["scale_tensor_uid"] = sdpa.scale_tensor_uid();
    inputs["seq_len_q_tensor_uid"] = sdpa.seq_len_q_tensor_uid();
    inputs["seq_len_kv_tensor_uid"] = sdpa.seq_len_kv_tensor_uid();
    inputs["seed_tensor_uid"] = sdpa.seed_tensor_uid();
    inputs["offset_tensor_uid"] = sdpa.offset_tensor_uid();
    inputs["dropout_mask_tensor_uid"] = sdpa.dropout_mask_tensor_uid();
    inputs["dropout_scale_tensor_uid"] = sdpa.dropout_scale_tensor_uid();
    inputs["page_table_k_tensor_uid"] = sdpa.page_table_k_tensor_uid();
    inputs["page_table_v_tensor_uid"] = sdpa.page_table_v_tensor_uid();
    inputs["block_mask_tensor_uid"] = sdpa.block_mask_tensor_uid();
    inputs["sink_token_tensor_uid"] = sdpa.sink_token_tensor_uid();

    // FP8-specific input tensor UIDs
    inputs["descale_q_tensor_uid"] = sdpa.descale_q_tensor_uid();
    inputs["descale_k_tensor_uid"] = sdpa.descale_k_tensor_uid();
    inputs["descale_v_tensor_uid"] = sdpa.descale_v_tensor_uid();
    inputs["descale_s_tensor_uid"] = sdpa.descale_s_tensor_uid();
    inputs["scale_s_tensor_uid"] = sdpa.scale_s_tensor_uid();
    inputs["scale_o_tensor_uid"] = sdpa.scale_o_tensor_uid();

    // Required output tensor UID
    outputs["o_tensor_uid"] = sdpa.o_tensor_uid();

    // Optional output tensor UIDs
    outputs["stats_tensor_uid"] = sdpa.stats_tensor_uid();
    outputs["max_tensor_uid"] = sdpa.max_tensor_uid();
    outputs["sum_exp_tensor_uid"] = sdpa.sum_exp_tensor_uid();
    outputs["rng_dump_tensor_uid"] = sdpa.rng_dump_tensor_uid();

    // FP8-specific output tensor UIDs
    outputs["amax_s_tensor_uid"] = sdpa.amax_s_tensor_uid();
    outputs["amax_o_tensor_uid"] = sdpa.amax_o_tensor_uid();

    // Boolean flags
    attributes["generate_stats"] = sdpa.generate_stats();
    attributes["alibi_mask"] = sdpa.alibi_mask();
    attributes["padding_mask"] = sdpa.padding_mask();
    attributes["causal_mask"] = sdpa.causal_mask();
    attributes["causal_mask_bottom_right"] = sdpa.causal_mask_bottom_right();

    // Scalar attributes
    attributes["dropout_probability"] = sdpa.dropout_probability();
    attributes["attn_scale_value"] = sdpa.attn_scale_value();
    attributes["left_bound"] = sdpa.left_bound();
    attributes["right_bound"] = sdpa.right_bound();
    attributes["max_seq_len_kv"] = sdpa.max_seq_len_kv();

    // Enum attributes
    attributes["diagonal_alignment"] = sdpa.diagonal_alignment();
    attributes["mma_core_mode"] = sdpa.mma_core_mode();
    attributes["implementation"] = sdpa.implementation();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::SdpaAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                             const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");
    auto& attributes = entry.at("attributes");

    return data_objects::CreateSdpaAttributes(
        builder,
        // Required input tensor UIDs
        inputs.at("q_tensor_uid").get<int64_t>(),
        inputs.at("k_tensor_uid").get<int64_t>(),
        inputs.at("v_tensor_uid").get<int64_t>(),
        // Required output tensor UID
        outputs.at("o_tensor_uid").get<int64_t>(),
        // Optional input tensor UIDs
        inputs.at("attn_mask_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("scale_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seq_len_q_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seq_len_kv_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("seed_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("offset_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dropout_mask_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dropout_scale_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("page_table_k_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("page_table_v_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("block_mask_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("sink_token_tensor_uid").get<std::optional<int64_t>>(),
        // FP8-specific input tensor UIDs
        inputs.at("descale_q_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("descale_k_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("descale_v_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("descale_s_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("scale_s_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("scale_o_tensor_uid").get<std::optional<int64_t>>(),
        // Optional output tensor UIDs
        outputs.at("stats_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("max_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("sum_exp_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("rng_dump_tensor_uid").get<std::optional<int64_t>>(),
        // FP8-specific output tensor UIDs
        outputs.at("amax_s_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("amax_o_tensor_uid").get<std::optional<int64_t>>(),
        // Boolean flags
        attributes.at("generate_stats").get<std::optional<bool>>(),
        attributes.at("alibi_mask").get<bool>(),
        attributes.at("padding_mask").get<bool>(),
        attributes.at("causal_mask").get<bool>(),
        attributes.at("causal_mask_bottom_right").get<bool>(),
        // Scalar attributes
        attributes.at("dropout_probability").get<std::optional<float>>(),
        attributes.at("attn_scale_value").get<std::optional<float>>(),
        attributes.at("left_bound").get<std::optional<int64_t>>(),
        attributes.at("right_bound").get<std::optional<int64_t>>(),
        attributes.at("max_seq_len_kv").get<std::optional<int32_t>>(),
        // Enum attributes
        attributes.at("diagonal_alignment").get<data_objects::DiagonalAlignment>(),
        attributes.at("mma_core_mode").get<data_objects::DataType>(),
        attributes.at("implementation").get<data_objects::AttentionImplementation>());
}

}
