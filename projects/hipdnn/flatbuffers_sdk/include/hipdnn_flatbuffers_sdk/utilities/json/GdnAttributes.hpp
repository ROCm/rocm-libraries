// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/gdn_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const GdnAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["q_tensor_uid"] = attr.q_tensor_uid();
    inputs["k_tensor_uid"] = attr.k_tensor_uid();
    inputs["v_tensor_uid"] = attr.v_tensor_uid();
    inputs["g_tensor_uid"] = attr.g_tensor_uid();
    inputs["beta_tensor_uid"] = attr.beta_tensor_uid();
    inputs["cu_seqlens_tensor_uid"] = attr.cu_seqlens_tensor_uid();
    inputs["initial_state_tensor_uid"] = attr.initial_state_tensor_uid();
    inputs["a_log_tensor_uid"] = attr.a_log_tensor_uid();
    inputs["dt_bias_tensor_uid"] = attr.dt_bias_tensor_uid();
    inputs["state_indices_tensor_uid"] = attr.state_indices_tensor_uid();

    auto& outputs = j["outputs"] = {};
    outputs["o_tensor_uid"] = attr.o_tensor_uid();
    outputs["final_state_tensor_uid"] = attr.final_state_tensor_uid();
    outputs["state_checkpoints_tensor_uid"] = attr.state_checkpoints_tensor_uid();

    auto& attributes = j["attributes"] = {};
    attributes["scale"] = attr.scale();
    attributes["output_final_state"] = attr.output_final_state();
    attributes["use_qk_l2norm"] = attr.use_qk_l2norm();
    attributes["checkpoint_every_n_tokens"] = attr.checkpoint_every_n_tokens();
    attributes["use_beta_sigmoid"] = attr.use_beta_sigmoid();
    attributes["allow_neg_eigval"] = attr.allow_neg_eigval();
    attributes["safe_gate"] = attr.safe_gate();
    attributes["gate_domain"] = attr.gate_domain();
    attributes["batch_invariant"] = attr.batch_invariant();
    attributes["overwrite_initial_state"] = attr.overwrite_initial_state();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::GdnAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                            const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");
    const auto& attributes = entry.at("attributes");

    return data_objects::CreateGdnAttributes(
        builder,
        inputs.at("q_tensor_uid").get<int64_t>(),
        inputs.at("k_tensor_uid").get<int64_t>(),
        inputs.at("v_tensor_uid").get<int64_t>(),
        inputs.at("g_tensor_uid").get<int64_t>(),
        inputs.at("beta_tensor_uid").get<int64_t>(),
        inputs.at("cu_seqlens_tensor_uid").get<int64_t>(),
        inputs.at("initial_state_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("a_log_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("dt_bias_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("state_indices_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("o_tensor_uid").get<int64_t>(),
        outputs.at("final_state_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("state_checkpoints_tensor_uid").get<std::optional<int64_t>>(),
        attributes.at("scale").get<std::optional<float>>(),
        attributes.at("output_final_state").get<bool>(),
        attributes.at("use_qk_l2norm").get<bool>(),
        attributes.at("checkpoint_every_n_tokens").get<int64_t>(),
        attributes.at("use_beta_sigmoid").get<bool>(),
        attributes.at("allow_neg_eigval").get<bool>(),
        attributes.at("safe_gate").get<bool>(),
        parseGateDomain(attributes.at("gate_domain")),
        attributes.at("batch_invariant").get<bool>(),
        attributes.at("overwrite_initial_state").get<bool>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
