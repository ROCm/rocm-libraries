// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <stdexcept>

namespace hipdnn_flatbuffers_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(CausalConv1dActivation,
                             {{CausalConv1dActivation::IDENTITY, "identity"},
                              {CausalConv1dActivation::SILU, "silu"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const CausalConv1dAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["x_tensor_uid"] = attr.x_tensor_uid();
    inputs["weight_tensor_uid"] = attr.weight_tensor_uid();
    inputs["cu_seqlens_tensor_uid"] = attr.cu_seqlens_tensor_uid();
    inputs["bias_tensor_uid"] = attr.bias_tensor_uid();
    inputs["initial_state_tensor_uid"] = attr.initial_state_tensor_uid();
    inputs["state_indices_tensor_uid"] = attr.state_indices_tensor_uid();

    auto& outputs = j["outputs"] = {};
    outputs["y_tensor_uid"] = attr.y_tensor_uid();
    outputs["final_state_tensor_uid"] = attr.final_state_tensor_uid();

    j["attributes"]["activation"] = attr.activation();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

// Strict activation parser. NLOHMANN_JSON_SERIALIZE_ENUM maps unknown strings to the
// first entry (IDENTITY), which would silently drop a requested activation.
inline data_objects::CausalConv1dActivation parseCausalConv1dActivation(const nlohmann::json& entry)
{
    if(entry == "identity")
    {
        return data_objects::CausalConv1dActivation::IDENTITY;
    }
    if(entry == "silu")
    {
        return data_objects::CausalConv1dActivation::SILU;
    }
    throw std::runtime_error("hipdnn_flatbuffers_sdk::json::parseCausalConv1dActivation(): "
                             "activation must be \"identity\" or \"silu\", got "
                             + entry.dump());
}

template <>
inline auto to<data_objects::CausalConv1dAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                     const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");
    const auto& attributes = entry.at("attributes");

    return data_objects::CreateCausalConv1dAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("weight_tensor_uid").get<int64_t>(),
        inputs.at("cu_seqlens_tensor_uid").get<int64_t>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        inputs.at("bias_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("initial_state_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("state_indices_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("final_state_tensor_uid").get<std::optional<int64_t>>(),
        parseCausalConv1dActivation(attributes.at("activation")));
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
