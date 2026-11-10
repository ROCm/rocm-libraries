// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/moe_routing_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const MoeRoutingAttributes& attr)
{
    j["inputs"]["topk_ids_tensor_uid"] = attr.topk_ids_tensor_uid();

    auto& outputs = j["outputs"] = {};
    outputs["first_token_offset_tensor_uid"] = attr.first_token_offset_tensor_uid();
    outputs["token_index_tensor_uid"] = attr.token_index_tensor_uid();
    outputs["token_ks_tensor_uid"] = attr.token_ks_tensor_uid();

    j["num_experts"] = attr.num_experts();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::MoeRoutingAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                   const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");

    return data_objects::CreateMoeRoutingAttributes(
        builder,
        inputs.at("topk_ids_tensor_uid").get<int64_t>(),
        outputs.at("first_token_offset_tensor_uid").get<int64_t>(),
        outputs.at("token_index_tensor_uid").get<int64_t>(),
        outputs.at("token_ks_tensor_uid").get<int64_t>(),
        entry.at("num_experts").get<int64_t>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
