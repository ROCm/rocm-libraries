// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(MoeGroupedMatmulMode,
                             {{MoeGroupedMatmulMode::NONE, "none"},
                              {MoeGroupedMatmulMode::GATHER, "gather"},
                              {MoeGroupedMatmulMode::SCATTER, "scatter"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const MoeGroupedMatmulAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["token_tensor_uid"] = attr.token_tensor_uid();
    inputs["weight_tensor_uid"] = attr.weight_tensor_uid();
    inputs["first_token_offset_tensor_uid"] = attr.first_token_offset_tensor_uid();
    if(attr.token_index_tensor_uid().has_value())
    {
        inputs["token_index_tensor_uid"] = attr.token_index_tensor_uid().value();
    }
    if(attr.token_ks_tensor_uid().has_value())
    {
        inputs["token_ks_tensor_uid"] = attr.token_ks_tensor_uid().value();
    }

    j["outputs"]["output_tensor_uid"] = attr.output_tensor_uid();
    j["mode"] = attr.mode();
    j["top_k"] = attr.top_k();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::MoeGroupedMatmulAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                         const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");
    auto tokenIndexUid
        = inputs.contains("token_index_tensor_uid")
              ? ::flatbuffers::Optional<int64_t>(inputs.at("token_index_tensor_uid").get<int64_t>())
              : ::flatbuffers::nullopt;
    auto tokenKsUid
        = inputs.contains("token_ks_tensor_uid")
              ? ::flatbuffers::Optional<int64_t>(inputs.at("token_ks_tensor_uid").get<int64_t>())
              : ::flatbuffers::nullopt;

    return data_objects::CreateMoeGroupedMatmulAttributes(
        builder,
        inputs.at("token_tensor_uid").get<int64_t>(),
        inputs.at("weight_tensor_uid").get<int64_t>(),
        inputs.at("first_token_offset_tensor_uid").get<int64_t>(),
        tokenIndexUid,
        tokenKsUid,
        outputs.at("output_tensor_uid").get<int64_t>(),
        entry.at("mode").get<data_objects::MoeGroupedMatmulMode>(),
        entry.at("top_k").get<int32_t>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
