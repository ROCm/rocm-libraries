// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const MoeGroupedMatmulBwdAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["doutput_tensor_uid"] = attr.doutput_tensor_uid();
    inputs["token_tensor_uid"] = attr.token_tensor_uid();
    inputs["first_token_offset_tensor_uid"] = attr.first_token_offset_tensor_uid();

    j["outputs"]["dweight_tensor_uid"] = attr.dweight_tensor_uid();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::MoeGroupedMatmulBwdAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                            const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");

    return data_objects::CreateMoeGroupedMatmulBwdAttributes(
        builder,
        inputs.at("doutput_tensor_uid").get<int64_t>(),
        inputs.at("token_tensor_uid").get<int64_t>(),
        inputs.at("first_token_offset_tensor_uid").get<int64_t>(),
        outputs.at("dweight_tensor_uid").get<int64_t>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
