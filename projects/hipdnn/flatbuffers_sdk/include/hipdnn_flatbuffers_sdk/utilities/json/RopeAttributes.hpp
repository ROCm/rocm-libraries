// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/rope_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const RopeAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["x_tensor_uid"] = attr.x_tensor_uid();
    inputs["freqs_tensor_uid"] = attr.freqs_tensor_uid();

    j["outputs"]["y_tensor_uid"] = attr.y_tensor_uid();

    j["output_scale"] = attr.output_scale();
    j["rope_dim"] = attr.rope_dim();
    j["rope_offset_ext"] = attr.rope_offset_ext();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::RopeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                             const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");

    return data_objects::CreateRopeAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("freqs_tensor_uid").get<int64_t>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        entry.value("output_scale", 1.0f),
        entry.value("rope_dim", int64_t{0}),
        entry.value("rope_offset_ext", nlohmann::json{}).get<std::optional<int64_t>>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
