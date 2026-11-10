// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/gather_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const GatherAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["x_tensor_uid"] = attr.x_tensor_uid();
    inputs["indices_tensor_uid"] = attr.indices_tensor_uid();

    j["outputs"]["y_tensor_uid"] = attr.y_tensor_uid();

    j["axis"] = attr.axis();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::GatherAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                               const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");

    return data_objects::CreateGatherAttributes(builder,
                                                inputs.at("x_tensor_uid").get<int64_t>(),
                                                inputs.at("indices_tensor_uid").get<int64_t>(),
                                                outputs.at("y_tensor_uid").get<int64_t>(),
                                                entry.value("axis", int64_t{0}));
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
