// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/topk_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const TopkAttributes& attr)
{
    j["inputs"]["x_tensor_uid"] = attr.x_tensor_uid();

    auto& outputs = j["outputs"] = {};
    outputs["indices_tensor_uid"] = attr.indices_tensor_uid();
    if(attr.values_tensor_uid().has_value())
    {
        outputs["values_tensor_uid"] = attr.values_tensor_uid().value();
    }

    j["k"] = attr.k();
    j["largest"] = attr.largest();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::TopkAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                             const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");
    auto valuesUid
        = outputs.contains("values_tensor_uid")
              ? ::flatbuffers::Optional<int64_t>(outputs.at("values_tensor_uid").get<int64_t>())
              : ::flatbuffers::nullopt;

    return data_objects::CreateTopkAttributes(builder,
                                              inputs.at("x_tensor_uid").get<int64_t>(),
                                              outputs.at("indices_tensor_uid").get<int64_t>(),
                                              valuesUid,
                                              entry.at("k").get<int64_t>(),
                                              entry.value("largest", true));
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
