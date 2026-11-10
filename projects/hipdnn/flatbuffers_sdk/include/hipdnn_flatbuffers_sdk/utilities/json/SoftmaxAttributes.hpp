// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/softmax_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const SoftmaxAttributes& attr)
{
    j["inputs"]["p_tensor_uid"] = attr.p_tensor_uid();

    auto& outputs = j["outputs"] = {};
    outputs["s_tensor_uid"] = attr.s_tensor_uid();
    outputs["stats_tensor_uid"] = attr.stats_tensor_uid();

    j["stats_use_log2"] = attr.stats_use_log2();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::SoftmaxAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");

    return data_objects::CreateSoftmaxAttributes(
        builder,
        inputs.at("p_tensor_uid").get<int64_t>(),
        outputs.at("s_tensor_uid").get<int64_t>(),
        outputs.value("stats_tensor_uid", nlohmann::json{}).get<std::optional<int64_t>>(),
        entry.value("stats_use_log2", false));
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
