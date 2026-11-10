// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const PagedCacheStoreAttributes& attr)
{
    auto& inputs = j["inputs"] = {};
    inputs["x_tensor_uid"] = attr.x_tensor_uid();
    inputs["slot_mapping_tensor_uid"] = attr.slot_mapping_tensor_uid();
    if(attr.scale_tensor_uid().has_value())
    {
        inputs["scale_tensor_uid"] = attr.scale_tensor_uid().value();
    }

    j["outputs"]["container_tensor_uid"] = attr.container_tensor_uid();
}

}
namespace hipdnn_flatbuffers_sdk::json
{

template <>
inline auto to<data_objects::PagedCacheStoreAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                        const nlohmann::json& entry)
{
    const auto& inputs = entry.at("inputs");
    const auto& outputs = entry.at("outputs");
    auto scaleUid
        = inputs.contains("scale_tensor_uid")
              ? ::flatbuffers::Optional<int64_t>(inputs.at("scale_tensor_uid").get<int64_t>())
              : ::flatbuffers::nullopt;

    return data_objects::CreatePagedCacheStoreAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("slot_mapping_tensor_uid").get<int64_t>(),
        outputs.at("container_tensor_uid").get<int64_t>(),
        scaleUid);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
