// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/resample_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/resample_common_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const ResampleBwdAttributes& attr)
{
    j["dy_tensor_uid"] = attr.dy_tensor_uid();
    j["dx_tensor_uid"] = attr.dx_tensor_uid();
    if(attr.index_tensor_uid().has_value())
    {
        j["index_tensor_uid"] = attr.index_tensor_uid().value();
    }
    j["pre_padding"] = attr.pre_padding();
    j["post_padding"] = attr.post_padding();
    j["stride"] = attr.stride();
    j["window"] = attr.window();
    j["resample_mode"] = attr.resample_mode();
    j["padding_mode"] = attr.padding_mode();
    if(attr.generate_index().has_value())
    {
        j["generate_index"] = attr.generate_index().value();
    }
}

}

namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::ResampleBwdAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                    const nlohmann::json& entry)
{
    auto dyUid = entry.at("dy_tensor_uid").get<int64_t>();
    auto dxUid = entry.at("dx_tensor_uid").get<int64_t>();
    auto indexUid
        = entry.contains("index_tensor_uid")
              ? ::flatbuffers::Optional<int64_t>(entry.at("index_tensor_uid").get<int64_t>())
              : ::flatbuffers::nullopt;
    auto prePadding = builder.CreateVector(entry.at("pre_padding").get<std::vector<int64_t>>());
    auto postPadding = builder.CreateVector(entry.at("post_padding").get<std::vector<int64_t>>());
    auto stride = builder.CreateVector(entry.at("stride").get<std::vector<int64_t>>());
    auto window = builder.CreateVector(entry.at("window").get<std::vector<int64_t>>());
    auto resampleMode = entry.at("resample_mode").get<data_objects::ResampleMode>();
    auto paddingMode = entry.at("padding_mode").get<data_objects::PaddingMode>();
    auto generateIndex = entry.contains("generate_index")
                             ? ::flatbuffers::Optional<bool>(entry.at("generate_index").get<bool>())
                             : ::flatbuffers::nullopt;

    return data_objects::CreateResampleBwdAttributes(builder,
                                                     dyUid,
                                                     dxUid,
                                                     indexUid,
                                                     prePadding,
                                                     postPadding,
                                                     stride,
                                                     window,
                                                     resampleMode,
                                                     paddingMode,
                                                     generateIndex);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
