// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const CustomOpAttributes& attr)
{
    j["custom_op_id"] = attr.custom_op_id()->c_str();
    j["input_tensor_uids"] = attr.input_tensor_uids();
    j["output_tensor_uids"] = attr.output_tensor_uids();
    j["data"] = attr.data();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::CustomOpAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                 const nlohmann::json& entry)
{
    auto customOpId = entry.at("custom_op_id").get<std::string>();
    auto inputUids = entry.at("input_tensor_uids").get<std::vector<int64_t>>();
    auto outputUids = entry.at("output_tensor_uids").get<std::vector<int64_t>>();
    auto data = entry.at("data").get<std::vector<uint8_t>>();

    return data_objects::CreateCustomOpAttributesDirect(
        builder, customOpId.c_str(), &inputUids, &outputUids, &data);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
