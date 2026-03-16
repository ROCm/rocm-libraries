// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_DATA_SDK_SKIP_JSON_LIB

#include <hipdnn_data_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(ReductionMode,
                             {{ReductionMode::NOT_SET, "NOT_SET"},
                              {ReductionMode::ADD, "ADD"},
                              {ReductionMode::MUL, "MUL"},
                              {ReductionMode::MIN_OP, "MIN"},
                              {ReductionMode::MAX_OP, "MAX"},
                              {ReductionMode::AMAX, "AMAX"},
                              {ReductionMode::AVG, "AVG"},
                              {ReductionMode::NORM1, "NORM1"},
                              {ReductionMode::NORM2, "NORM2"},
                              {ReductionMode::MUL_NO_ZEROS, "MUL_NO_ZEROS"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const ReductionAttributes& attr)
{
    j["mode"] = attr.mode();
    j["in_tensor_uid"] = attr.in_tensor_uid();
    j["out_tensor_uid"] = attr.out_tensor_uid();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::ReductionAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nlohmann::json& entry)
{
    auto mode = entry.at("mode").get<data_objects::ReductionMode>();
    auto inUid = entry.at("in_tensor_uid").get<int64_t>();
    auto outUid = entry.at("out_tensor_uid").get<int64_t>();

    return data_objects::CreateReductionAttributes(builder, mode, inUid, outUid);
}

}

#endif // HIPDNN_DATA_SDK_SKIP_JSON_LIB
