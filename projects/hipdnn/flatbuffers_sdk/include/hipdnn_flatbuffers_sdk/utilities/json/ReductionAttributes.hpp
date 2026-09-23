// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{

NLOHMANN_JSON_SERIALIZE_ENUM(ReductionMode,
                             {{ReductionMode::NOT_SET, "not_set"},
                              {ReductionMode::ADD, "add"},
                              {ReductionMode::MUL, "mul"},
                              {ReductionMode::MIN_OP, "min"},
                              {ReductionMode::MAX_OP, "max"},
                              {ReductionMode::AMAX, "amax"},
                              {ReductionMode::AVG, "avg"},
                              {ReductionMode::NORM1, "norm1"},
                              {ReductionMode::NORM2, "norm2"},
                              {ReductionMode::MUL_NO_ZEROS, "mul_no_zeros"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& j, const ReductionAttributes& attr)
{
    j["mode"] = attr.mode();
    j["in_tensor_uid"] = attr.in_tensor_uid();
    j["out_tensor_uid"] = attr.out_tensor_uid();
    j["is_deterministic"] = attr.is_deterministic();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::ReductionAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nlohmann::json& entry)
{
    auto mode = entry.at("mode").get<data_objects::ReductionMode>();
    auto inUid = entry.at("in_tensor_uid").get<int64_t>();
    auto outUid = entry.at("out_tensor_uid").get<int64_t>();
    auto isDeterministic = entry.value("is_deterministic", false);

    return data_objects::CreateReductionAttributes(builder, mode, inUid, outUid, isDeterministic);
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
