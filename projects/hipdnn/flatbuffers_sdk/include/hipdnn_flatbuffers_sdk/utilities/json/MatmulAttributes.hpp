// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& matmulJson, const MatmulAttributes& mm)
{
    auto& inputs = matmulJson["inputs"] = {};
    inputs["a_tensor_uid"] = mm.a_tensor_uid();
    inputs["b_tensor_uid"] = mm.b_tensor_uid();
    matmulJson["outputs"]["c_tensor_uid"] = mm.c_tensor_uid();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::MatmulAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                               const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreateMatmulAttributes(builder,
                                                inputs.at("a_tensor_uid").get<int64_t>(),
                                                inputs.at("b_tensor_uid").get<int64_t>(),
                                                outputs.at("c_tensor_uid").get<int64_t>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
