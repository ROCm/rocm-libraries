// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/layernorm_fprop_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& layernormJson, const LayernormFpropAttributes& ln)
{
    auto& inputs = layernormJson["inputs"] = {};
    auto& outputs = layernormJson["outputs"] = {};

    inputs["x_tensor_uid"] = ln.x_tensor_uid();
    inputs["scale_tensor_uid"] = ln.scale_tensor_uid();
    inputs["bias_tensor_uid"] = ln.bias_tensor_uid();
    inputs["epsilon_tensor_uid"] = ln.epsilon_tensor_uid();

    outputs["y_tensor_uid"] = ln.y_tensor_uid();
    outputs["mean_tensor_uid"] = ln.mean_tensor_uid();
    outputs["rstd_tensor_uid"] = ln.rstd_tensor_uid();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::LayernormFpropAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                       const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreateLayernormFpropAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("bias_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("epsilon_tensor_uid").get<int64_t>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        outputs.at("mean_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("rstd_tensor_uid").get<std::optional<int64_t>>());
}

}
