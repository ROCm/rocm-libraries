// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& layernormJson, const LayernormAttributes& ln)
{
    auto& inputs = layernormJson["inputs"] = {};
    auto& outputs = layernormJson["outputs"] = {};

    inputs["x_tensor_uid"] = ln.x_tensor_uid();
    inputs["scale_tensor_uid"] = ln.scale_tensor_uid();
    inputs["bias_tensor_uid"] = ln.bias_tensor_uid();
    inputs["epsilon_tensor_uid"] = ln.epsilon_tensor_uid();

    outputs["y_tensor_uid"] = ln.y_tensor_uid();
    outputs["mean_tensor_uid"] = ln.mean_tensor_uid();
    outputs["inv_variance_tensor_uid"] = ln.inv_variance_tensor_uid();

    layernormJson["normalized_dim_count"] = ln.normalized_dim_count();
    layernormJson["forward_phase"] = static_cast<int8_t>(ln.forward_phase());
}

} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::LayernormAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    const int64_t normalizedDimCount = entry.at("normalized_dim_count").get<int64_t>();

    auto forwardPhase = data_objects::NormFwdPhase::NOT_SET;
    if(entry.contains("forward_phase"))
    {
        forwardPhase
            = static_cast<data_objects::NormFwdPhase>(entry.at("forward_phase").get<int8_t>());
    }

    return data_objects::CreateLayernormAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("bias_tensor_uid").get<int64_t>(),
        inputs.at("epsilon_tensor_uid").get<int64_t>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        normalizedDimCount,
        outputs.at("mean_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("inv_variance_tensor_uid").get<std::optional<int64_t>>(),
        forwardPhase);
}

} // namespace hipdnn_flatbuffers_sdk::json
