// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& layernormJson, const LayernormBackwardAttributes& ln)
{
    auto& inputs = layernormJson["inputs"] = {};
    auto& outputs = layernormJson["outputs"] = {};

    inputs["dy_tensor_uid"] = ln.dy_tensor_uid();
    inputs["x_tensor_uid"] = ln.x_tensor_uid();
    inputs["scale_tensor_uid"] = ln.scale_tensor_uid();
    inputs["mean_tensor_uid"] = ln.mean_tensor_uid();
    inputs["inv_variance_tensor_uid"] = ln.inv_variance_tensor_uid();
    inputs["epsilon_uid"] = ln.epsilon_tensor_uid();

    outputs["dx_tensor_uid"] = ln.dx_tensor_uid();
    outputs["dscale_tensor_uid"] = ln.dscale_tensor_uid();
    outputs["dbias_tensor_uid"] = ln.dbias_tensor_uid();

    layernormJson["normalized_dim_count"] = ln.normalized_dim_count();
}

} // namespace hipdnn_flatbuffers_sdk::data_objects

namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::LayernormBackwardAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                          const nlohmann::json& entry)
{
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreateLayernormBackwardAttributes(
        builder,
        inputs.at("dy_tensor_uid").get<int64_t>(),
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("mean_tensor_uid").get<int64_t>(),
        inputs.at("inv_variance_tensor_uid").get<int64_t>(),
        inputs.at("epsilon_uid").get<int64_t>(),
        outputs.at("dx_tensor_uid").get<int64_t>(),
        outputs.at("dscale_tensor_uid").get<int64_t>(),
        outputs.at("dbias_tensor_uid").get<int64_t>(),
        entry.at("normalized_dim_count").get<int64_t>());
}

} // namespace hipdnn_flatbuffers_sdk::json
