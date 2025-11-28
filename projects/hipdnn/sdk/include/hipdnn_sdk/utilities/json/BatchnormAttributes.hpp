// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_sdk/utilities/json/Common.hpp>

namespace hipdnn_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& batchnormJson, const BatchnormAttributes& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};
    auto& outputs = batchnormJson["outputs"] = {};

    inputs["x_tensor_uid"] = bn.x_tensor_uid();
    inputs["scale_tensor_uid"] = bn.scale_tensor_uid();
    inputs["bias_tensor_uid"] = bn.bias_tensor_uid();
    inputs["epsilon_tensor_uid"] = bn.epsilon_tensor_uid();
    inputs["peer_stats_tensor_uid"] = bn.peer_stats_tensor_uid();
    inputs["prev_running_mean_tensor_uid"] = bn.prev_running_mean_tensor_uid();
    inputs["prev_running_variance_tensor_uid"] = bn.prev_running_variance_tensor_uid();
    inputs["momentum_tensor_uid"] = bn.momentum_tensor_uid();

    outputs["y_tensor_uid"] = bn.y_tensor_uid();
    outputs["mean_tensor_uid"] = bn.mean_tensor_uid();
    outputs["inv_variance_tensor_uid"] = bn.inv_variance_tensor_uid();
    outputs["next_running_mean_tensor_uid"] = bn.next_running_mean_tensor_uid();
    outputs["next_running_variance_tensor_uid"] = bn.next_running_variance_tensor_uid();
}

}
namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::BatchnormAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    auto peerStats = inputs.at("peer_stats_tensor_uid").get<std::vector<int64_t>>();

    return data_objects::CreateBatchnormAttributesDirect(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("bias_tensor_uid").get<int64_t>(),
        inputs.at("epsilon_tensor_uid").get<int64_t>(),
        &peerStats,
        inputs.at("prev_running_mean_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("prev_running_variance_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("momentum_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("y_tensor_uid").get<int64_t>(),
        outputs.at("mean_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("inv_variance_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("next_running_mean_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("next_running_variance_tensor_uid").get<std::optional<int64_t>>());
}

}
