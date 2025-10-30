// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_sdk/utilities/json/Common.hpp>

namespace hipdnn_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& batchnormJson, const BatchnormBackwardAttributes& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["dy_tensor_uid"] = bn.dy_tensor_uid();
    inputs["x_tensor_uid"] = bn.x_tensor_uid();
    inputs["mean_tensor_uid"] = bn.mean_tensor_uid();
    inputs["inv_variance_tensor_uid"] = bn.inv_variance_tensor_uid();
    inputs["scale_tensor_uid"] = bn.scale_tensor_uid();
    inputs["peer_stats_tensor_uid"] = bn.peer_stats_tensor_uid();

    auto& outputs = batchnormJson["outputs"] = {};
    outputs["dbias_tensor_uid"] = bn.dbias_tensor_uid();
    outputs["dscale_tensor_uid"] = bn.dscale_tensor_uid();
    outputs["dx_tensor_uid"] = bn.dx_tensor_uid();
}

}
namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::BatchnormBackwardAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                          const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    auto peerStats = inputs.at("peer_stats_tensor_uid").get<std::vector<int64_t>>();

    return data_objects::CreateBatchnormBackwardAttributesDirect(
        builder,
        inputs.at("dy_tensor_uid").get<int64_t>(),
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("mean_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("inv_variance_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        &peerStats,
        outputs.at("dx_tensor_uid").get<int64_t>(),
        outputs.at("dscale_tensor_uid").get<int64_t>(),
        outputs.at("dbias_tensor_uid").get<int64_t>());
}
}
