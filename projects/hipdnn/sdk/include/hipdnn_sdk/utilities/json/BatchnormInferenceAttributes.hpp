// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_sdk/utilities/json/Common.hpp>

namespace hipdnn_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& batchnormJson, const BatchnormInferenceAttributes& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["x_tensor_uid"] = bn.x_tensor_uid();
    inputs["mean_tensor_uid"] = bn.mean_tensor_uid();
    inputs["scale_tensor_uid"] = bn.scale_tensor_uid();
    inputs["inv_variance_tensor_uid"] = bn.inv_variance_tensor_uid();
    inputs["bias_tensor_uid"] = bn.bias_tensor_uid();

    batchnormJson["outputs"]["y_tensor_uid"] = bn.y_tensor_uid();
}

}
namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::BatchnormInferenceAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                           const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];
    return data_objects::CreateBatchnormInferenceAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("mean_tensor_uid").get<int64_t>(),
        inputs.at("inv_variance_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("bias_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("y_tensor_uid").get<int64_t>());
}

}
