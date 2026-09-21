// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& batchnormJson,
                    const BatchnormInferenceAttributesVarianceExt& bn)
{
    auto& inputs = batchnormJson["inputs"] = {};

    inputs["x_tensor_uid"] = bn.x_tensor_uid();
    inputs["mean_tensor_uid"] = bn.mean_tensor_uid();
    inputs["scale_tensor_uid"] = bn.scale_tensor_uid();
    inputs["variance_tensor_uid"] = bn.variance_tensor_uid();
    inputs["bias_tensor_uid"] = bn.bias_tensor_uid();
    inputs["epsilon_tensor_uid"] = bn.epsilon_tensor_uid();

    batchnormJson["outputs"]["y_tensor_uid"] = bn.y_tensor_uid();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::BatchnormInferenceAttributesVarianceExt>(
    flatbuffers::FlatBufferBuilder& builder, const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];
    return data_objects::CreateBatchnormInferenceAttributesVarianceExt(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("mean_tensor_uid").get<int64_t>(),
        inputs.at("variance_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("bias_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("y_tensor_uid").get<int64_t>(),
        inputs.at("epsilon_tensor_uid").get<int64_t>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
