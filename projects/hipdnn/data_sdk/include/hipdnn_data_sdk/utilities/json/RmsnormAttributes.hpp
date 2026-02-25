// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& rmsnormJson, const RMSNormAttributes& rms)
{
    auto& inputs = rmsnormJson["inputs"] = {};

    inputs["x_tensor_uid"] = rms.x_tensor_uid();
    inputs["scale_tensor_uid"] = rms.scale_tensor_uid();
    inputs["epsilon_tensor_uid"] = rms.epsilon_tensor_uid();
    if(rms.bias_tensor_uid().has_value())
    {
        inputs["bias_tensor_uid"] = rms.bias_tensor_uid().value();
    }

    auto& outputs = rmsnormJson["outputs"] = {};
    outputs["y_tensor_uid"] = rms.y_tensor_uid();
    if(rms.inv_rms_tensor_uid().has_value())
    {
        outputs["inv_rms_tensor_uid"] = rms.inv_rms_tensor_uid().value();
    }

    if(rms.forward_phase() != data_objects::NormFwdPhase::NOT_SET)
    {
        rmsnormJson["forward_phase"] = EnumNameNormFwdPhase(rms.forward_phase());
    }
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::RMSNormAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];

    flatbuffers::Optional<int64_t> invRmsUid = flatbuffers::nullopt;
    if(entry.contains("outputs") && entry["outputs"].contains("inv_rms_tensor_uid"))
    {
        invRmsUid = entry["outputs"]["inv_rms_tensor_uid"].get<int64_t>();
    }

    flatbuffers::Optional<int64_t> biasUid = flatbuffers::nullopt;
    if(inputs.contains("bias_tensor_uid"))
    {
        biasUid = inputs["bias_tensor_uid"].get<int64_t>();
    }

    auto forwardPhase = data_objects::NormFwdPhase::NOT_SET;
    if(entry.contains("forward_phase"))
    {
        auto phaseStr = entry["forward_phase"].get<std::string>();
        if(phaseStr == "INFERENCE")
        {
            forwardPhase = data_objects::NormFwdPhase::INFERENCE;
        }
        else if(phaseStr == "TRAINING")
        {
            forwardPhase = data_objects::NormFwdPhase::TRAINING;
        }
    }

    return data_objects::CreateRMSNormAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        inputs.at("epsilon_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("y_tensor_uid").get<int64_t>(),
        biasUid,
        invRmsUid,
        forwardPhase);
}

}
