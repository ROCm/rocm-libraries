// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <string_view>

namespace hipdnn_flatbuffers_sdk
{
namespace rmsnorm_json_keys
{
constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view OUTPUTS = "outputs";
constexpr std::string_view X_TENSOR_UID = "x_tensor_uid";
constexpr std::string_view SCALE_TENSOR_UID = "scale_tensor_uid";
constexpr std::string_view EPSILON_TENSOR_UID = "epsilon_tensor_uid";
constexpr std::string_view Y_TENSOR_UID = "y_tensor_uid";
constexpr std::string_view BIAS_TENSOR_UID = "bias_tensor_uid";
constexpr std::string_view INV_RMS_TENSOR_UID = "inv_rms_tensor_uid";
constexpr std::string_view FORWARD_PHASE = "forward_phase";
} // namespace rmsnorm_json_keys
} // namespace hipdnn_flatbuffers_sdk

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& rmsnormJson, const RMSNormAttributes& rms)
{
    namespace keys = hipdnn_flatbuffers_sdk::rmsnorm_json_keys;

    auto& inputs = rmsnormJson[keys::INPUTS] = {};

    inputs[keys::X_TENSOR_UID] = rms.x_tensor_uid();
    inputs[keys::SCALE_TENSOR_UID] = rms.scale_tensor_uid();
    inputs[keys::EPSILON_TENSOR_UID] = rms.epsilon_tensor_uid();
    if(rms.bias_tensor_uid().has_value())
    {
        inputs[keys::BIAS_TENSOR_UID] = rms.bias_tensor_uid().value();
    }

    auto& outputs = rmsnormJson[keys::OUTPUTS] = {};
    outputs[keys::Y_TENSOR_UID] = rms.y_tensor_uid();
    if(rms.inv_rms_tensor_uid().has_value())
    {
        outputs[keys::INV_RMS_TENSOR_UID] = rms.inv_rms_tensor_uid().value();
    }

    if(rms.forward_phase() != data_objects::NormFwdPhase::NOT_SET)
    {
        rmsnormJson[keys::FORWARD_PHASE] = EnumNameNormFwdPhase(rms.forward_phase());
    }
}

} // namespace hipdnn_flatbuffers_sdk::data_objects
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::RMSNormAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                const nlohmann::json& entry)
{
    namespace keys = hipdnn_flatbuffers_sdk::rmsnorm_json_keys;

    auto& inputs = entry[keys::INPUTS];

    flatbuffers::Optional<int64_t> invRmsUid = flatbuffers::nullopt;
    if(entry.contains(keys::OUTPUTS) && entry[keys::OUTPUTS].contains(keys::INV_RMS_TENSOR_UID))
    {
        invRmsUid = entry[keys::OUTPUTS][keys::INV_RMS_TENSOR_UID].get<int64_t>();
    }

    flatbuffers::Optional<int64_t> biasUid = flatbuffers::nullopt;
    if(inputs.contains(keys::BIAS_TENSOR_UID))
    {
        biasUid = inputs[keys::BIAS_TENSOR_UID].get<int64_t>();
    }

    auto forwardPhase = data_objects::NormFwdPhase::NOT_SET;
    if(entry.contains(keys::FORWARD_PHASE))
    {
        auto phaseStr = entry[keys::FORWARD_PHASE].get<std::string>();
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
        inputs.at(keys::X_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::SCALE_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::EPSILON_TENSOR_UID).get<int64_t>(),
        entry.at(keys::OUTPUTS).at(keys::Y_TENSOR_UID).get<int64_t>(),
        biasUid,
        invRmsUid,
        forwardPhase);
}

} // namespace hipdnn_flatbuffers_sdk::json
