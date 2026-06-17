// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <string_view>

namespace hipdnn_flatbuffers_sdk
{
namespace rmsnorm_bwd_json_keys
{
constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view OUTPUTS = "outputs";
constexpr std::string_view DY_TENSOR_UID = "dy_tensor_uid";
constexpr std::string_view X_TENSOR_UID = "x_tensor_uid";
constexpr std::string_view SCALE_TENSOR_UID = "scale_tensor_uid";
constexpr std::string_view INV_RMS_TENSOR_UID = "inv_rms_tensor_uid";
constexpr std::string_view DX_TENSOR_UID = "dx_tensor_uid";
constexpr std::string_view DSCALE_TENSOR_UID = "dscale_tensor_uid";
constexpr std::string_view DBIAS_TENSOR_UID = "dbias_tensor_uid";
} // namespace rmsnorm_bwd_json_keys
} // namespace hipdnn_flatbuffers_sdk

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& rmsnormBwdJson, const RMSNormBackwardAttributes& rms)
{
    namespace keys = hipdnn_flatbuffers_sdk::rmsnorm_bwd_json_keys;

    auto& inputs = rmsnormBwdJson[keys::INPUTS] = {};

    inputs[keys::DY_TENSOR_UID] = rms.dy_tensor_uid();
    inputs[keys::X_TENSOR_UID] = rms.x_tensor_uid();
    inputs[keys::SCALE_TENSOR_UID] = rms.scale_tensor_uid();
    inputs[keys::INV_RMS_TENSOR_UID] = rms.inv_rms_tensor_uid();

    auto& outputs = rmsnormBwdJson[keys::OUTPUTS] = {};
    outputs[keys::DX_TENSOR_UID] = rms.dx_tensor_uid();
    outputs[keys::DSCALE_TENSOR_UID] = rms.dscale_tensor_uid();
    if(rms.dbias_tensor_uid().has_value())
    {
        outputs[keys::DBIAS_TENSOR_UID] = rms.dbias_tensor_uid().value();
    }
}

} // namespace hipdnn_flatbuffers_sdk::data_objects
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::RMSNormBackwardAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                        const nlohmann::json& entry)
{
    namespace keys = hipdnn_flatbuffers_sdk::rmsnorm_bwd_json_keys;

    auto& inputs = entry.at(keys::INPUTS);
    auto& outputs = entry.at(keys::OUTPUTS);

    flatbuffers::Optional<int64_t> dbiasUid = flatbuffers::nullopt;
    if(outputs.contains(keys::DBIAS_TENSOR_UID))
    {
        dbiasUid = outputs[keys::DBIAS_TENSOR_UID].get<int64_t>();
    }

    return data_objects::CreateRMSNormBackwardAttributes(
        builder,
        inputs.at(keys::DY_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::X_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::SCALE_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::INV_RMS_TENSOR_UID).get<int64_t>(),
        outputs.at(keys::DX_TENSOR_UID).get<int64_t>(),
        outputs.at(keys::DSCALE_TENSOR_UID).get<int64_t>(),
        dbiasUid);
}

} // namespace hipdnn_flatbuffers_sdk::json
