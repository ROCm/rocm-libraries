// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <string_view>

namespace hipdnn_flatbuffers_sdk
{
namespace block_scale_quantize_json_keys
{
constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view OUTPUTS = "outputs";
constexpr std::string_view X_TENSOR_UID = "x_tensor_uid";
constexpr std::string_view Y_TENSOR_UID = "y_tensor_uid";
constexpr std::string_view SCALE_TENSOR_UID = "scale_tensor_uid";
constexpr std::string_view BLOCK_SIZE = "block_size";
constexpr std::string_view AXIS = "axis";
constexpr std::string_view TRANSPOSE = "transpose";
} // namespace block_scale_quantize_json_keys
} // namespace hipdnn_flatbuffers_sdk

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& blockScaleJson, const BlockScaleQuantizeAttributes& bsq)
{
    namespace keys = hipdnn_flatbuffers_sdk::block_scale_quantize_json_keys;

    auto& inputs = blockScaleJson[keys::INPUTS] = {};

    inputs[keys::X_TENSOR_UID] = bsq.x_tensor_uid();

    auto& outputs = blockScaleJson[keys::OUTPUTS] = {};
    outputs[keys::Y_TENSOR_UID] = bsq.y_tensor_uid();
    outputs[keys::SCALE_TENSOR_UID] = bsq.scale_tensor_uid();

    if(bsq.block_size() != 0)
    {
        blockScaleJson[keys::BLOCK_SIZE] = bsq.block_size();
    }

    if(bsq.axis().has_value())
    {
        blockScaleJson[keys::AXIS] = bsq.axis().value();
    }

    blockScaleJson[keys::TRANSPOSE] = bsq.transpose();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::BlockScaleQuantizeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                           const nlohmann::json& entry)
{
    namespace keys = hipdnn_flatbuffers_sdk::block_scale_quantize_json_keys;

    auto& inputs = entry[keys::INPUTS];

    int32_t blockSize = 0;
    if(entry.contains(keys::BLOCK_SIZE))
    {
        blockSize = entry[keys::BLOCK_SIZE].get<int32_t>();
    }

    flatbuffers::Optional<int64_t> axis = flatbuffers::nullopt;
    if(entry.contains(keys::AXIS))
    {
        axis = entry[keys::AXIS].get<int64_t>();
    }

    bool transpose = false;
    if(entry.contains(keys::TRANSPOSE))
    {
        transpose = entry[keys::TRANSPOSE].get<bool>();
    }

    return data_objects::CreateBlockScaleQuantizeAttributes(
        builder,
        inputs.at(keys::X_TENSOR_UID).get<int64_t>(),
        entry.at(keys::OUTPUTS).at(keys::Y_TENSOR_UID).get<int64_t>(),
        entry.at(keys::OUTPUTS).at(keys::SCALE_TENSOR_UID).get<int64_t>(),
        blockSize,
        axis,
        transpose);
}

}
