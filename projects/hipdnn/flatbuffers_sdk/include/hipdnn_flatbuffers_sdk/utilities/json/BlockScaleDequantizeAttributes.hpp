// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

#include <string_view>

namespace hipdnn_flatbuffers_sdk
{
namespace block_scale_dequantize_json_keys
{
constexpr std::string_view INPUTS = "inputs";
constexpr std::string_view OUTPUTS = "outputs";
constexpr std::string_view X_TENSOR_UID = "x_tensor_uid";
constexpr std::string_view SCALE_TENSOR_UID = "scale_tensor_uid";
constexpr std::string_view Y_TENSOR_UID = "y_tensor_uid";
constexpr std::string_view BLOCK_SIZE = "block_size";
constexpr std::string_view IS_NEGATIVE_SCALE = "is_negative_scale";
} // namespace block_scale_dequantize_json_keys
} // namespace hipdnn_flatbuffers_sdk

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& blockScaleJson, const BlockScaleDequantizeAttributes& bsd)
{
    namespace keys = hipdnn_flatbuffers_sdk::block_scale_dequantize_json_keys;

    auto& inputs = blockScaleJson[keys::INPUTS] = {};

    inputs[keys::X_TENSOR_UID] = bsd.x_tensor_uid();
    inputs[keys::SCALE_TENSOR_UID] = bsd.scale_tensor_uid();

    auto& outputs = blockScaleJson[keys::OUTPUTS] = {};
    outputs[keys::Y_TENSOR_UID] = bsd.y_tensor_uid();

    blockScaleJson[keys::BLOCK_SIZE] = bsd.block_size();
    blockScaleJson[keys::IS_NEGATIVE_SCALE] = bsd.is_negative_scale();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto
    to<data_objects::BlockScaleDequantizeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                     const nlohmann::json& entry)
{
    namespace keys = hipdnn_flatbuffers_sdk::block_scale_dequantize_json_keys;

    auto& inputs = entry[keys::INPUTS];

    std::vector<int32_t> blockSize;
    if(entry.contains(keys::BLOCK_SIZE))
    {
        for(const auto& val : entry[keys::BLOCK_SIZE])
        {
            blockSize.push_back(val.get<int32_t>());
        }
    }

    bool isNegativeScale = false;
    if(entry.contains(keys::IS_NEGATIVE_SCALE))
    {
        isNegativeScale = entry[keys::IS_NEGATIVE_SCALE].get<bool>();
    }

    auto blockSizeVector = builder.CreateVector(blockSize);

    return data_objects::CreateBlockScaleDequantizeAttributes(
        builder,
        inputs.at(keys::X_TENSOR_UID).get<int64_t>(),
        inputs.at(keys::SCALE_TENSOR_UID).get<int64_t>(),
        entry.at(keys::OUTPUTS).at(keys::Y_TENSOR_UID).get<int64_t>(),
        blockSizeVector,
        isNegativeScale);
}

}
