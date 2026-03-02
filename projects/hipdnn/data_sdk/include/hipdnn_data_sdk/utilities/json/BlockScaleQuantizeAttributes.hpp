// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& blockScaleJson, const BlockScaleQuantizeAttributes& bsq)
{
    auto& inputs = blockScaleJson["inputs"] = {};

    inputs["x_tensor_uid"] = bsq.x_tensor_uid();

    auto& outputs = blockScaleJson["outputs"] = {};
    outputs["y_tensor_uid"] = bsq.y_tensor_uid();
    outputs["scale_tensor_uid"] = bsq.scale_tensor_uid();

    if(bsq.block_size().has_value())
    {
        blockScaleJson["block_size"] = bsq.block_size().value();
    }

    if(bsq.axis().has_value())
    {
        blockScaleJson["axis"] = bsq.axis().value();
    }

    blockScaleJson["transpose"] = bsq.transpose();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::BlockScaleQuantizeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                           const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];

    flatbuffers::Optional<int32_t> blockSize = flatbuffers::nullopt;
    if(entry.contains("block_size"))
    {
        blockSize = entry["block_size"].get<int32_t>();
    }

    flatbuffers::Optional<int64_t> axis = flatbuffers::nullopt;
    if(entry.contains("axis"))
    {
        axis = entry["axis"].get<int64_t>();
    }

    bool transpose = false;
    if(entry.contains("transpose"))
    {
        transpose = entry["transpose"].get<bool>();
    }

    return data_objects::CreateBlockScaleQuantizeAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("y_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("scale_tensor_uid").get<int64_t>(),
        blockSize,
        axis,
        transpose);
}

}
