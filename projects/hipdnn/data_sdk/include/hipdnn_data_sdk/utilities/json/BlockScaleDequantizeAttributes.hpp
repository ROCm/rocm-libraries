// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <hipdnn_data_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_data_sdk/utilities/json/Common.hpp>

namespace hipdnn_data_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& blockScaleJson, const BlockScaleDequantizeAttributes& bsd)
{
    auto& inputs = blockScaleJson["inputs"] = {};

    inputs["x_tensor_uid"] = bsd.x_tensor_uid();
    inputs["scale_tensor_uid"] = bsd.scale_tensor_uid();

    auto& outputs = blockScaleJson["outputs"] = {};
    outputs["y_tensor_uid"] = bsd.y_tensor_uid();

    if(bsd.block_size() != nullptr && bsd.block_size()->size() > 0)
    {
        auto& blockSizeArray = blockScaleJson["block_size"] = nlohmann::json::array();
        for(auto val : *bsd.block_size())
        {
            blockSizeArray.push_back(val);
        }
    }

    blockScaleJson["is_negative_scale"] = bsd.is_negative_scale();
}

}
namespace hipdnn_data_sdk::json
{
template <>
inline auto to<data_objects::BlockScaleDequantizeAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                             const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];

    std::vector<int32_t> blockSize;
    if(entry.contains("block_size"))
    {
        for(const auto& val : entry["block_size"])
        {
            blockSize.push_back(val.get<int32_t>());
        }
    }

    bool isNegativeScale = false;
    if(entry.contains("is_negative_scale"))
    {
        isNegativeScale = entry["is_negative_scale"].get<bool>();
    }

    auto blockSizeVector = builder.CreateVector(blockSize);

    return data_objects::CreateBlockScaleDequantizeAttributes(
        builder,
        inputs.at("x_tensor_uid").get<int64_t>(),
        inputs.at("scale_tensor_uid").get<int64_t>(),
        entry.at("outputs").at("y_tensor_uid").get<int64_t>(),
        blockSizeVector,
        isNegativeScale);
}

}
