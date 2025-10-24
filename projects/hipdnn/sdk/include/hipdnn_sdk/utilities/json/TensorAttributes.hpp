// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/utilities/json/Common.hpp>

namespace hipdnn_sdk::data_objects
{

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& tensorAttrJson,
                    const data_objects::TensorAttributes& tensorAttr)
{
    tensorAttrJson["uid"] = tensorAttr.uid();
    tensorAttrJson["data_type"] = tensorAttr.data_type();
    tensorAttrJson["dims"] = *tensorAttr.dims();
    tensorAttrJson["strides"] = *tensorAttr.strides();
    tensorAttrJson["name"] = tensorAttr.name()->c_str();
    tensorAttrJson["virtual"] = tensorAttr.virtual_();
}

}

namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::TensorAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                               const nlohmann::json& entry)
{
    auto uid = entry.at("uid").get<int64_t>();
    auto name = entry.at("name").get<std::string>();
    auto dataType = entry.at("data_type").get<data_objects::DataType>();
    auto dims = entry.at("dims").get<std::vector<int64_t>>();
    auto strides = entry.at("strides").get<std::vector<int64_t>>();
    bool isVirtual = entry.at("virtual").get<bool>();

    return data_objects::CreateTensorAttributesDirect(
        builder, uid, name.c_str(), dataType, &strides, &dims, isVirtual);
}
}
