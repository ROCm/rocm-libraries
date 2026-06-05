// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#ifndef HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB

#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/json/Common.hpp>

namespace hipdnn_flatbuffers_sdk::data_objects
{
// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& convJson, const ConvolutionBwdAttributes& conv)
{
    auto& inputs = convJson["inputs"] = {};

    inputs["dy_tensor_uid"] = conv.dy_tensor_uid();
    inputs["w_tensor_uid"] = conv.w_tensor_uid();

    convJson["outputs"]["dx_tensor_uid"] = conv.dx_tensor_uid();

    auto& params = convJson["parameters"] = {};
    params["pre_padding"] = conv.pre_padding();
    params["post_padding"] = conv.post_padding();
    params["stride"] = conv.stride();
    params["dilation"] = conv.dilation();
    params["conv_mode"] = conv.conv_mode();
}

}
namespace hipdnn_flatbuffers_sdk::json
{
template <>
inline auto to<data_objects::ConvolutionBwdAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                       const nlohmann::json& entry)
{
    auto& inputs = entry["inputs"];
    auto& outputs = entry["outputs"];
    auto& params = entry["parameters"];

    const std::vector<int64_t> prePadding = params.at("pre_padding").get<std::vector<int64_t>>();
    const std::vector<int64_t> postPadding = params.at("post_padding").get<std::vector<int64_t>>();
    const std::vector<int64_t> stride = params.at("stride").get<std::vector<int64_t>>();
    const std::vector<int64_t> dilation = params.at("dilation").get<std::vector<int64_t>>();

    return data_objects::CreateConvolutionBwdAttributesDirect(
        builder,
        inputs.at("dy_tensor_uid").get<int64_t>(),
        inputs.at("w_tensor_uid").get<int64_t>(),
        outputs.at("dx_tensor_uid").get<int64_t>(),
        &prePadding,
        &postPadding,
        &stride,
        &dilation,
        params.at("conv_mode").get<data_objects::ConvMode>());
}

}

#endif // HIPDNN_FLATBUFFERS_SDK_SKIP_JSON_LIB
