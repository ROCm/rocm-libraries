// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_sdk/utilities/json/Common.hpp>

namespace hipdnn_sdk::data_objects
{
NLOHMANN_JSON_SERIALIZE_ENUM(PointwiseMode,
                             {{PointwiseMode::UNSET, "unset"},
                              {PointwiseMode::ABS, "abs"},
                              {PointwiseMode::ADD, "add"},
                              {PointwiseMode::ADD_SQUARE, "add_square"},
                              {PointwiseMode::BINARY_SELECT, "binary_select"},
                              {PointwiseMode::CEIL, "ceil"},
                              {PointwiseMode::CMP_EQ, "cmp_eq"},
                              {PointwiseMode::CMP_GE, "cmp_ge"},
                              {PointwiseMode::CMP_GT, "cmp_gt"},
                              {PointwiseMode::CMP_LE, "cmp_le"},
                              {PointwiseMode::CMP_LT, "cmp_lt"},
                              {PointwiseMode::CMP_NEQ, "cmp_neq"},
                              {PointwiseMode::DIV, "div"},
                              {PointwiseMode::ELU_BWD, "elu_bwd"},
                              {PointwiseMode::ELU_FWD, "elu_fwd"},
                              {PointwiseMode::ERF, "erf"},
                              {PointwiseMode::EXP, "exp"},
                              {PointwiseMode::FLOOR, "floor"},
                              {PointwiseMode::GELU_APPROX_TANH_BWD, "gelu_approx_tanh_bwd"},
                              {PointwiseMode::GELU_APPROX_TANH_FWD, "gelu_approx_tanh_fwd"},
                              {PointwiseMode::GELU_BWD, "gelu_bwd"},
                              {PointwiseMode::GELU_FWD, "gelu_fwd"},
                              {PointwiseMode::GEN_INDEX, "gen_index"},
                              {PointwiseMode::IDENTITY, "identity"},
                              {PointwiseMode::LOG, "log"},
                              {PointwiseMode::LOGICAL_AND, "logical_and"},
                              {PointwiseMode::LOGICAL_NOT, "logical_not"},
                              {PointwiseMode::LOGICAL_OR, "logical_or"},
                              {PointwiseMode::MAX_OP, "max_op"}, // Max is reserved
                              {PointwiseMode::MIN_OP, "min_op"}, // Min is reserved
                              {PointwiseMode::MUL, "mul"},
                              {PointwiseMode::NEG, "neg"},
                              {PointwiseMode::RECIPROCAL, "reciprocal"},
                              {PointwiseMode::RELU_BWD, "relu_bwd"},
                              {PointwiseMode::RELU_FWD, "relu_fwd"},
                              {PointwiseMode::RSQRT, "rsqrt"},
                              {PointwiseMode::SIGMOID_BWD, "sigmoid_bwd"},
                              {PointwiseMode::SIGMOID_FWD, "sigmoid_fwd"},
                              {PointwiseMode::SIN, "sin"},
                              {PointwiseMode::SOFTPLUS_BWD, "softplus_bwd"},
                              {PointwiseMode::SOFTPLUS_FWD, "softplus_fwd"},
                              {PointwiseMode::SQRT, "sqrt"},
                              {PointwiseMode::SUB, "sub"},
                              {PointwiseMode::SWISH_BWD, "swish_bwd"},
                              {PointwiseMode::SWISH_FWD, "swish_fwd"},
                              {PointwiseMode::TAN, "tan"},
                              {PointwiseMode::TANH_BWD, "tanh_bwd"},
                              {PointwiseMode::TANH_FWD, "tanh_fwd"}})

// NOLINTNEXTLINE(readability-identifier-naming)
inline void to_json(nlohmann::json& pointwiseJson, const PointwiseAttributes& pw)
{
    auto& inputs = pointwiseJson["inputs"] = {};

    inputs["operation"] = pw.operation();
    inputs["relu_lower_clip"] = pw.relu_lower_clip();
    inputs["relu_upper_clip"] = pw.relu_upper_clip();
    inputs["relu_lower_clip_slope"] = pw.relu_lower_clip_slope();
    inputs["axis_tensor_uid"] = pw.axis_tensor_uid();
    inputs["in_0_tensor_uid"] = pw.in_0_tensor_uid();
    inputs["in_1_tensor_uid"] = pw.in_1_tensor_uid();
    inputs["in_2_tensor_uid"] = pw.in_2_tensor_uid();
    pointwiseJson["outputs"]["out_0_tensor_uid"] = pw.out_0_tensor_uid();
    inputs["swish_beta"] = pw.swish_beta();
    inputs["elu_alpha"] = pw.elu_alpha();
    inputs["softplus_beta"] = pw.softplus_beta();
}

}
namespace hipdnn_sdk::json
{
template <>
inline auto to<data_objects::PointwiseAttributes>(flatbuffers::FlatBufferBuilder& builder,
                                                  const nlohmann::json& entry)
{
    using namespace data_objects;
    auto& inputs = entry.at("inputs");
    auto& outputs = entry.at("outputs");

    return data_objects::CreatePointwiseAttributes(
        builder,
        inputs.at("operation").get<PointwiseMode>(),
        inputs.at("relu_lower_clip").get<std::optional<float>>(),
        inputs.at("relu_upper_clip").get<std::optional<float>>(),
        inputs.at("relu_lower_clip_slope").get<std::optional<float>>(),
        inputs.at("axis_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("in_0_tensor_uid").get<int64_t>(),
        inputs.at("in_1_tensor_uid").get<std::optional<int64_t>>(),
        inputs.at("in_2_tensor_uid").get<std::optional<int64_t>>(),
        outputs.at("out_0_tensor_uid").get<int64_t>(),
        inputs.at("swish_beta").get<std::optional<float>>(),
        inputs.at("elu_alpha").get<std::optional<float>>(),
        inputs.at("softplus_beta").get<std::optional<float>>());
}

}
