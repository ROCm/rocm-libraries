// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <gtest/gtest.h>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>

#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace asm_sdpa_engine
{

// Build a forward SDPA graph that sets the deprecated causal booleans and the
// modern bounds trio explicitly, so contradictory combinations can be
// constructed. Returns the FlatBufferBuilder owning the graph buffer.
inline flatbuffers::FlatBufferBuilder createSdpaFwdGraphWithMask(
    bool causalMask,
    bool causalMaskBottomRight,
    flatbuffers::Optional<int64_t> leftBound,
    flatbuffers::Optional<int64_t> rightBound,
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment diagAlignment)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensorAttributes;

    const std::vector<int64_t> dims = {4, 8, 256, 128};
    const std::vector<int64_t> strides = hipdnn_data_sdk::utilities::generateStrides(dims);

    int64_t uid = 1;
    const auto qUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, qUid, "q", DataType::BFLOAT16, &strides, &dims));
    const auto kUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, kUid, "k", DataType::BFLOAT16, &strides, &dims));
    const auto vUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, vUid, "v", DataType::BFLOAT16, &strides, &dims));
    const auto oUid = uid++;
    tensorAttributes.push_back(
        CreateTensorAttributesDirect(builder, oUid, "o", DataType::BFLOAT16, &strides, &dims));

    const auto sdpaAttributes
        = CreateSdpaAttributes(builder,
                               qUid,
                               kUid,
                               vUid,
                               oUid,
                               flatbuffers::nullopt, // attn_mask_tensor_uid
                               flatbuffers::nullopt, // scale_tensor_uid
                               flatbuffers::nullopt, // seq_len_q_tensor_uid
                               flatbuffers::nullopt, // seq_len_kv_tensor_uid
                               flatbuffers::nullopt, // seed_tensor_uid
                               flatbuffers::nullopt, // offset_tensor_uid
                               flatbuffers::nullopt, // dropout_mask_tensor_uid
                               flatbuffers::nullopt, // dropout_scale_tensor_uid
                               flatbuffers::nullopt, // page_table_k_tensor_uid
                               flatbuffers::nullopt, // page_table_v_tensor_uid
                               flatbuffers::nullopt, // block_mask_tensor_uid
                               flatbuffers::nullopt, // sink_token_tensor_uid
                               flatbuffers::nullopt, // descale_q_tensor_uid
                               flatbuffers::nullopt, // descale_k_tensor_uid
                               flatbuffers::nullopt, // descale_v_tensor_uid
                               flatbuffers::nullopt, // descale_s_tensor_uid
                               flatbuffers::nullopt, // scale_s_tensor_uid
                               flatbuffers::nullopt, // scale_o_tensor_uid
                               flatbuffers::nullopt, // stats_tensor_uid
                               flatbuffers::nullopt, // max_tensor_uid
                               flatbuffers::nullopt, // sum_exp_tensor_uid
                               flatbuffers::nullopt, // rng_dump_tensor_uid
                               flatbuffers::nullopt, // amax_s_tensor_uid
                               flatbuffers::nullopt, // amax_o_tensor_uid
                               flatbuffers::nullopt, // generate_stats
                               false, // alibi_mask
                               false, // padding_mask
                               causalMask,
                               causalMaskBottomRight,
                               flatbuffers::nullopt, // dropout_probability
                               flatbuffers::nullopt, // attn_scale_value
                               leftBound,
                               rightBound,
                               flatbuffers::nullopt, // max_seq_len_kv
                               diagAlignment,
                               DataType::UNSET, // mma_core_mode
                               AttentionImplementation::AUTO);

    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttributes.Union()));

    const auto graphOffset = CreateGraphDirect(builder,
                                               "test",
                                               DataType::FLOAT,
                                               DataType::HALF,
                                               DataType::BFLOAT16,
                                               &tensorAttributes,
                                               &nodes);
    builder.Finish(graphOffset);
    return builder;
}

struct DeprecatedCausalMaskCase
{
    const char* name;
    bool causalMask;
    bool causalMaskBottomRight;
    flatbuffers::Optional<int64_t> leftBound;
    flatbuffers::Optional<int64_t> rightBound;
    hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment alignment;
    std::optional<plan_utils::MaskType> expected;
};

inline constexpr auto ALIGN_TOP_LEFT
    = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT;
inline constexpr auto ALIGN_BOTTOM_RIGHT
    = hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::BOTTOM_RIGHT;
inline constexpr flatbuffers::Optional<int64_t> NO_BOUND = flatbuffers::nullopt;

// BottomRightAloneDefaultAlignment: TOP_LEFT is the schema default, so it must not
// be read as contradicting causal_mask_bottom_right.
inline const std::vector<DeprecatedCausalMaskCase>& deprecatedCausalMaskCases()
{
    static const std::vector<DeprecatedCausalMaskCase> s_cases{
        DeprecatedCausalMaskCase{"CausalAlone",
                                 true,
                                 false,
                                 NO_BOUND,
                                 NO_BOUND,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::TOP_LEFT_CAUSAL},
        DeprecatedCausalMaskCase{"BottomRightAlone",
                                 false,
                                 true,
                                 NO_BOUND,
                                 NO_BOUND,
                                 ALIGN_BOTTOM_RIGHT,
                                 plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL},
        DeprecatedCausalMaskCase{"BottomRightAloneDefaultAlignment",
                                 false,
                                 true,
                                 NO_BOUND,
                                 NO_BOUND,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL},
        DeprecatedCausalMaskCase{
            "BothDeprecated", true, true, NO_BOUND, NO_BOUND, ALIGN_TOP_LEFT, std::nullopt},
        DeprecatedCausalMaskCase{"CausalWithBottomRightAlignment",
                                 true,
                                 false,
                                 NO_BOUND,
                                 NO_BOUND,
                                 ALIGN_BOTTOM_RIGHT,
                                 plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL},
        DeprecatedCausalMaskCase{"CausalWithConsistentBounds",
                                 true,
                                 false,
                                 -1,
                                 0,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::TOP_LEFT_CAUSAL},
        DeprecatedCausalMaskCase{"CausalWithLeftBoundOnly",
                                 true,
                                 false,
                                 64,
                                 NO_BOUND,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::SLIDING_WINDOW},
        DeprecatedCausalMaskCase{"CausalWithRightBoundOnly",
                                 true,
                                 false,
                                 NO_BOUND,
                                 0,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::TOP_LEFT_CAUSAL},
        DeprecatedCausalMaskCase{"CausalWithUnboundedRight",
                                 true,
                                 false,
                                 NO_BOUND,
                                 -1,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::TOP_LEFT_CAUSAL},
        DeprecatedCausalMaskCase{"CausalWithPositiveRightBound",
                                 true,
                                 false,
                                 NO_BOUND,
                                 16,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::SLIDING_WINDOW},
        DeprecatedCausalMaskCase{"BottomRightWithConsistentBounds",
                                 false,
                                 true,
                                 -1,
                                 0,
                                 ALIGN_BOTTOM_RIGHT,
                                 plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL},
        DeprecatedCausalMaskCase{"BottomRightWithRightBoundOnly",
                                 false,
                                 true,
                                 NO_BOUND,
                                 64,
                                 ALIGN_BOTTOM_RIGHT,
                                 plan_utils::MaskType::SLIDING_WINDOW},
        DeprecatedCausalMaskCase{"RightBoundBelowMinusOneWithCausal",
                                 true,
                                 false,
                                 NO_BOUND,
                                 -5,
                                 ALIGN_TOP_LEFT,
                                 std::nullopt},
        DeprecatedCausalMaskCase{
            "LeftBoundBelowMinusOne", false, false, -2, 0, ALIGN_TOP_LEFT, std::nullopt},
        DeprecatedCausalMaskCase{"BoundsOnlyTopLeftCausal",
                                 false,
                                 false,
                                 NO_BOUND,
                                 0,
                                 ALIGN_TOP_LEFT,
                                 plan_utils::MaskType::TOP_LEFT_CAUSAL},
        DeprecatedCausalMaskCase{"BoundsOnlyBottomRightCausal",
                                 false,
                                 false,
                                 NO_BOUND,
                                 0,
                                 ALIGN_BOTTOM_RIGHT,
                                 plan_utils::MaskType::BOTTOM_RIGHT_CAUSAL}};
    return s_cases;
}

inline std::string
    deprecatedCausalMaskCaseName(const ::testing::TestParamInfo<DeprecatedCausalMaskCase>& info)
{
    return info.param.name;
}

} // namespace asm_sdpa_engine
