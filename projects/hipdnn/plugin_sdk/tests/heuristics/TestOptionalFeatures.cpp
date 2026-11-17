// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/heuristics/OptionalFeatures.hpp>

namespace
{
using hipdnn_plugin_sdk::heuristics::findOptionalFeature;
using hipdnn_plugin_sdk::heuristics::OptionalFeatureKind;
using hipdnn_plugin_sdk::heuristics::unguardedOptionalReferences;

constexpr const char* SCALE = "$graph.nodes[0].attn_scale_value";
constexpr const char* MASK_DIM = "$graph.nodes[0].attn_mask.dims[2]";

std::vector<std::string> unguarded(const nlohmann::json& signature,
                                   const std::string& metric = "time")
{
    std::vector<std::string> references;
    for(const auto& found :
        unguardedOptionalReferences(signature.get<std::vector<nlohmann::json>>(), metric))
    {
        references.push_back(found.reference);
    }
    return references;
}

TEST(TestOptionalFeatures, NamesAreMatchedByHowTheBinderPublishesEachOperand)
{
    const auto* scale = findOptionalFeature(SCALE);
    ASSERT_NE(scale, nullptr);
    EXPECT_EQ(scale->kind, OptionalFeatureKind::ATTRIBUTE);
    EXPECT_NE(std::find(scale->nodeTypes.begin(), scale->nodeTypes.end(), "SdpaAttributes"),
              scale->nodeTypes.end());

    const auto* mask = findOptionalFeature("graph.nodes[12].attn_mask.strides[3]");
    ASSERT_NE(mask, nullptr);
    EXPECT_EQ(mask->kind, OptionalFeatureKind::TENSOR);
    // The role alone is no published feature, only its fields.
    EXPECT_EQ(findOptionalFeature("graph.nodes[0].attn_mask"), nullptr);

    const auto* dilation = findOptionalFeature("graph.nodes[0].dilation[1]");
    ASSERT_NE(dilation, nullptr);
    EXPECT_EQ(dilation->kind, OptionalFeatureKind::ELEMENTS);

    EXPECT_EQ(findOptionalFeature("graph.flops_by_type.SdpaAttributes")->kind,
              OptionalFeatureKind::WORK);
    EXPECT_EQ(findOptionalFeature("graph.nodes[3].flops")->kind, OptionalFeatureKind::WORK);

    // Required operands, and a GDN scalar `scale` beside SDPA's tensor role of that name.
    EXPECT_EQ(findOptionalFeature("graph.nodes[0].q.dims[0]"), nullptr);
    EXPECT_EQ(findOptionalFeature("graph.nodes[0].data_dependent"), nullptr);
    EXPECT_EQ(findOptionalFeature("device.cu_count"), nullptr);
    EXPECT_EQ(findOptionalFeature("graph.nodes[0].scale")->kind, OptionalFeatureKind::ATTRIBUTE);
    EXPECT_EQ(findOptionalFeature("graph.nodes[0].scale.dims[0]")->kind,
              OptionalFeatureKind::TENSOR);
    EXPECT_EQ(findOptionalFeature("graph.nodes[x].attn_scale_value"), nullptr);
}

TEST(TestOptionalFeatures, ABareOptionalReferenceIsUnguardedWhereverItIsRead)
{
    EXPECT_EQ(unguarded({SCALE, "$graph.nodes[0].q.dims[0]"}), std::vector<std::string>{SCALE});
    EXPECT_EQ(unguarded({{{"*", {SCALE, 2}}}}), std::vector<std::string>{SCALE});
    // The default of value_or_default is read whenever the first argument is unbound.
    EXPECT_EQ(unguarded({{{"value_or_default", {"$graph.nodes[0].left_bound", SCALE}}}}),
              std::vector<std::string>{SCALE});
}

TEST(TestOptionalFeatures, ValueOrDefaultAndPresenceGuardTheirReferences)
{
    EXPECT_TRUE(unguarded({{{"value_or_default", {SCALE, 1.0}}}}).empty());
    EXPECT_TRUE(unguarded({{{"present", {MASK_DIM}}}}).empty());
    EXPECT_TRUE(unguarded({{{"not_present", MASK_DIM}}}).empty());
    // A guarded expression, not only a bare reference.
    EXPECT_TRUE(unguarded({{{"value_or_default", {{{"*", {SCALE, 2}}}, 0}}}}).empty());
}

TEST(TestOptionalFeatures, AConditionGuardsOnlyTheArmsItProvesBound)
{
    EXPECT_TRUE(unguarded({{{"if", {{{"present", {MASK_DIM}}}, MASK_DIM, -1}}}}).empty());
    EXPECT_TRUE(unguarded({{{"?:", {{{"not_present", {MASK_DIM}}}, -1, MASK_DIM}}}}).empty());
    EXPECT_EQ(unguarded({{{"if", {{{"present", {MASK_DIM}}}, -1, MASK_DIM}}}}),
              std::vector<std::string>{MASK_DIM});
    EXPECT_TRUE(unguarded({{{"and", {{{"present", {SCALE}}}, {{">", {SCALE, 0}}}}}}}).empty());
    EXPECT_TRUE(unguarded({{{"or", {{{"not_present", {SCALE}}}, {{">", {SCALE, 0}}}}}}}).empty());
    EXPECT_EQ(unguarded({{{"or", {{{"present", {SCALE}}}, {{">", {SCALE, 0}}}}}}}),
              std::vector<std::string>{SCALE});
}

TEST(TestOptionalFeatures, OneProvenFieldOfAnOperandProvesItsOtherFields)
{
    const std::string rank = "$graph.nodes[0].attn_mask.rank";
    EXPECT_TRUE(unguarded({{{"if", {{{"present", {rank}}}, MASK_DIM, -1}}}}).empty());
    // Another node's operand of the same role is not proven.
    EXPECT_EQ(
        unguarded({{{"if", {{{"present", {rank}}}, "$graph.nodes[1].attn_mask.dims[2]", -1}}}}),
        std::vector<std::string>{"$graph.nodes[1].attn_mask.dims[2]"});
}

TEST(TestOptionalFeatures, WorkFeaturesMayBeBareOnlyInAModelOfAMetricDerivedFromWork)
{
    const nlohmann::json signature = {"$graph.flops", {{"log2", {"$graph.nodes[0].flops"}}}};
    EXPECT_TRUE(unguarded(signature, "tflops").empty());
    EXPECT_EQ(unguarded(signature, "time"),
              (std::vector<std::string>{"$graph.flops", "$graph.nodes[0].flops"}));
    EXPECT_EQ(unguarded({SCALE}, "tflops"), std::vector<std::string>{SCALE});
}

// A feature's name does not say its node's type, so the node is any type having every operand
// the signature reads at it. `scale` is required by the norms and optional in SDPA.
TEST(TestOptionalFeatures, AnOperandRequiredByEveryTypeTheNodeCanBeIsNotOptionalThere)
{
    const std::string normScale = "$graph.nodes[0].scale.dims[1]";
    EXPECT_TRUE(
        unguarded({"$graph.nodes[0].x.dims[0]", normScale, "$graph.nodes[0].type"}).empty());
    EXPECT_EQ(unguarded({"$graph.nodes[0].q.dims[0]", normScale}),
              std::vector<std::string>{normScale});
    // Alone, a node could be any type: the name is optional somewhere, so it is flagged.
    EXPECT_EQ(unguarded({normScale}), std::vector<std::string>{normScale});
    // Each node is judged by its own operands.
    EXPECT_EQ(unguarded({"$graph.nodes[0].x.dims[0]",
                         normScale,
                         "$graph.nodes[1].q.dims[0]",
                         "$graph.nodes[1].scale.rank"}),
              std::vector<std::string>{"$graph.nodes[1].scale.rank"});
}
} // namespace
