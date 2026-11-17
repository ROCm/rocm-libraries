// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/heuristics/EngineFeatures.hpp>
#include <hipdnn_tools/OptionalFeatures.hpp>

namespace
{
using hipdnn_tools::findOptionalFeature;
using hipdnn_tools::OptionalFeatureKind;
using hipdnn_tools::unguardedOptionalReferences;

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

// Attention graphs as TestEngineFeatures builds them, for the cross-check below: Q [2, 8, 3,
// 16], K and V [2, 2, 7, *], O [2, 8, 3, 32], and four [2, 1, 1, 1] operands for uids 5-8.
using namespace hipdnn_flatbuffers_sdk::data_objects;

struct Device
{
    int multiProcessorCount = 64;
    int warpSize = 32;
    size_t totalGlobalMem = 1024;
    int memoryBusWidth = 256;
    int memoryClockRate = 1000;
    size_t sharedMemPerBlock = 65536;
};

void addTensor(GraphT& graph, int64_t uid, std::vector<int64_t> dims, std::vector<int64_t> strides)
{
    auto tensor = std::make_unique<TensorAttributesT>();
    tensor->uid = uid;
    tensor->dims = std::move(dims);
    tensor->strides = std::move(strides);
    tensor->data_type = DataType::HALF;
    graph.tensors.push_back(std::move(tensor));
}

SdpaAttributesT attention()
{
    SdpaAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    return sdpa;
}

GraphT attentionGraph(SdpaAttributesT sdpa)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 3, 16}, {384, 48, 16, 1});
    addTensor(graph, 2, {2, 2, 7, 16}, {224, 112, 16, 1});
    addTensor(graph, 3, {2, 2, 7, 32}, {448, 224, 32, 1});
    addTensor(graph, 4, {2, 8, 3, 32}, {768, 96, 32, 1});
    for(int64_t uid = 5; uid <= 8; ++uid)
    {
        addTensor(graph, uid, {2, 1, 1, 1}, {1, 1, 1, 1});
    }
    auto node = std::make_unique<NodeT>();
    node->compute_data_type = DataType::FLOAT;
    node->attributes.Set(sdpa);
    graph.nodes.push_back(std::move(node));
    return graph;
}

/// Every feature hipDNN publishes for @p graph at selection time.
nlohmann::json features(const GraphT& graph)
{
    flatbuffers::FlatBufferBuilder graphBuffer;
    graphBuffer.Finish(Graph::Pack(graphBuffer, &graph));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper wrapper(
        graphBuffer.GetBufferPointer(), graphBuffer.GetSize());
    flatbuffers::FlatBufferBuilder configBuffer;
    configBuffer.Finish(CreateEngineConfig(configBuffer, 1));
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper config(
        configBuffer.GetBufferPointer(), configBuffer.GetSize());
    return hipdnn_plugin_sdk::heuristics::engineFeatures(wrapper, config, Device{}).toJson();
}

// RFC 0019 section 6.2: a model must read every feature a valid graph may leave unbound
// through value_or_default or behind present, and this list is them. So each feature one valid
// form of attention publishes and another omits must be listed, whichever way round. Tensor
// ranks are equal across these forms: how many dims a tensor publishes is shape, which training
// sees as a column absent in some rows.
TEST(TestOptionalFeatures, EveryFeatureAValidFormOmitsIsAnOptionalFeature)
{
    auto dropout = attention();
    dropout.dropout_probability = 0.1F;
    dropout.seed_tensor_uid = 5;
    dropout.offset_tensor_uid = 6;
    auto stated = attention();
    stated.attn_scale_value = 0.125F;
    stated.left_bound = 4;
    stated.right_bound = 0;
    stated.generate_stats = true;
    stated.stats_tensor_uid = 5;
    auto masked = attention();
    masked.attn_mask_tensor_uid = 5;
    masked.sink_token_tensor_uid = 6;
    masked.scale_tensor_uid = 7;
    auto paged = attention();
    paged.seq_len_q_tensor_uid = 5;
    paged.seq_len_kv_tensor_uid = 6;
    paged.page_table_k_tensor_uid = 7;
    paged.page_table_v_tensor_uid = 8;
    paged.padding_mask = true;
    auto ragged = attentionGraph(attention());
    ragged.tensors[0]->ragged_offset_tensor_uid = 5;

    const auto minimal = features(attentionGraph(attention()));
    for(const auto& [name, form] :
        {std::pair<std::string, nlohmann::json>{"dropout", features(attentionGraph(dropout))},
         std::pair<std::string, nlohmann::json>{"stated", features(attentionGraph(stated))},
         std::pair<std::string, nlohmann::json>{"masked", features(attentionGraph(masked))},
         std::pair<std::string, nlohmann::json>{"paged variable-length",
                                                features(attentionGraph(paged))}})
    {
        SCOPED_TRACE(name);
        std::set<std::string> differing;
        for(const auto& [one, other] : {std::pair{&minimal, &form}, std::pair{&form, &minimal}})
        {
            for(const auto& item : one->items())
            {
                if(!other->contains(item.key()))
                {
                    differing.insert(item.key());
                }
            }
        }
        EXPECT_FALSE(differing.empty());
        for(const auto& feature : differing)
        {
            EXPECT_NE(findOptionalFeature(feature), nullptr)
                << feature << " is absent from a valid graph but not an optional feature";
        }
    }
    const auto raggedForm = features(ragged);
    for(const auto& item : minimal.items())
    {
        if(!raggedForm.contains(item.key()))
        {
            EXPECT_NE(findOptionalFeature(item.key()), nullptr)
                << item.key() << " is absent from a ragged graph but not an optional feature";
        }
    }
}
} // namespace
