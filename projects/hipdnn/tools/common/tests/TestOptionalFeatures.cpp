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
#include <hipdnn_plugin_sdk/heuristics/uhd/Expressions.hpp>
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

/// A signature of the one expression @p entry, which a braced list would not wrap in an array
/// when it is an object.
std::vector<std::string> unguardedEntry(const nlohmann::json& entry)
{
    return unguarded(nlohmann::json::array({entry}));
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

    // An element of a vector of tensor references (CustomOpAttributes.input) is optional too.
    const auto* input = findOptionalFeature("graph.nodes[0].input[2].dims[0]");
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(input->kind, OptionalFeatureKind::TENSORS);

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

TEST(TestOptionalFeatures, AnAndOrOrConditionProvesItsOperandsByPolarity)
{
    const nlohmann::json bothPresent
        = {{"and", {{{"present", {SCALE}}}, {{"present", {MASK_DIM}}}}}};
    const nlohmann::json eitherPresent
        = {{"or", {{{"present", {SCALE}}}, {{"present", {MASK_DIM}}}}}};
    const nlohmann::json eitherAbsent
        = {{"or", {{{"not_present", {SCALE}}}, {{"not_present", {MASK_DIM}}}}}};
    const nlohmann::json sum = {{"+", {SCALE, MASK_DIM}}};

    // `and` of presence proves every operand bound when true.
    EXPECT_TRUE(unguarded({{{"if", {bothPresent, sum, 0}}}}).empty());
    // ...but proves nothing when false: the else arm reads unbound values.
    EXPECT_EQ(unguarded({{{"if", {bothPresent, 0, SCALE}}}}), std::vector<std::string>{SCALE});

    // `or` of presence proves nothing when true (either may be the absent one).
    EXPECT_EQ(unguarded({{{"if", {eitherPresent, SCALE, 0}}}}), std::vector<std::string>{SCALE});
    // ...and proves nothing when false either: false means both absent, not both bound.
    EXPECT_EQ(unguarded({{{"if", {eitherPresent, 0, SCALE}}}}), std::vector<std::string>{SCALE});

    // `or` of absence is false only when every operand is bound, which the else arm then reads.
    EXPECT_TRUE(unguarded({{{"if", {eitherAbsent, 0, sum}}}}).empty());
    // ...and proves nothing when true.
    EXPECT_EQ(unguarded({{{"if", {eitherAbsent, SCALE, 0}}}}), std::vector<std::string>{SCALE});

    // A presence proof inside an `and` condition guards the rest of that condition and the arm.
    const nlohmann::json guardedCompare = {{"and", {{{"present", {SCALE}}}, {{">", {SCALE, 0}}}}}};
    EXPECT_TRUE(unguarded({{{"if", {guardedCompare, SCALE, 0}}}}).empty());
    EXPECT_EQ(unguarded({{{"if", {guardedCompare, 0, SCALE}}}}), std::vector<std::string>{SCALE});
}

// A presence proof guards what follows it, so a read before it, or beside it, is unguarded.
TEST(TestOptionalFeatures, AnAndOrOrConditionReadsAFeatureBeforeItsProofUnguarded)
{
    const nlohmann::json readThenProve = {{"and", {{{">", {SCALE, 0}}}, {{"present", {SCALE}}}}}};
    const nlohmann::json readThenProveAbsent
        = {{"or", {{{">", {SCALE, 0}}}, {{"not_present", {SCALE}}}}}};
    EXPECT_EQ(unguardedEntry(readThenProve), std::vector<std::string>{SCALE});
    EXPECT_EQ(unguardedEntry(readThenProveAbsent), std::vector<std::string>{SCALE});

    // The same inside an outer and/or: the inner condition's own proof comes after its read.
    EXPECT_EQ(unguarded({{{"and", {readThenProve, 1}}}}), std::vector<std::string>{SCALE});
    EXPECT_EQ(unguarded({{{"or", {readThenProveAbsent, 1}}}}), std::vector<std::string>{SCALE});

    // A proof in a later argument does not reach an earlier operand's other reads either.
    const nlohmann::json proveThenRead = {{"and", {{{"present", {SCALE}}}, {{">", {SCALE, 0}}}}}};
    EXPECT_TRUE(unguarded({{{"and", {proveThenRead, 1}}}}).empty());
}

TEST(TestOptionalFeatures, AProvenFieldProvesOnlyTheFieldsAnOperandAlwaysPublishes)
{
    const auto read = [](const std::string& field, const std::string& proof) {
        const auto operand = std::string("$graph.nodes[0].attn_mask.");
        return unguarded({{{"if", {{{"present", {operand + proof}}}, operand + field, -1}}}});
    };
    // A present operand has its data type, its virtual flag and its rank...
    for(const auto* field : {"data_type", "virtual", "rank"})
    {
        EXPECT_TRUE(read(field, "dims[2]").empty()) << field;
        EXPECT_TRUE(read(field, "strides[0]").empty()) << field;
    }
    // ...but only the dims and strides its rank gives it, and a numel only of sound dims.
    for(const auto* field : {"dims[2]", "strides[0]", "numel"})
    {
        for(const auto* proof : {"rank", "data_type", "virtual", "numel"})
        {
            if(std::string(field) != proof)
            {
                EXPECT_EQ(
                    read(field, proof),
                    std::vector<std::string>{std::string("$graph.nodes[0].attn_mask.") + field})
                    << field << " proven by " << proof;
            }
        }
        EXPECT_TRUE(read(field, field).empty()) << field;
    }
    // Another node's operand of the same role is not proven.
    const std::string rank = "$graph.nodes[0].attn_mask.rank";
    EXPECT_EQ(unguarded({{{"if", {{{"present", {rank}}}, "$graph.nodes[1].attn_mask.rank", -1}}}}),
              std::vector<std::string>{"$graph.nodes[1].attn_mask.rank"});
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

TEST(TestOptionalFeatures, AGuardedReadDoesNotNarrowTheTypesOfItsNode)
{
    // `stats` is optional in SDPA and required in its backward, which alone has `dbias`.
    const std::string stats = "$graph.nodes[0].stats.dims[0]";
    const std::string dbias = "$graph.nodes[0].dbias.dims[0]";
    EXPECT_EQ(unguarded({stats, {{"value_or_default", {dbias, -1}}}}),
              std::vector<std::string>{stats});
    EXPECT_EQ(unguarded({stats, {{"if", {{{"present", {dbias}}}, dbias, -1}}}}),
              std::vector<std::string>{stats});
    // A bare read of dbias does say it: the node is a backward, where stats is required.
    EXPECT_EQ(unguarded({stats, dbias}), std::vector<std::string>{dbias});
}

// ExpressionSet lowers `shape` and `rank` before it evaluates, so a signature that spells a read
// that way reads the same feature as one that does not.
TEST(TestOptionalFeatures, ShapeAndRankReadTheFeaturesTheyStandFor)
{
    const std::string mask = "$graph.nodes[0].attn_mask";
    const nlohmann::json shape = {{"shape", {mask, 1}}};
    const nlohmann::json rank = {{"rank", mask}};
    const nlohmann::json wrappedRank = {{"rank", {mask}}};
    EXPECT_EQ(unguardedEntry(shape), std::vector<std::string>{mask + ".dims[1]"});
    EXPECT_EQ(unguardedEntry(rank), std::vector<std::string>{mask + ".rank"});
    EXPECT_EQ(unguarded({{{"+", {shape, wrappedRank}}}}),
              (std::vector<std::string>{mask + ".dims[1]", mask + ".rank"}));

    // What the signature's expressions read, as the runtime compiles them.
    const hipdnn_plugin_sdk::uhd::ExpressionSet compiled({shape, rank, wrappedRank});
    EXPECT_EQ(compiled.variables(0), unguardedEntry(shape));
    EXPECT_EQ(compiled.variables(1), unguardedEntry(rank));
    EXPECT_EQ(compiled.variables(2), unguardedEntry(wrappedRank));

    // Guards and defaults name the lowered feature as well.
    EXPECT_TRUE(unguarded({{{"value_or_default", {shape, -1}}}}).empty());
    EXPECT_TRUE(unguarded({{{"if", {{{"present", {rank}}}, rank, -1}}}}).empty());
    EXPECT_TRUE(unguarded({{{"if", {{{"present", {shape}}}, shape, -1}}}}).empty());
    // A proven rank shows the tensor present, not that it has this dim.
    EXPECT_EQ(unguarded({{{"if", {{{"present", {rank}}}, shape, -1}}}}),
              std::vector<std::string>{mask + ".dims[1]"});
    // A required tensor is no optional feature however it is read.
    EXPECT_TRUE(unguarded({{{"shape", {"$graph.nodes[0].q", 1}}}}).empty());
}

TEST(TestOptionalFeatures, PresenceOfSeveralReferencesOrOfAnExpressionProvesOnlyWhatItSays)
{
    const nlohmann::json sum = {{"+", {SCALE, MASK_DIM}}};
    // `present` of references holds when every one is bound.
    EXPECT_TRUE(unguarded({{{"if", {{{"present", {SCALE, MASK_DIM}}}, sum, 0}}}}).empty());
    // `not_present` of several fails as soon as any one is bound, so its else arm proves none.
    EXPECT_EQ(unguarded({{{"if", {{{"not_present", {SCALE, MASK_DIM}}}, 0, sum}}}}),
              (std::vector<std::string>{MASK_DIM, SCALE}));
    EXPECT_EQ(unguarded({{{"or", {{{"not_present", {SCALE, MASK_DIM}}}, {{">", {SCALE, 0}}}}}}}),
              std::vector<std::string>{SCALE});
    // ...and of one it proves that one.
    EXPECT_TRUE(unguarded({{{"if", {{{"not_present", {SCALE}}}, 0, SCALE}}}}).empty());
    // An expression resolves without its references having to: `value_or_default` always does.
    const nlohmann::json defaulted = {{"value_or_default", {SCALE, 1.0}}};
    EXPECT_EQ(unguarded({{{"if", {{{"present", {defaulted}}}, SCALE, 0}}}}),
              std::vector<std::string>{SCALE});
    EXPECT_EQ(unguarded({{{"if", {{{"present", {sum}}}, sum, 0}}}}),
              (std::vector<std::string>{MASK_DIM, SCALE}));
}

TEST(TestOptionalFeatures, ANegatedOrComparedPresenceProvesWhatThePresenceDoes)
{
    const nlohmann::json present = {{"present", {SCALE}}};
    const nlohmann::json absent = {{"not_present", {SCALE}}};
    const std::vector<std::string> bare{SCALE};
    // The arm that runs when the guard holds, and the one that runs when it does not.
    const auto arms = [](const nlohmann::json& guard) {
        return std::make_pair(unguarded({{{"if", {guard, SCALE, 1.0}}}}),
                              unguarded({{{"if", {guard, 1.0, SCALE}}}}));
    };
    using Arms = std::pair<std::vector<std::string>, std::vector<std::string>>;
    const Arms boundThenUnbound{{}, bare};
    const Arms unboundThenBound{bare, {}};

    EXPECT_EQ(arms(present), boundThenUnbound);
    EXPECT_EQ(arms({{"!", {present}}}), unboundThenBound);
    EXPECT_EQ(arms({{"!", present}}), unboundThenBound);
    EXPECT_EQ(arms({{"!!", {present}}}), boundThenUnbound);
    EXPECT_EQ(arms({{"!", {{{"!", {present}}}}}}), boundThenUnbound);
    EXPECT_EQ(arms({{"!", {absent}}}), boundThenUnbound);
    EXPECT_EQ(arms({{"==", {present, true}}}), boundThenUnbound);
    EXPECT_EQ(arms({{"==", {true, present}}}), boundThenUnbound);
    EXPECT_EQ(arms({{"==", {present, false}}}), unboundThenBound);
    EXPECT_EQ(arms({{"!=", {present, true}}}), unboundThenBound);
    EXPECT_EQ(arms({{"!=", {false, present}}}), boundThenUnbound);
    // Whatever wraps an `and` of presence reads its proof the same way round.
    const nlohmann::json bothPresent
        = {{"and", {{{"present", {SCALE}}}, {{"present", {MASK_DIM}}}}}};
    EXPECT_EQ(unguarded({{{"if", {{{"!", {bothPresent}}}, 0, {{"+", {SCALE, MASK_DIM}}}}}}}),
              std::vector<std::string>{});
    EXPECT_EQ(unguarded({{{"if", {{{"!", {bothPresent}}}, {{"+", {SCALE, MASK_DIM}}}, 0}}}}),
              (std::vector<std::string>{MASK_DIM, SCALE}));
    // A comparison whose other side is no presence test proves nothing.
    EXPECT_EQ(unguarded({{{"if", {{{"==", {{{">", {1, 0}}}, true}}}, SCALE, 0}}}}), bare);
    EXPECT_EQ(unguarded({{{"if", {{{"==", {present, 1}}}, SCALE, 0}}}}), bare);
}

TEST(TestOptionalFeatures, EngineConfigConstraintsAreOptionalFeatures)
{
    const auto* limit = findOptionalFeature("$constraint.workspace_limit");
    ASSERT_NE(limit, nullptr);
    EXPECT_EQ(limit->kind, OptionalFeatureKind::CONSTRAINT);
    const auto* knob = findOptionalFeature("constraint.knobs.some_knob");
    ASSERT_NE(knob, nullptr);
    EXPECT_EQ(knob->kind, OptionalFeatureKind::CONSTRAINT);
    // Nothing else lives below `constraint`, and the knobs prefix alone is no feature.
    EXPECT_EQ(findOptionalFeature("constraint.knobs"), nullptr);
    EXPECT_EQ(findOptionalFeature("constraint.other"), nullptr);

    const std::string knobRead = "$constraint.knobs.some_knob";
    EXPECT_EQ(
        unguarded({"$constraint.workspace_limit", knobRead}, "tflops"),
        (std::vector<std::string>{"$constraint.knobs.some_knob", "$constraint.workspace_limit"}));
    EXPECT_TRUE(unguarded({{{"value_or_default", {"$constraint.workspace_limit", -1}}},
                           {{"if", {{{"present", {knobRead}}}, knobRead, -1}}}})
                    .empty());
}

// Attention graphs, convolutions and the features hipDNN publishes for them, for the
// cross-check below. The attention fixture is the one TestEngineFeatures builds, which must
// stay equal to it: Q [2, 8, 3, 16], K and V [2, 2, 7, *], O [2, 8, 3, 32], and four [2, 1, 1,
// 1] operands for uids 5-8 (plugin_sdk/tests/heuristics/TestEngineFeatures.cpp). There is no
// non-installed place both test binaries may include from without one component's tests
// reaching into another's, and the installed test SDK takes no test-only fixtures.
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

// The features hipDNN publishes for each of @p forms of a graph that @p minimal leaves its
// optional operands out of, as the schema allows: whichever way round, a feature one publishes
// and the other omits must be a listed optional feature. @p shapeDependent names the
// features whose presence follows the shape of a tensor, which the list leaves to training.
void expectEveryOmissionListed(const nlohmann::json& minimal,
                               const std::vector<std::pair<std::string, nlohmann::json>>& forms,
                               bool (*shapeDependent)(const std::string&) = nullptr)
{
    for(const auto& [name, form] : forms)
    {
        SCOPED_TRACE(name);
        std::set<std::string> differing;
        for(const auto& [one, other] : {std::pair{&minimal, &form}, std::pair{&form, &minimal}})
        {
            for(const auto& item : one->items())
            {
                if(!other->contains(item.key())
                   && (shapeDependent == nullptr || !shapeDependent(item.key())))
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
}

bool isDimOrStride(const std::string& feature)
{
    return feature.find(".dims[") != std::string::npos
           || feature.find(".strides[") != std::string::npos;
}

// RFC 0019 section 6.7 (absent is not zero) and section 6.8 rule 9: a model must read every
// feature a valid graph may leave unbound through value_or_default or behind present, and
// this list is them. So each feature one valid form of attention publishes and another omits
// must be listed, whichever way round. Tensor ranks are equal across these forms: how many
// dims a tensor publishes is shape, which training sees as a column absent in some rows.
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
    expectEveryOmissionListed(minimal,
                              {{"dropout", features(attentionGraph(dropout))},
                               {"stated", features(attentionGraph(stated))},
                               {"masked", features(attentionGraph(masked))},
                               {"paged variable-length", features(attentionGraph(paged))}});
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

SdpaBackwardAttributesT backwardAttention()
{
    SdpaBackwardAttributesT sdpa;
    sdpa.q_tensor_uid = 1;
    sdpa.k_tensor_uid = 2;
    sdpa.v_tensor_uid = 3;
    sdpa.o_tensor_uid = 4;
    sdpa.do_tensor_uid = 5;
    sdpa.stats_tensor_uid = 6;
    sdpa.dq_tensor_uid = 7;
    sdpa.dk_tensor_uid = 8;
    sdpa.dv_tensor_uid = 9;
    return sdpa;
}

// Q [2, 8, 3, 16], K and V [2, 2, 7, *], O and dO [2, 8, 3, 32], stats [2, 8, 3, 1], their
// gradients, and nine [2, 1, 1, 1] operands for uids 10-18.
GraphT backwardAttentionGraph(SdpaBackwardAttributesT sdpa)
{
    GraphT graph;
    addTensor(graph, 1, {2, 8, 3, 16}, {384, 48, 16, 1});
    addTensor(graph, 2, {2, 2, 7, 16}, {224, 112, 16, 1});
    addTensor(graph, 3, {2, 2, 7, 32}, {448, 224, 32, 1});
    addTensor(graph, 4, {2, 8, 3, 32}, {768, 96, 32, 1});
    addTensor(graph, 5, {2, 8, 3, 32}, {768, 96, 32, 1});
    addTensor(graph, 6, {2, 8, 3, 1}, {24, 3, 1, 1});
    addTensor(graph, 7, {2, 8, 3, 16}, {384, 48, 16, 1});
    addTensor(graph, 8, {2, 2, 7, 16}, {224, 112, 16, 1});
    addTensor(graph, 9, {2, 2, 7, 32}, {448, 224, 32, 1});
    for(int64_t uid = 10; uid <= 18; ++uid)
    {
        addTensor(graph, uid, {2, 1, 1, 1}, {1, 1, 1, 1});
    }
    auto node = std::make_unique<NodeT>();
    node->compute_data_type = DataType::FLOAT;
    node->attributes.Set(sdpa);
    graph.nodes.push_back(std::move(node));
    return graph;
}

TEST(TestOptionalFeatures, EveryFeatureAValidFormOfBackwardAttentionOmitsIsAnOptionalFeature)
{
    auto dropout = backwardAttention();
    dropout.dropout_probability = 0.1F;
    dropout.seed_tensor_uid = 10;
    dropout.offset_tensor_uid = 11;
    dropout.dropout_mask_tensor_uid = 12;
    dropout.dropout_scale_tensor_uid = 13;
    dropout.dropout_scale_inv_tensor_uid = 14;
    auto stated = backwardAttention();
    stated.attn_scale_value = 0.125F;
    stated.left_bound = 4;
    stated.right_bound = 0;
    auto biased = backwardAttention();
    biased.attn_mask_tensor_uid = 10;
    biased.dbias_tensor_uid = 11;
    biased.scale_tensor_uid = 12;
    auto variable = backwardAttention();
    variable.seq_len_q_tensor_uid = 10;
    variable.seq_len_kv_tensor_uid = 11;
    variable.padding_mask = true;

    expectEveryOmissionListed(features(backwardAttentionGraph(backwardAttention())),
                              {{"dropout", features(backwardAttentionGraph(dropout))},
                               {"stated", features(backwardAttentionGraph(stated))},
                               {"biased", features(backwardAttentionGraph(biased))},
                               {"variable-length", features(backwardAttentionGraph(variable))}});
}

// x [2, 8, N...], w [12, 8, K...], y [2, 12, P...] over @p spatial dimensions, and a vector
// attribute of that length for each of padding, stride and dilation.
GraphT convolutionGraph(size_t spatial)
{
    std::vector<int64_t> x{2, 8};
    std::vector<int64_t> w{12, 8};
    std::vector<int64_t> y{2, 12};
    for(size_t dimension = 0; dimension < spatial; ++dimension)
    {
        x.push_back(7);
        w.push_back(3);
        y.push_back(5);
    }
    const auto contiguous = [](const std::vector<int64_t>& dims) {
        std::vector<int64_t> strides(dims.size(), 1);
        for(size_t dimension = dims.size() - 1; dimension > 0; --dimension)
        {
            strides[dimension - 1] = strides[dimension] * dims[dimension];
        }
        return strides;
    };
    GraphT graph;
    addTensor(graph, 1, x, contiguous(x));
    addTensor(graph, 2, w, contiguous(w));
    addTensor(graph, 3, y, contiguous(y));
    ConvolutionFwdAttributesT conv;
    conv.x_tensor_uid = 1;
    conv.w_tensor_uid = 2;
    conv.y_tensor_uid = 3;
    conv.pre_padding = std::vector<int64_t>(spatial, 0);
    conv.post_padding = std::vector<int64_t>(spatial, 0);
    conv.stride = std::vector<int64_t>(spatial, 1);
    conv.dilation = std::vector<int64_t>(spatial, 1);
    conv.conv_mode = ConvMode::CROSS_CORRELATION;
    auto node = std::make_unique<NodeT>();
    node->compute_data_type = DataType::FLOAT;
    node->attributes.Set(conv);
    graph.nodes.push_back(std::move(node));
    return graph;
}

// How many elements a vector attribute has is the number of spatial dimensions, which the
// schema leaves open. The dims and strides a tensor publishes follow its rank, and are left to
// training.
TEST(TestOptionalFeatures, EveryFeatureAConvolutionOfAnotherDimensionOmitsIsAnOptionalFeature)
{
    expectEveryOmissionListed(
        features(convolutionGraph(1)),
        {{"2-D", features(convolutionGraph(2))}, {"3-D", features(convolutionGraph(3))}},
        &isDimOrStride);
}
} // namespace
