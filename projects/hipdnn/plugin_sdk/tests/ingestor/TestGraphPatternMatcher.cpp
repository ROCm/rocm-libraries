// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/ingestor/BindingPublication.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPatternMatcher.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include "GraphPatternTestSupport.hpp"

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::testing::compilePatternText;
using hipdnn_plugin_sdk::ingestor::testing::convolutionBwdDataNode;
using hipdnn_plugin_sdk::ingestor::testing::convolutionFwdNode;
using hipdnn_plugin_sdk::ingestor::testing::customOpNode;
using hipdnn_plugin_sdk::ingestor::testing::layernormNode;
using hipdnn_plugin_sdk::ingestor::testing::PATTERN_TEST_UED;
using hipdnn_plugin_sdk::ingestor::testing::PatternGraphBuilder;
using hipdnn_plugin_sdk::ingestor::testing::patternTensor;
using hipdnn_plugin_sdk::ingestor::testing::PatternTestGraph;
using hipdnn_plugin_sdk::ingestor::testing::pointwiseNode;

/// The pattern equivalent of the native single-node binary pointwise matcher.
const std::string BINARY_POINTWISE = R"([{"kind": "op", "id": "pointwise", "op": "pointwise",
    "operands": {"in_0": "$input_a", "in_1": "$input_b"}, "results": {"out_0": "$output"}}])";

const std::string OPTIONAL_SECOND_OPERAND = R"([{"kind": "op", "id": "pointwise",
    "op": "pointwise", "operands": {"in_0": "$input_a", "in_1": "$input_b?"},
    "results": {"out_0": "$output"}}])";

const std::string CONV_ADD_CHAIN = R"([
    {"kind": "op", "id": "conv", "op": "convolution_fwd",
     "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
    {"kind": "op", "id": "add", "op": "pointwise",
     "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$y"}}
])";

const std::string TWO_UNARY = R"([
    {"kind": "op", "id": "left", "op": "pointwise",
     "operands": {"in_0": "$a"}, "results": {"out_0": "$b"}},
    {"kind": "op", "id": "right", "op": "pointwise",
     "operands": {"in_0": "$c"}, "results": {"out_0": "$d"}}
])";

std::optional<BoundTokens> match(const std::string& pattern, const PatternGraphBuilder& graph)
{
    const auto compiled = compilePatternText(pattern);
    const auto built = graph.build();
    return matchGraphPattern(*compiled, built.context());
}

bool hasTokenWithRoot(const BoundTokens& bound, const std::string& root)
{
    for(const auto& entry : bound)
    {
        if(entry.first == root || entry.first.rfind(root + ".", 0) == 0)
        {
            return true;
        }
    }
    return false;
}

PatternGraphBuilder binaryPointwiseGraph(PointwiseAttributesT attributes)
{
    attributes.in_0_tensor_uid = 1;
    attributes.in_1_tensor_uid = 2;
    attributes.out_0_tensor_uid = 3;
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).node(pointwiseNode(attributes));
    return graph;
}

PointwiseAttributesT addAttributes()
{
    PointwiseAttributesT attributes;
    attributes.operation = PointwiseMode::ADD;
    return attributes;
}

/// A one-node `in_0 + in_1 -> out_0` graph whose node tags its pointwise table as @p type.
/// Built without the object API, which cannot write a discriminant its union lacks.
flatbuffers::FlatBufferBuilder pointwiseGraphTaggedAs(NodeAttributes type)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    for(const int64_t uid : {1, 2, 3})
    {
        const auto tensor = patternTensor(uid);
        tensors.push_back(TensorAttributes::Pack(builder, tensor.get()));
    }
    auto attributes = addAttributes();
    attributes.in_0_tensor_uid = 1;
    attributes.in_1_tensor_uid = 2;
    attributes.out_0_tensor_uid = 3;
    const auto table = PointwiseAttributes::Pack(builder, &attributes).Union();
    const auto name = builder.CreateString("pointwise");
    const std::vector<flatbuffers::Offset<Node>> nodes{
        CreateNode(builder, name, DataType::FLOAT, type, table)};
    const auto tensorVector = builder.CreateVector(tensors);
    const auto nodeVector = builder.CreateVector(nodes);
    GraphBuilder graph(builder);
    graph.add_tensors(tensorVector);
    graph.add_nodes(nodeVector);
    builder.Finish(graph.Finish());
    return builder;
}

TEST(TestGraphPatternMatcher, SinglePointwisePublishesTheCanonicalTokenMap)
{
    const auto compiled = compilePatternText(BINARY_POINTWISE);
    const auto graph = binaryPointwiseGraph(addAttributes()).build();
    const auto context = graph.context();

    const auto bound = matchGraphPattern(*compiled, context);
    ASSERT_TRUE(bound.has_value());

    BoundTokens expected;
    const auto& tensors = *context.graph.getGraph().tensors();
    ASSERT_TRUE(publishGraph(expected, context.graph));
    ASSERT_TRUE(publishTensor(expected, "input_a", tensors.Get(0)));
    ASSERT_TRUE(publishTensor(expected, "input_b", tensors.Get(1)));
    ASSERT_TRUE(publishTensor(expected, "output", tensors.Get(2)));
    expected.emplace("pointwise.operation", std::string("ADD"));
    expected.emplace("pointwise.compute_data_type", std::string("FLOAT"));
    EXPECT_EQ(*bound, expected);
    EXPECT_EQ(tryGetBoundInt(*bound, "input_b"), 2);
}

TEST(TestGraphPatternMatcher, PublishesOptionalAttributesOnlyWhenSet)
{
    auto attributes = addAttributes();
    attributes.relu_lower_clip = 0.5F;
    attributes.axis_tensor_uid = 7;
    const auto bound = match(BINARY_POINTWISE, binaryPointwiseGraph(attributes));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(std::get<double>(bound->at("pointwise.relu_lower_clip")), 0.5);
    EXPECT_EQ(tryGetBoundInt(*bound, "pointwise.axis_tensor_uid"), 7);
    EXPECT_EQ(bound->count("pointwise.relu_upper_clip"), 0U);
    EXPECT_EQ(bound->count("pointwise.swish_beta"), 0U);
}

TEST(TestGraphPatternMatcher, DeclinesNonfiniteFloatAttribute)
{
    for(const float value :
        {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        auto attributes = addAttributes();
        attributes.swish_beta = value;
        EXPECT_FALSE(match(BINARY_POINTWISE, binaryPointwiseGraph(attributes)).has_value());
    }
}

TEST(TestGraphPatternMatcher, DeclinesNodeCountMismatch)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4);
    graph.node(pointwiseNode(PointwiseMode::ADD, 1, 2, 3));
    graph.node(pointwiseNode(PointwiseMode::RELU_FWD, 3, std::nullopt, 4));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesEmptyGraph)
{
    EXPECT_FALSE(match(BINARY_POINTWISE, PatternGraphBuilder{}).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesWrongOpType)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).node(convolutionFwdNode(1, 2, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesMissingRequiredOperand)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesEdgeNamingAnUnknownTensor)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ADD, 1, 9, 3));
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesGraphRepeatingATensorUid)
{
    auto graph = binaryPointwiseGraph(addAttributes());
    ASSERT_TRUE(match(BINARY_POINTWISE, graph).has_value());
    graph.tensor(3);
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesTensorTwoNodesProduce)
{
    // Neither pattern node reads the other's output, so no variable links them. Only the
    // graph's one-producer rule catches both graph nodes writing uid 3.
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3);
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    graph.node(pointwiseNode(PointwiseMode::EXP, 2, std::nullopt, 3));
    EXPECT_FALSE(match(TWO_UNARY, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesNodeTypeTheRegistryDoesNotDescribe)
{
    const auto compiled = compilePatternText(BINARY_POINTWISE);
    const PatternTestGraph known(pointwiseGraphTaggedAs(NodeAttributes::PointwiseAttributes));
    ASSERT_TRUE(matchGraphPattern(*compiled, known.context()).has_value());

    // One past the last discriminant in this build's union, as a graph from a newer schema
    // might carry. The verifier accepts it, so the matcher must reject it.
    const auto newer = static_cast<NodeAttributes>(static_cast<uint8_t>(NodeAttributes::MAX) + 1);
    const PatternTestGraph unknown(pointwiseGraphTaggedAs(newer));
    EXPECT_FALSE(matchGraphPattern(*compiled, unknown.context()).has_value());
}

TEST(TestGraphPatternMatcher, OptionalOperandMayBeAbsentAndPublishesOnlyWhenPresent)
{
    PatternGraphBuilder unary;
    unary.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 3));
    const auto absent = match(OPTIONAL_SECOND_OPERAND, unary);
    ASSERT_TRUE(absent.has_value());
    EXPECT_FALSE(hasTokenWithRoot(*absent, "input_b"));
    EXPECT_EQ(tryGetBoundInt(*absent, "input_a"), 1);

    const auto present = match(OPTIONAL_SECOND_OPERAND, binaryPointwiseGraph(addAttributes()));
    ASSERT_TRUE(present.has_value());
    EXPECT_EQ(tryGetBoundInt(*present, "input_b"), 2);
    EXPECT_EQ(tryGetBoundInt(*present, "input_b.rank"), 4);
}

TEST(TestGraphPatternMatcher, DeclinesUnnamedOptionalOperandThatIsPresent)
{
    const std::string unaryPattern = R"([{"kind": "op", "id": "pw", "op": "pointwise",
        "operands": {"in_0": "$a"}, "results": {"out_0": "$out"}}])";
    EXPECT_FALSE(match(unaryPattern, binaryPointwiseGraph(addAttributes())).has_value());

    auto ternary = addAttributes();
    ternary.in_2_tensor_uid = 4;
    auto graph = binaryPointwiseGraph(ternary);
    graph.tensor(4);
    EXPECT_FALSE(match(BINARY_POINTWISE, graph).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesUnnamedOptionalResultThatIsPresent)
{
    const std::string withoutMean = R"([{"kind": "op", "id": "norm", "op": "layernorm",
        "operands": {"x": "$x", "scale": "$scale", "bias": "$bias", "epsilon": "$epsilon"},
        "results": {"y": "$y"}}])";
    const std::string withMean = R"([{"kind": "op", "id": "norm", "op": "layernorm",
        "operands": {"x": "$x", "scale": "$scale", "bias": "$bias", "epsilon": "$epsilon"},
        "results": {"y": "$y", "mean": "$mean"}}])";
    const auto layernormGraph = [](std::optional<int64_t> mean) {
        PatternGraphBuilder graph;
        graph.tensor(1).tensor(2).tensor(3).tensor(4).tensor(5).tensor(6);
        graph.node(layernormNode(1, 2, 3, 4, 5, mean));
        return graph;
    };

    EXPECT_TRUE(match(withoutMean, layernormGraph(std::nullopt)).has_value());
    EXPECT_FALSE(match(withoutMean, layernormGraph(6)).has_value());

    const auto named = match(withMean, layernormGraph(6));
    ASSERT_TRUE(named.has_value());
    EXPECT_EQ(tryGetBoundInt(*named, "mean"), 6);
}

TEST(TestGraphPatternMatcher, AliasedOperandsMatchDistinctVariables)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(3).node(pointwiseNode(PointwiseMode::MUL, 1, 1, 3));
    const auto bound = match(BINARY_POINTWISE, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "input_a"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "input_b"), 1);
}

PatternGraphBuilder convAddGraph(int64_t addInput)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3, true).tensor(4).tensor(5).tensor(6);
    graph.node(convolutionFwdNode(1, 2, 3));
    graph.node(pointwiseNode(PointwiseMode::ADD, addInput, 4, 5));
    return graph;
}

TEST(TestGraphPatternMatcher, ChainBindsIntermediateConsistently)
{
    const auto bound = match(CONV_ADD_CHAIN, convAddGraph(3));
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "x"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "w"), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, "conv_out"), 3);
    EXPECT_TRUE(std::get<bool>(bound->at("conv_out.virtual")));
    EXPECT_EQ(tryGetBoundInt(*bound, "bias"), 4);
    EXPECT_EQ(tryGetBoundInt(*bound, "y"), 5);
    EXPECT_EQ(std::get<std::string>(bound->at("conv.conv_mode")), "CROSS_CORRELATION");
    EXPECT_EQ(std::get<std::string>(bound->at("add.operation")), "ADD");
    EXPECT_EQ(tryGetBoundInt(*bound, "graph.node_count"), 2);
}

TEST(TestGraphPatternMatcher, DeclinesChainWhoseConsumerReadsAnotherTensor)
{
    EXPECT_FALSE(match(CONV_ADD_CHAIN, convAddGraph(6)).has_value());
}

TEST(TestGraphPatternMatcher, DeclinesGraphInputVariableProducedInsideTheGraph)
{
    // No pattern node produces `$a`, so it must be a graph input. Here the convolution
    // computes it instead.
    const std::string disconnected = R"([
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$a", "in_1": "$bias"}, "results": {"out_0": "$y"}}
    ])";
    EXPECT_FALSE(match(disconnected, convAddGraph(3)).has_value());
    EXPECT_TRUE(match(disconnected, convAddGraph(6)).has_value());
}

TEST(TestGraphPatternMatcher, CountsATensorACustomOpListProducesAsProduced)
{
    // A custom op reads and writes uid lists, which a pattern cannot name, so the pattern
    // node binds no edges. `$a` is a graph input, and uid 2 is the custom op's output.
    const std::string customThenPointwise = R"([
        {"kind": "op", "id": "custom", "op": "custom_op"},
        {"kind": "op", "id": "pw", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$y"}}
    ])";
    const auto graphWithPointwise = [](int64_t input, int64_t output) {
        PatternGraphBuilder graph;
        graph.tensor(1).tensor(2).tensor(3);
        graph.node(customOpNode({1}, {2}));
        graph.node(pointwiseNode(PointwiseMode::RELU_FWD, input, std::nullopt, output));
        return graph;
    };

    const auto sharedInput = match(customThenPointwise, graphWithPointwise(1, 3));
    ASSERT_TRUE(sharedInput.has_value());
    EXPECT_EQ(tryGetBoundInt(*sharedInput, "a"), 1);

    EXPECT_FALSE(match(customThenPointwise, graphWithPointwise(2, 3)).has_value());
    // The pointwise node also writes uid 2, giving it a second producer.
    EXPECT_FALSE(match(customThenPointwise, graphWithPointwise(1, 2)).has_value());
}

TEST(TestGraphPatternMatcher, OpcodeSetMatchesEitherMember)
{
    const std::string anyConvolution = R"([{"kind": "op", "id": "conv",
        "op": {"one_of": ["convolution_fwd", "convolution_bwd_data"]},
        "operands": {"w": "$w"}}])";

    PatternGraphBuilder forward;
    forward.tensor(1).tensor(2).tensor(3).node(convolutionFwdNode(1, 2, 3));
    const auto forwardBound = match(anyConvolution, forward);
    ASSERT_TRUE(forwardBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*forwardBound, "w"), 2);
    EXPECT_EQ(std::get<std::string>(forwardBound->at("conv.conv_mode")), "CROSS_CORRELATION");

    PatternGraphBuilder backward;
    backward.tensor(1).tensor(2).tensor(3).node(convolutionBwdDataNode(1, 2, 3));
    const auto backwardBound = match(anyConvolution, backward);
    ASSERT_TRUE(backwardBound.has_value());
    EXPECT_EQ(tryGetBoundInt(*backwardBound, "w"), 2);
    EXPECT_EQ(std::get<std::string>(backwardBound->at("conv.conv_mode")), "CONVOLUTION");
}

TEST(TestGraphPatternMatcher, MatchesChainWhoseGraphListsTheConsumerFirst)
{
    const std::string unaryChain = R"([
        {"kind": "op", "id": "first", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$mid"}},
        {"kind": "op", "id": "second", "op": "pointwise",
         "operands": {"in_0": "$mid"}, "results": {"out_0": "$out"}}
    ])";
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3);
    graph.node(pointwiseNode(PointwiseMode::EXP, 2, std::nullopt, 3));
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));

    const auto bound = match(unaryChain, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "a"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "mid"), 2);
    EXPECT_EQ(tryGetBoundInt(*bound, "out"), 3);
    EXPECT_EQ(std::get<std::string>(bound->at("first.operation")), "ABS");
    EXPECT_EQ(std::get<std::string>(bound->at("second.operation")), "EXP");
}

TEST(TestGraphPatternMatcher, UndoesAnAssignmentThatStrandsALaterNode)
{
    // The match order is [p, q]. p first takes node 0 with `$b` absent, then q refuses
    // node 1, whose in_1 it leaves unnamed. Only after p's bindings are undone can p take
    // node 1 and q take node 0.
    const std::string pattern = R"([
        {"kind": "op", "id": "p", "op": "pointwise",
         "operands": {"in_0": "$a", "in_1": "$b?"}, "results": {"out_0": "$p_out"}},
        {"kind": "op", "id": "q", "op": "pointwise",
         "operands": {"in_0": "$c"}, "results": {"out_0": "$q_out"}}
    ])";
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4).tensor(5);
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));
    graph.node(pointwiseNode(PointwiseMode::ADD, 3, 5, 4));

    const auto bound = match(pattern, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "a"), 3);
    EXPECT_EQ(tryGetBoundInt(*bound, "b"), 5);
    EXPECT_EQ(tryGetBoundInt(*bound, "c"), 1);
    EXPECT_EQ(std::get<std::string>(bound->at("p.operation")), "ADD");
    EXPECT_EQ(std::get<std::string>(bound->at("q.operation")), "ABS");
}

TEST(TestGraphPatternMatcher, DeclinesAndWarnsWhenTheSearchExceedsItsBudget)
{
    auto recorder
        = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_WARN);

    // Ten interchangeable unary nodes. If the last graph node is binary, no complete
    // assignment exists. Proving that means trying every order of the other nine, which
    // exceeds the search budget.
    constexpr int64_t NODE_COUNT = 10;
    nlohmann::json nodes = nlohmann::json::array();
    for(int64_t i = 0; i < NODE_COUNT; ++i)
    {
        const auto index = std::to_string(i);
        nlohmann::json node;
        node["kind"] = "op";
        node["id"] = "n" + index;
        node["op"] = "pointwise";
        node["operands"]["in_0"] = "$in" + index;
        node["results"]["out_0"] = "$out" + index;
        nodes.push_back(std::move(node));
    }
    const auto compiled
        = compileGraphPattern(parseGraphPattern(nodes, PATTERN_TEST_UED), PATTERN_TEST_UED);
    const auto graphEndingIn = [](bool binary) {
        PatternGraphBuilder graph;
        for(int64_t uid = 1; uid <= (2 * NODE_COUNT) + 1; ++uid)
        {
            graph.tensor(uid);
        }
        for(int64_t i = 0; i + 1 < NODE_COUNT; ++i)
        {
            graph.node(pointwiseNode(PointwiseMode::ABS, (2 * i) + 1, std::nullopt, (2 * i) + 2));
        }
        const int64_t lastInput = (2 * NODE_COUNT) - 1;
        graph.node(binary
                       ? pointwiseNode(PointwiseMode::ADD, lastInput, lastInput + 2, lastInput + 1)
                       : pointwiseNode(PointwiseMode::ABS, lastInput, std::nullopt, lastInput + 1));
        return graph;
    };

    const auto unary = graphEndingIn(false).build();
    EXPECT_TRUE(matchGraphPattern(*compiled, unary.context()).has_value());
    EXPECT_FALSE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "search steps"))
        << recorder.getRecordedLogsAsString();

    const auto lastBinary = graphEndingIn(true).build();
    EXPECT_FALSE(matchGraphPattern(*compiled, lastBinary.context()).has_value());
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, PATTERN_TEST_UED))
        << recorder.getRecordedLogsAsString();
    EXPECT_TRUE(recorder.hasLogContaining(HIPDNN_SEV_WARN, "search steps"))
        << recorder.getRecordedLogsAsString();
}

TEST(TestGraphPatternMatcher, SearchesEveryConsumerOfABoundTensor)
{
    // `$mid` has two consumers. The binary node must take the second one, and the unary
    // node must take the first, which the binary node already rejected.
    const std::string fanOut = R"([
        {"kind": "op", "id": "source", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$mid"}},
        {"kind": "op", "id": "binary", "op": "pointwise",
         "operands": {"in_0": "$mid", "in_1": "$b"}, "results": {"out_0": "$sum"}},
        {"kind": "op", "id": "unary", "op": "pointwise",
         "operands": {"in_0": "$mid"}, "results": {"out_0": "$exp"}}
    ])";
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4).tensor(5);
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));
    graph.node(pointwiseNode(PointwiseMode::EXP, 2, std::nullopt, 4));
    graph.node(pointwiseNode(PointwiseMode::ADD, 2, 5, 3));

    const auto bound = match(fanOut, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(std::get<std::string>(bound->at("binary.operation")), "ADD");
    EXPECT_EQ(tryGetBoundInt(*bound, "b"), 5);
    EXPECT_EQ(tryGetBoundInt(*bound, "sum"), 3);
    EXPECT_EQ(std::get<std::string>(bound->at("unary.operation")), "EXP");
    EXPECT_EQ(tryGetBoundInt(*bound, "exp"), 4);
}

TEST(TestGraphPatternMatcher, AmbiguousGraphBindsTheFirstAssignment)
{
    PatternGraphBuilder graph;
    graph.tensor(1).tensor(2).tensor(3).tensor(4);
    graph.node(pointwiseNode(PointwiseMode::ABS, 1, std::nullopt, 2));
    graph.node(pointwiseNode(PointwiseMode::EXP, 3, std::nullopt, 4));

    const auto bound = match(TWO_UNARY, graph);
    ASSERT_TRUE(bound.has_value());
    EXPECT_EQ(tryGetBoundInt(*bound, "a"), 1);
    EXPECT_EQ(tryGetBoundInt(*bound, "c"), 3);
    EXPECT_EQ(std::get<std::string>(bound->at("left.operation")), "ABS");
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
