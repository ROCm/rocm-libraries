// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <algorithm>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/CompiledGraphPattern.hpp>

#include "GraphPatternTestSupport.hpp"

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::ingestor::testing::compilePatternText;

std::string compileError(const std::string& nodes)
{
    try
    {
        compilePatternText(nodes);
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        return error.what();
    }
    ADD_FAILURE() << "compiled: " << nodes;
    return {};
}

const PublishedSymbol* findSymbol(const CompiledGraphPattern& pattern, const std::string& name)
{
    const auto& symbols = pattern.publishedSymbols();
    const auto it = std::find_if(symbols.begin(), symbols.end(), [&name](const auto& symbol) {
        return symbol.name == name;
    });
    return it == symbols.end() ? nullptr : &*it;
}

void expectSymbol(const CompiledGraphPattern& pattern,
                  const std::string& name,
                  SymbolKind kind,
                  bool optional)
{
    const auto* symbol = findSymbol(pattern, name);
    ASSERT_NE(symbol, nullptr) << name;
    EXPECT_EQ(symbol->kind, kind) << name;
    EXPECT_EQ(symbol->optional, optional) << name;
}

const std::string CONV_RELU_CHAIN = R"([
    {"kind": "op", "id": "conv", "op": "convolution_fwd",
     "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
    {"kind": "op", "id": "add", "op": "pointwise",
     "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$add_out"}},
    {"kind": "op", "id": "act", "op": "pointwise",
     "operands": {"in_0": "$add_out"}, "results": {"out_0": "$y"}}
])";

const std::string NORM_SET_PREFIX = R"([{"kind": "op", "id": "norm",
    "op": {"one_of": ["rms_norm", "layernorm"]},)";

TEST(TestCompiledGraphPattern, RejectsUnknownOpcodeNamingUedNodeAndOpcode)
{
    EXPECT_EQ(compileError(R"([{"kind": "op", "id": "conv", "op": "convolution_fwdd"}])"),
              "node 'conv' in test.ued.json: opcode 'convolution_fwdd' is not in the op-schema "
              "registry");
}

TEST(TestCompiledGraphPattern, RejectsOpcodeSetWithOneUnknownMember)
{
    EXPECT_EQ(compileError(
                  R"([{"kind": "op", "id": "pw", "op": {"one_of": ["pointwise", "pointwize"]}}])"),
              "node 'pw' in test.ued.json: opcode 'pointwize' is not in the op-schema registry");
}

TEST(TestCompiledGraphPattern, RejectsOperandTheOpDoesNotDeclare)
{
    EXPECT_EQ(compileError(R"([{"kind": "op", "id": "pw", "op": "pointwise",
                                "operands": {"in_3": "$a"}}])"),
              "node 'pw' in test.ued.json: opcode 'pointwise' declares no operand 'in_3'");
}

TEST(TestCompiledGraphPattern, RejectsResultNameBoundAsOperand)
{
    EXPECT_THAT(compileError(R"([{"kind": "op", "id": "pw", "op": "pointwise",
                                  "operands": {"out_0": "$a"}}])"),
                ::testing::HasSubstr("node 'pw' in test.ued.json: opcode 'pointwise' declares no "
                                     "operand 'out_0'; it is a result, so it belongs under "
                                     "'results'"));
    EXPECT_THAT(compileError(R"([{"kind": "op", "id": "conv", "op": "convolution_fwd",
                                  "results": {"x": "$a"}}])"),
                ::testing::HasSubstr("declares no result 'x'; it is an operand"));
}

TEST(TestCompiledGraphPattern, RejectsOptionalBindingOnRequiredOperand)
{
    EXPECT_EQ(compileError(R"([{"kind": "op", "id": "pw", "op": "pointwise",
                                "operands": {"in_0": "$a?"}}])"),
              "node 'pw' in test.ued.json: operand 'in_0' of opcode 'pointwise' is required, so "
              "binding '$a?' cannot be optional");
}

TEST(TestCompiledGraphPattern, RequiredBindingOnOptionalOperandNarrowsThePattern)
{
    const auto pattern = compilePatternText(R"([{"kind": "op", "id": "pw", "op": "pointwise",
        "operands": {"in_0": "$a", "in_1": "$b"}, "results": {"out_0": "$c"}}])");
    expectSymbol(*pattern, "b", SymbolKind::TENSOR, false);
    // in_1 is named, so only in_2 is left as an optional edge the graph must not supply.
    const auto& candidate = pattern->nodes().front().candidates.front();
    ASSERT_EQ(candidate.unnamedOptionalEdges.size(), 1U);
    EXPECT_EQ(candidate.unnamedOptionalEdges.front()->name, "in_2");
}

TEST(TestCompiledGraphPattern, RejectsOpcodeSetDisagreeingOnEdgeOptionality)
{
    // rms_norm declares `bias` optional, layernorm declares it required.
    EXPECT_EQ(compileError(NORM_SET_PREFIX + R"("operands": {"bias": "$bias"}}])"),
              "node 'norm' in test.ued.json: one_of members disagree on operand 'bias': opcode "
              "'rms_norm' declares it optional, opcode 'layernorm' declares it required");
}

TEST(TestCompiledGraphPattern, RejectsOpcodeSetMemberMissingAnEdge)
{
    // layernorm produces `mean`; rms_norm has no such result.
    EXPECT_EQ(compileError(NORM_SET_PREFIX + R"("results": {"mean": "$mean"}}])"),
              "node 'norm' in test.ued.json: opcode 'rms_norm' declares no result 'mean'");
}

TEST(TestCompiledGraphPattern, OpcodeSetPublishesOnlyAttributesEveryMemberDeclares)
{
    const auto pattern = compilePatternText(
        NORM_SET_PREFIX + R"("operands": {"x": "$x", "scale": "$scale", "epsilon": "$epsilon"},
             "results": {"y": "$y"}}])");
    expectSymbol(*pattern, "norm.forward_phase", SymbolKind::ENUM_NAME, false);
    expectSymbol(*pattern, "norm.compute_data_type", SymbolKind::ENUM_NAME, false);
    // Only layernorm declares normalized_dim_count.
    EXPECT_EQ(findSymbol(*pattern, "norm.normalized_dim_count"), nullptr);

    const auto& node = pattern->nodes().front();
    ASSERT_EQ(node.candidates.size(), 2U);
    EXPECT_EQ(node.candidates[0].schema->opcode, "rms_norm");
    EXPECT_EQ(node.candidates[1].schema->opcode, "layernorm");
    for(const auto& candidate : node.candidates)
    {
        ASSERT_EQ(candidate.attributes.size(), node.attributes.size());
        for(size_t i = 0; i < node.attributes.size(); ++i)
        {
            EXPECT_EQ(node.attributes[i].token,
                      "norm." + std::string(candidate.attributes[i]->name));
        }
    }
}

TEST(TestCompiledGraphPattern, RootOpcodesAreTheChainEntry)
{
    EXPECT_EQ(compilePatternText(CONV_RELU_CHAIN)->rootOpcodes(),
              std::vector<std::string>{"convolution_fwd"});
}

TEST(TestCompiledGraphPattern, RootOpcodesUnionEveryIndependentEntry)
{
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "join", "op": "pointwise",
         "operands": {"in_0": "$conv_out", "in_1": "$scaled"}, "results": {"out_0": "$y"}},
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "scale", "op": {"one_of": ["pointwise", "matmul"]},
         "results": {}}
    ])");
    EXPECT_EQ(pattern->rootOpcodes(),
              (std::vector<std::string>{"convolution_fwd", "matmul", "pointwise"}));
}

TEST(TestCompiledGraphPattern, PointwiseNodePublishesScalarAttributesOnly)
{
    const auto pattern = compilePatternText(R"([{"kind": "op", "id": "pw", "op": "pointwise",
        "operands": {"in_0": "$a", "in_1": "$b?"}, "results": {"out_0": "$c"}}])");
    expectSymbol(*pattern, "pw.operation", SymbolKind::ENUM_NAME, false);
    expectSymbol(*pattern, "pw.compute_data_type", SymbolKind::ENUM_NAME, false);
    expectSymbol(*pattern, "pw.relu_lower_clip", SymbolKind::FLOAT, true);
    expectSymbol(*pattern, "pw.axis_tensor_uid", SymbolKind::INT, true);
    expectSymbol(*pattern, "a", SymbolKind::TENSOR, false);
    expectSymbol(*pattern, "b", SymbolKind::TENSOR, true);
    expectSymbol(*pattern, "c", SymbolKind::TENSOR, false);
    expectSymbol(*pattern, "graph.node_count", SymbolKind::INT, false);
    expectSymbol(*pattern, "graph.is_override_shape_enabled", SymbolKind::BOOL, false);
    // Edge uid fields are tensors, not attributes.
    EXPECT_EQ(findSymbol(*pattern, "pw.in_0"), nullptr);
    EXPECT_EQ(findSymbol(*pattern, "pw.in_0_tensor_uid"), nullptr);

    const auto& symbols = pattern->publishedSymbols();
    EXPECT_TRUE(
        std::is_sorted(symbols.begin(), symbols.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.name < rhs.name;
        }));
}

TEST(TestCompiledGraphPattern, ConvolutionNodeSkipsVectorAttributes)
{
    const auto pattern = compilePatternText(CONV_RELU_CHAIN);
    expectSymbol(*pattern, "conv.conv_mode", SymbolKind::ENUM_NAME, false);
    expectSymbol(*pattern, "conv.compute_data_type", SymbolKind::ENUM_NAME, false);
    EXPECT_EQ(findSymbol(*pattern, "conv.dilation"), nullptr);
    EXPECT_EQ(findSymbol(*pattern, "conv.stride"), nullptr);
    EXPECT_EQ(findSymbol(*pattern, "conv.pre_padding"), nullptr);
}

TEST(TestCompiledGraphPattern, VariablesRecordBindingAndReadingEdges)
{
    const auto pattern = compilePatternText(CONV_RELU_CHAIN);

    const auto* intermediate = pattern->findVariable("conv_out");
    ASSERT_NE(intermediate, nullptr);
    EXPECT_EQ(intermediate->binding.nodeId, "conv");
    EXPECT_EQ(intermediate->binding.edge, "y");
    EXPECT_EQ(intermediate->binding.direction, EdgeDirection::RESULT);
    ASSERT_EQ(intermediate->readers.size(), 1U);
    EXPECT_EQ(intermediate->readers.front().nodeId, "add");
    EXPECT_EQ(intermediate->readers.front().edge, "in_0");

    const auto* input = pattern->findVariable("bias");
    ASSERT_NE(input, nullptr);
    EXPECT_EQ(input->binding.nodeId, "add");
    EXPECT_EQ(input->binding.edge, "in_1");
    EXPECT_EQ(input->binding.direction, EdgeDirection::OPERAND);
    EXPECT_TRUE(input->readers.empty());

    EXPECT_EQ(pattern->findVariable("missing"), nullptr);
}

TEST(TestCompiledGraphPattern, MatchOrderPutsProducersFirst)
{
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "act", "op": "pointwise",
         "operands": {"in_0": "$add_out"}, "results": {"out_0": "$y"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$add_out"}},
        {"kind": "op", "id": "other", "op": "pointwise", "operands": {"in_0": "$z"}},
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}}
    ])");
    EXPECT_EQ(pattern->matchOrder(), (std::vector<size_t>{2, 3, 1, 0}));
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
