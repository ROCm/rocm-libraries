// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/GraphPattern.hpp>

#include "GraphPatternTestSupport.hpp"

namespace
{
using namespace hipdnn_plugin_sdk::ingestor;
using hipdnn_plugin_sdk::ingestor::testing::parsePatternText;
using hipdnn_plugin_sdk::ingestor::testing::PATTERN_TEST_UED;

/// One pointwise node binding operand `in_0` to @p a and result `out_0` to @p b.
std::string pointwiseNode(const std::string& id, const std::string& a, const std::string& b)
{
    return R"({"kind": "op", "id": ")" + id + R"(", "op": "pointwise", "operands": {"in_0": ")" + a
           + R"("}, "results": {"out_0": ")" + b + R"("}})";
}

std::string unaryChain(size_t nodeCount)
{
    nlohmann::json nodes = nlohmann::json::array();
    for(size_t i = 0; i < nodeCount; ++i)
    {
        nodes.push_back({{"kind", "op"},
                         {"id", "n" + std::to_string(i)},
                         {"op", "pointwise"},
                         {"operands", {{"in_0", "$v" + std::to_string(i)}}},
                         {"results", {{"out_0", "$v" + std::to_string(i + 1)}}}});
    }
    return nodes.dump();
}

std::string nodeWithOperandCount(size_t operandCount)
{
    nlohmann::json operands = nlohmann::json::object();
    for(size_t i = 0; i < operandCount; ++i)
    {
        operands["e" + std::to_string(i)] = "$v" + std::to_string(i);
    }
    return nlohmann::json::array({{{"kind", "op"},
                                   {"id", "wide"},
                                   {"op", "pointwise"},
                                   {"operands", operands},
                                   {"results", {{"out_0", "$out"}}}}})
        .dump();
}

std::string nodeWithOpcodeSet(size_t memberCount)
{
    nlohmann::json members = nlohmann::json::array();
    for(size_t i = 0; i < memberCount; ++i)
    {
        members.push_back("op" + std::to_string(i));
    }
    return nlohmann::json::array({{{"kind", "op"}, {"id", "set"}, {"op", {{"one_of", members}}}}})
        .dump();
}

struct MalformedCase
{
    std::string name;
    std::string nodes;
    std::string expected;
};

std::vector<MalformedCase> malformedCases()
{
    const auto pw
        = [](const std::string& a, const std::string& b) { return pointwiseNode("pw", a, b); };
    return {
        {"EmptyBlock", "[]", "must not be empty"},
        {"NonArrayBlock", R"({"kind": "op"})", "must be an array"},
        {"NodeNotAnObject", "[1]", "graph_match.nodes[0] in test.ued.json must be a JSON object"},
        {"KindNotOp",
         R"([{"kind": "tensor", "id": "pw", "op": "pointwise"}])",
         "key 'kind' of node 'pw' in test.ued.json must be \"op\""},
        {"KindMissing", R"([{"id": "pw", "op": "pointwise"}])", "missing required key 'kind'"},
        {"IdMissing", R"([{"kind": "op", "op": "pointwise"}])", "missing required key 'id'"},
        {"OpMissing", R"([{"kind": "op", "id": "pw"}])", "missing required key 'op'"},
        {"IdNotAnIdentifier",
         R"([{"kind": "op", "id": "1pw", "op": "pointwise"}])",
         "must be an identifier"},
        {"IdNotAString",
         R"([{"kind": "op", "id": 7, "op": "pointwise"}])",
         "must be an identifier"},
        {"DuplicateId",
         "[" + pointwiseNode("pw", "$a", "$b") + "," + pointwiseNode("pw", "$b", "$c") + "]",
         "node id 'pw' is used by two nodes"},
        {"ReservedId",
         "[" + pointwiseNode("graph", "$a", "$b") + "]",
         "node id 'graph' is a reserved root"},
        {"ReservedVariable",
         "[" + pw("$kernel", "$b") + "]",
         "variable '$kernel' is a reserved root"},
        {"VariableEqualToNodeId",
         "[" + pw("$a", "$pw") + "]",
         "variable '$pw' collides with node id 'pw'"},
        {"VariableBoundByTwoResults",
         "[" + pointwiseNode("p", "$a", "$out") + "," + pointwiseNode("q", "$b", "$out") + "]",
         "variable '$out' is bound by two results"},
        {"GraphInputReadByTwoNodes",
         "[" + pointwiseNode("p", "$a", "$b") + "," + pointwiseNode("q", "$a", "$c") + "]",
         "graph input variable '$a' is bound by two operands"},
        {"GraphInputReadTwiceByOneNode",
         R"([{"kind": "op", "id": "pw", "op": "pointwise",
              "operands": {"in_0": "$a", "in_1": "$a"}, "results": {"out_0": "$b"}}])",
         "graph input variable '$a' is bound by two operands"},
        {"OptionalResult", "[" + pw("$a", "$b?") + "]", "'?' is only allowed on an operand"},
        {"BindingWithoutSigil", "[" + pw("a", "$b") + "]", "malformed binding 'a'"},
        {"BindingWithMidIdentifierMark", "[" + pw("$a?b", "$b") + "]", "malformed binding '$a?b'"},
        {"BindingWithTwoMarks", "[" + pw("$a??", "$b") + "]", "malformed binding '$a?\?'"},
        {"BindingWithBadIdentifier", "[" + pw("$1a", "$b") + "]", "malformed binding '$1a'"},
        {"BindingNotAString",
         R"([{"kind": "op", "id": "pw", "op": "pointwise", "operands": {"in_0": 3}}])",
         "operand 'in_0' of node 'pw' in test.ued.json must be a string binding"},
        {"OperandsNotAnObject",
         R"([{"kind": "op", "id": "pw", "op": "pointwise", "operands": ["$a"]}])",
         "key 'operands' of node 'pw' in test.ued.json must be an object"},
        {"OpEmpty", R"([{"kind": "op", "id": "pw", "op": ""}])", "must not be empty"},
        {"OpNotStringOrObject",
         R"([{"kind": "op", "id": "pw", "op": 4}])",
         "must be an opcode string"},
        {"OneOfSingleMember",
         R"([{"kind": "op", "id": "pw", "op": {"one_of": ["pointwise"]}}])",
         "an opcode set needs at least 2"},
        {"OneOfDuplicateMember",
         R"([{"kind": "op", "id": "pw", "op": {"one_of": ["pointwise", "pointwise"]}}])",
         "opcode 'pointwise' is listed twice"},
        {"OneOfEmptyMember",
         R"([{"kind": "op", "id": "pw", "op": {"one_of": ["pointwise", ""]}}])",
         "must list non-empty opcode strings"},
        {"OneOfNotAnArray",
         R"([{"kind": "op", "id": "pw", "op": {"one_of": "pointwise"}}])",
         "key 'one_of' in key 'op' of node 'pw' in test.ued.json must be an array"},
        {"OpcodeSetWithoutOneOf",
         R"([{"kind": "op", "id": "pw", "op": {"x-note": "set"}}])",
         "missing required key 'one_of' in key 'op' of node 'pw'"},
        {"UnknownNodeKey",
         R"([{"kind": "op", "id": "pw", "op": "pointwise", "attrs": {}}])",
         "unknown key 'attrs' in graph_match.nodes[0] in test.ued.json"},
        {"UnknownOpcodeSetKey",
         R"([{"kind": "op", "id": "pw", "op": {"one_of": ["matmul", "pointwise"], "any": true}}])",
         "unknown key 'any' in key 'op' of node 'pw' in test.ued.json"},
        {"OptionalBindingOnProducedVariable",
         "[" + pointwiseNode("p", "$a", "$m") + "," + pointwiseNode("q", "$m?", "$out") + "]",
         "optional binding '$m?' on operand 'in_0' names a variable node 'p' result 'out_0'"},
        {"TooManyNodes", unaryChain(MAX_PATTERN_NODES + 1), "nodes; the limit is 32"},
        {"TooManyEdges",
         nodeWithOperandCount(MAX_PATTERN_EDGES_PER_NODE),
         "node 'wide' in test.ued.json binds 65 edges; the limit is 64"},
        {"TooManyOpcodes",
         nodeWithOpcodeSet(MAX_PATTERN_OPCODE_SET + 1),
         "lists 33 opcodes; the limit is 32"},
    };
}

class TestGraphPatternMalformed : public ::testing::TestWithParam<MalformedCase>
{
};

TEST_P(TestGraphPatternMalformed, RejectsNamingUedAndOffendingValue)
{
    const auto& param = GetParam();
    try
    {
        parsePatternText(param.nodes);
        FAIL() << "accepted: " << param.nodes;
    }
    catch(const hipdnn_plugin_sdk::HipdnnPluginException& error)
    {
        EXPECT_EQ(error.getStatus(), HIPDNN_PLUGIN_STATUS_INVALID_VALUE);
        EXPECT_THAT(error.what(), ::testing::HasSubstr(PATTERN_TEST_UED));
        EXPECT_THAT(error.what(), ::testing::HasSubstr(param.expected));
    }
}

INSTANTIATE_TEST_SUITE_P(GraphPattern,
                         TestGraphPatternMalformed,
                         ::testing::ValuesIn(malformedCases()),
                         [](const ::testing::TestParamInfo<MalformedCase>& paramInfo) {
                             return paramInfo.param.name;
                         });

TEST(TestGraphPattern, LimitsAdmitTheirBoundaryValues)
{
    EXPECT_EQ(parsePatternText(unaryChain(MAX_PATTERN_NODES)).nodes().size(), MAX_PATTERN_NODES);
    EXPECT_EQ(parsePatternText(nodeWithOperandCount(MAX_PATTERN_EDGES_PER_NODE - 1))
                  .nodes()
                  .front()
                  .operands.size(),
              MAX_PATTERN_EDGES_PER_NODE - 1);
    EXPECT_EQ(
        parsePatternText(nodeWithOpcodeSet(MAX_PATTERN_OPCODE_SET)).nodes().front().opcodes.size(),
        MAX_PATTERN_OPCODE_SET);
}

TEST(TestGraphPattern, ParsesConvBiasReluWithEdgesSortedByName)
{
    const auto pattern = parsePatternText(R"([
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_1": "$bias", "in_0": "$conv_out"}, "results": {"out_0": "$bias_out"}},
        {"kind": "op", "id": "act", "op": "pointwise",
         "operands": {"in_0": "$bias_out"}, "results": {"out_0": "$y"}}
    ])");

    const auto& nodes = pattern.nodes();
    ASSERT_EQ(nodes.size(), 3U);
    EXPECT_EQ(nodes[0].id, "conv");
    EXPECT_EQ(nodes[0].opcodes, std::vector<std::string>{"convolution_fwd"});
    EXPECT_FALSE(nodes[0].opcodeSet);
    ASSERT_EQ(nodes[0].operands.size(), 2U);
    EXPECT_EQ(nodes[0].operands[0].name, "w");
    EXPECT_EQ(nodes[0].operands[0].binding.variable, "w");
    EXPECT_EQ(nodes[0].operands[1].name, "x");
    ASSERT_EQ(nodes[0].results.size(), 1U);
    EXPECT_EQ(nodes[0].results[0].binding.variable, "conv_out");

    EXPECT_EQ(nodes[1].id, "add");
    ASSERT_EQ(nodes[1].operands.size(), 2U);
    EXPECT_EQ(nodes[1].operands[0].name, "in_0");
    EXPECT_EQ(nodes[1].operands[0].binding.variable, "conv_out");
    EXPECT_EQ(nodes[1].operands[1].name, "in_1");
    EXPECT_EQ(nodes[1].operands[1].binding.variable, "bias");
    EXPECT_FALSE(nodes[1].operands[1].binding.optional);

    EXPECT_EQ(nodes[2].id, "act");
    EXPECT_EQ(nodes[2].results[0].binding.variable, "y");
}

TEST(TestGraphPattern, ProducedVariableMayBeReadByManyNodes)
{
    const auto pattern = parsePatternText("[" + pointwiseNode("p", "$a", "$m") + ","
                                          + pointwiseNode("q", "$m", "$x") + ","
                                          + pointwiseNode("r", "$m", "$y") + "]");
    EXPECT_EQ(pattern.nodes().size(), 3U);
}

TEST(TestGraphPattern, OptionalOperandBindsVariableWithoutMark)
{
    const auto pattern = parsePatternText(
        R"([{"kind": "op", "id": "pw", "op": "pointwise",
             "operands": {"in_0": "$a", "in_1": "$b?"}, "results": {"out_0": "$c"}}])");
    const auto& operands = pattern.nodes().front().operands;
    ASSERT_EQ(operands.size(), 2U);
    EXPECT_EQ(operands[1].binding.variable, "b");
    EXPECT_TRUE(operands[1].binding.optional);
    EXPECT_FALSE(operands[0].binding.optional);
}

TEST(TestGraphPattern, IgnoresExtensionKeysAndKeepsOpcodeSetOrder)
{
    const auto pattern = parsePatternText(R"([
        {"kind": "op", "id": "conv", "x-note": "kept out of the model", "_comment": 1,
         "provenance": {"tool": "test"},
         "op": {"one_of": ["convolution_fwd", "convolution_bwd_data"], "x-why": "either"},
         "operands": {"w": "$w"}}
    ])");
    const auto& node = pattern.nodes().front();
    EXPECT_TRUE(node.opcodeSet);
    EXPECT_EQ(node.opcodes, (std::vector<std::string>{"convolution_fwd", "convolution_bwd_data"}));
    EXPECT_TRUE(node.results.empty());
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
