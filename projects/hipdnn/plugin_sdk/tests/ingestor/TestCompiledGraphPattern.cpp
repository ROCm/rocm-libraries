// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <string>
#include <vector>

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

TEST(TestCompiledGraphPattern, RejectsOperandNameBoundAsResult)
{
    EXPECT_EQ(compileError(R"([{"kind": "op", "id": "conv", "op": "convolution",
                                "results": {"x": "$a"}}])"),
              "node 'conv' in test.ued.json: opcode 'convolution' declares no result 'x'; it "
              "is an operand, so it belongs under 'operands'");
}

TEST(TestCompiledGraphPattern, RootOpcodesAreTheChainEntry)
{
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "conv", "op": "convolution",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$add_out"}},
        {"kind": "op", "id": "act", "op": "pointwise",
         "operands": {"in_0": "$add_out"}, "results": {"out_0": "$y"}}
    ])");
    EXPECT_EQ(pattern->rootOpcodes(), std::vector<std::string>{"convolution"});
}

TEST(TestCompiledGraphPattern, RootOpcodesUnionEveryIndependentEntry)
{
    // `join` reads both entries' results, so its opcode is not a root opcode.
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "join", "op": "matmul",
         "operands": {"a": "$conv_out", "b": "$scaled"}, "results": {"c": "$y"}},
        {"kind": "op", "id": "conv", "op": "convolution",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "scale", "op": {"one_of": ["rmsnorm", "layernorm"]},
         "operands": {"x": "$z"}, "results": {"y": "$scaled"}}
    ])");
    EXPECT_EQ(pattern->rootOpcodes(),
              (std::vector<std::string>{"convolution", "layernorm", "rmsnorm"}));
}

TEST(TestCompiledGraphPattern, MatchOrderPutsProducersFirst)
{
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "act", "op": "pointwise",
         "operands": {"in_0": "$add_out"}, "results": {"out_0": "$y"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$add_out"}},
        {"kind": "op", "id": "other", "op": "pointwise", "operands": {"in_0": "$z"}},
        {"kind": "op", "id": "conv", "op": "convolution",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}}
    ])");
    EXPECT_EQ(pattern->matchOrder(), (std::vector<size_t>{2, 3, 1, 0}));
}

/// The reader that sits downstream of the cycle is authored first, so the search for a cycle
/// starts outside it. The message must still name only the nodes that feed each other.
TEST(TestCompiledGraphPattern, RejectsDependencyCycleAndNamesOnlyItsNodes)
{
    const auto message = compileError(R"([
        {"kind": "op", "id": "tail", "op": "pointwise",
         "operands": {"in_0": "$c"}, "results": {"out_0": "$out"}},
        {"kind": "op", "id": "n1", "op": "pointwise",
         "operands": {"in_0": "$c"}, "results": {"out_0": "$a"}},
        {"kind": "op", "id": "n2", "op": "pointwise",
         "operands": {"in_0": "$a"}, "results": {"out_0": "$b"}},
        {"kind": "op", "id": "n3", "op": "pointwise",
         "operands": {"in_0": "$b"}, "results": {"out_0": "$c"}}
    ])");
    EXPECT_NE(message.find("pattern nodes 'n1' -> 'n2' -> 'n3' -> 'n1' in test.ued.json"),
              std::string::npos)
        << message;
}

TEST(TestCompiledGraphPattern, RejectsNodeThatReadsItsOwnResult)
{
    const auto message = compileError(R"([
        {"kind": "op", "id": "loop", "op": "pointwise",
         "operands": {"in_0": "$x"}, "results": {"out_0": "$x"}}
    ])");
    EXPECT_NE(message.find("pattern nodes 'loop' -> 'loop' in test.ued.json"), std::string::npos)
        << message;
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
