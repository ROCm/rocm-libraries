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
    EXPECT_EQ(compileError(R"([{"kind": "op", "id": "conv", "op": "convolution_fwd",
                                "results": {"x": "$a"}}])"),
              "node 'conv' in test.ued.json: opcode 'convolution_fwd' declares no result 'x'; it "
              "is an operand, so it belongs under 'operands'");
}

TEST(TestCompiledGraphPattern, RootOpcodesAreTheChainEntry)
{
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "add", "op": "pointwise",
         "operands": {"in_0": "$conv_out", "in_1": "$bias"}, "results": {"out_0": "$add_out"}},
        {"kind": "op", "id": "act", "op": "pointwise",
         "operands": {"in_0": "$add_out"}, "results": {"out_0": "$y"}}
    ])");
    EXPECT_EQ(pattern->rootOpcodes(), std::vector<std::string>{"convolution_fwd"});
}

TEST(TestCompiledGraphPattern, RootOpcodesUnionEveryIndependentEntry)
{
    // `join` reads both entries' results, so its opcode is not a root opcode.
    const auto pattern = compilePatternText(R"([
        {"kind": "op", "id": "join", "op": "matmul",
         "operands": {"a": "$conv_out", "b": "$scaled"}, "results": {"c": "$y"}},
        {"kind": "op", "id": "conv", "op": "convolution_fwd",
         "operands": {"x": "$x", "w": "$w"}, "results": {"y": "$conv_out"}},
        {"kind": "op", "id": "scale", "op": {"one_of": ["rms_norm", "layernorm"]},
         "operands": {"x": "$z"}, "results": {"y": "$scaled"}}
    ])");
    EXPECT_EQ(pattern->rootOpcodes(),
              (std::vector<std::string>{"convolution_fwd", "layernorm", "rms_norm"}));
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
