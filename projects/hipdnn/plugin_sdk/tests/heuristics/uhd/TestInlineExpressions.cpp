// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/heuristics/uhd/DescriptorExpression.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

namespace
{
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::expression::Error;
using hipdnn_plugin_sdk::uhd::expression::Program;
using hipdnn_plugin_sdk::uhd::expression::VariableContext;
using nlohmann::json;

TEST(TestInlineExpressions, SharedSubexpressionsAndCandidateBindingsAreEvaluatedOnce)
{
    struct CountingData
    {
        VariableContext context;
        mutable std::unordered_map<std::string, size_t> reads;
        const VariableContext::ValueType* getData(const std::string& path) const
        {
            ++reads[path];
            return context.getData(path);
        }
    } data;
    const json shared = json::parse(R"({"*":["$input.batch","$input.heads"]})");
    const json candidate = {{"*", json::array({shared, "$kernel.tile_m"})}};
    const Program program({shared, candidate, {{"+", json::array({candidate, candidate})}}});
    data.context.bind("$input.batch", int64_t{16});
    data.context.bind("$input.heads", int64_t{32});
    auto work = program.workspace();
    program.prepare(data, work);
    EXPECT_DOUBLE_EQ(Program::number(program.evaluate(0, data, work)), 512);
    data.context.bind("$kernel.tile_m", int64_t{64});
    EXPECT_DOUBLE_EQ(Program::number(program.evaluate(1, data, work)), 32768);
    EXPECT_DOUBLE_EQ(Program::number(program.evaluate(2, data, work)), 65536);
    data.context.bind("$kernel.tile_m", int64_t{128});
    program.resetCandidate(work);
    EXPECT_DOUBLE_EQ(Program::number(program.evaluate(2, data, work)), 131072);
    EXPECT_EQ(data.reads.at("$input.batch"), 1u);
    EXPECT_EQ(data.reads.at("$input.heads"), 1u);
    EXPECT_EQ(data.reads.at("$kernel.tile_m"), 2u);
    // Structural reuse is exposed for compilation/partition benchmarks.
    EXPECT_EQ(program.nodeCount(), 6u);
}

TEST(TestInlineExpressions, NestedQuantizationRecomputesOnlyItsCandidateTail)
{
    const json elements = json::parse(R"({"*":["$q.dims[0]","$q.dims[2]"]})");
    const json tiles = json::parse(R"({"ceil_div":["$q.dims[2]","$kernel.tile_m"]})");
    const FeatureExtractor extractor(
        {elements, tiles, {{"ceil_div", json::array({elements, tiles})}}});
    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"q.dims[0]", int64_t{16}}, {"q.dims[2]", int64_t{2048}}});
    auto work = extractor.prepare(ctx);
    ctx.bindKernelVars({{"tile_m", int64_t{64}}});
    extractor.extractKernelInto(ctx, work);
    EXPECT_EQ(work.values, (std::vector<double>{32768, 32, 1024}));
    ctx.clearKernelVars();
    ctx.bindKernelVars({{"tile_m", int64_t{128}}});
    extractor.extractKernelInto(ctx, work);
    EXPECT_EQ(work.values, (std::vector<double>{32768, 16, 2048}));
    EXPECT_EQ(work.values, extractor.extract(ctx));
}

TEST(TestInlineExpressions, IntegerOverflowAndInvalidRealDomainsFailClosed)
{
    VariableContext ctx;
    ctx.bind("$input.n", std::numeric_limits<int64_t>::max());
    const Program overflow({json::parse(R"({"+":["$input.n",1]})")});
    auto work = overflow.workspace();
    EXPECT_THROW(overflow.evaluate(0, ctx, work), Error);
    const Program domain({json::parse(R"({"pow":[-1,0.5]})")});
    work = domain.workspace();
    EXPECT_THROW(domain.evaluate(0, ctx, work), Error);
}

TEST(TestInlineExpressions, UnknownOperatorInUnusedBranchIsRejectedAtCompilation)
{
    EXPECT_THROW(Program({json::parse(R"({"if":[false,{"custom_native":[1]},0]})")}), Error);
}

} // namespace
