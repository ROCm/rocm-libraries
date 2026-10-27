// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/heuristics/uhd/Expressions.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

#include <string>
#include <vector>

namespace
{
using hipdnn_plugin_sdk::uhd::ExpressionSet;
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;
using nlohmann::json;

TEST(TestInlineExpressions, ProblemEntriesAreEvaluatedOncePerSelection)
{
    // RFC 0019 §9.3: an entry that reads no `$kernel` symbol is the same for every candidate,
    // so prepare() evaluates it once and extractKernelInto() leaves it alone.
    const json shared = json::parse(R"({"*":["$input.batch","$input.heads"]})");
    const json candidate = {{"*", json::array({shared, "$kernel.tile_m"})}};
    const FeatureExtractor extractor({shared, candidate});
    EXPECT_EQ(extractor.kernelDependentCount(), 1u);

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"input.batch", int64_t{16}}, {"input.heads", int64_t{32}}});
    auto work = extractor.prepare(ctx);
    ctx.bind("input.batch", int64_t{1});
    ctx.bindKernelVars({{"tile_m", int64_t{64}}});
    extractor.extractKernelInto(ctx, work);
    EXPECT_EQ(work.values, (std::vector<double>{512, 2048}));
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

TEST(TestInlineExpressions, InvalidRealDomainsFailClosed)
{
    // The language declines these instead of yielding NaN or infinity; extraction must throw.
    FeatureExtractionContext ctx;
    ctx.bind("input.n", int64_t{0});
    for(const auto* expression : {R"({"pow":[-1,0.5]})",
                                  R"({"log2":"$input.n"})",
                                  R"({"rsqrt":"$input.n"})",
                                  R"({"/":[1,"$input.n"]})"})
    {
        const FeatureExtractor extractor({json::parse(expression)});
        EXPECT_THROW(extractor.extract(ctx), JsonLogicError) << expression;
    }
}

TEST(TestInlineExpressions, UnknownOperatorInUnusedBranchIsRejectedAtCompilation)
{
    EXPECT_THROW(FeatureExtractor({json::parse(R"({"if":[false,{"custom_native":[1]},0]})")}),
                 JsonLogicError);
}

/// @p innermost wrapped in `{"!": [...]}` until @p operators operators are nested.
json nestOperators(json innermost, size_t operators)
{
    for(size_t i = 1; i < operators; ++i)
    {
        innermost = {{"!", json::array({innermost})}};
    }
    return innermost;
}

TEST(TestExpressionBounds, OperatorNestingIsBoundedPerExpression)
{
    const json scalarArguments = {{"!", json::array({"$q.x"})}};
    // An array-literal argument is not an operator, so it does not count against the bound.
    const json arrayLiteralArgument = {{"in", json::array({"$q.x", json::array({1, 2})})}};

    for(const auto& innermost : {scalarArguments, arrayLiteralArgument})
    {
        SCOPED_TRACE(innermost.dump());
        EXPECT_EQ(
            ExpressionSet({nestOperators(innermost, ExpressionSet::MAX_EXPRESSION_DEPTH)}).size(),
            1U);
        EXPECT_THROW(
            ExpressionSet({nestOperators(innermost, ExpressionSet::MAX_EXPRESSION_DEPTH + 1)}),
            JsonLogicError);
    }
}

/// Hashing validates a signature without building an ExpressionSet, so it needs the same bound.
TEST(TestExpressionBounds, HashingASignatureAppliesTheSameNestingBound)
{
    const json innermost = {{"!", json::array({"$q.x"})}};
    EXPECT_NO_THROW(FeatureExtractor::computeHash(
        {nestOperators(innermost, ExpressionSet::MAX_EXPRESSION_DEPTH)}));
    EXPECT_THROW(FeatureExtractor::computeHash(
                     {nestOperators(innermost, ExpressionSet::MAX_EXPRESSION_DEPTH + 1)}),
                 JsonLogicError);
}

TEST(TestExpressionBounds, ExpressionCountIsBounded)
{
    EXPECT_EQ(ExpressionSet(std::vector<json>(ExpressionSet::MAX_EXPRESSIONS, "$q.x")).size(),
              ExpressionSet::MAX_EXPRESSIONS);
    EXPECT_THROW(ExpressionSet(std::vector<json>(ExpressionSet::MAX_EXPRESSIONS + 1, "$q.x")),
                 JsonLogicError);
}

TEST(TestExpressionBounds, StringLengthIsBounded)
{
    const std::string prefix = "$q.";
    const std::string atLimit
        = prefix + std::string(ExpressionSet::MAX_STRING_BYTES - prefix.size(), 'x');
    EXPECT_EQ(ExpressionSet({json(atLimit)}).size(), 1U);
    EXPECT_THROW(ExpressionSet({json(atLimit + "x")}), JsonLogicError);
}

TEST(TestExpressionBounds, InputNodesAreBoundedAcrossTheWholeSet)
{
    // The object, its argument list and every argument are nodes.
    json arguments = json::array({"$q.x"});
    while(arguments.size() < ExpressionSet::MAX_INPUT_NODES - 2)
    {
        arguments.push_back(1);
    }
    const json atLimit = {{"max", arguments}};
    EXPECT_EQ(ExpressionSet({atLimit}).size(), 1U);
    // One more node in a second expression crosses the shared bound.
    EXPECT_THROW(ExpressionSet({atLimit, json("$q.y")}), JsonLogicError);
}

} // namespace
