// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

namespace
{
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
    // The language declines these rather than producing NaN or infinity; a feature row
    // must refuse the decline rather than substitute a number for it.
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

} // namespace
