// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

#include <limits>

namespace
{
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;
using hipdnn_plugin_sdk::uhd::expression::CategoricalEncoding;
using nlohmann::json;

TEST(TestFeatureExtractor, InlineExpressionsUsePublishedNamesWithoutAQueryPrefix)
{
    const FeatureExtractor extractor(
        {"$attention.input.dims[0]",
         "$kernel.tile_m",
         json::parse(R"({"ceil_div":["$attention.input.dims[0]","$kernel.tile_m"]})")});
    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"attention.input.dims[0]", int64_t{257}}});
    ctx.bindKernelVars({{"tile_m", int64_t{64}}});
    EXPECT_EQ(extractor.extract(ctx), (std::vector<double>{257, 64, 5}));
    ctx.clear();
    ctx.bindQueryVars({{"$attention.input.dims[0]", int64_t{128}}});
    ctx.bindKernelVars({{"tile_m", int64_t{32}}});
    EXPECT_EQ(extractor.extract(ctx), (std::vector<double>{128, 32, 4}));
}

TEST(TestFeatureExtractor, SignatureAndEncodingHaveCrossLanguageFingerprints)
{
    EXPECT_EQ(FeatureExtractor::computeHash({"$q.batch", "$kernel.tile_m", "$device.cu_count"}),
              "sha256:fe9d0487031089e0");
    EXPECT_EQ(FeatureExtractor::computeHash(
                  {"$q.batch", json::parse(R"({"*":["$q.batch","$q.num_heads"]})")}),
              "sha256:d5ae6976facefe74");
    EXPECT_EQ(
        FeatureExtractor::computeHash(
            {"$kernel.dtype"}, CategoricalEncoding{{"$kernel.dtype", {{"fp16", 0}, {"fp32", 1}}}}),
        "sha256:ad7b1aef147b1197");
}

TEST(TestFeatureExtractor, ChangingAnInlineComputationChangesTheContract)
{
    const FeatureExtractor multiply({json::parse(R"({"*":["$q.batch",2]})")});
    const FeatureExtractor add({json::parse(R"({"+":["$q.batch",2]})")});
    EXPECT_NE(multiply.getSignatureHash(), add.getSignatureHash());
    EXPECT_NE(FeatureExtractor::computeHash({"$q.batch", "$kernel.tile_m"}),
              FeatureExtractor::computeHash({"$kernel.tile_m", "$q.batch"}));

    // Same operator, same operands, permuted -- the case a canonicalization that sorted
    // operand arrays would let through while the two expressions compute reciprocals of
    // each other. §6.5's reason for folding the encoding into the hash is this one: what
    // the model consumes must not be able to change while the fingerprint reads the same.
    EXPECT_NE(FeatureExtractor::computeHash({json::parse(R"({"/":["$q.flops","$q.bytes"]})")}),
              FeatureExtractor::computeHash({json::parse(R"({"/":["$q.bytes","$q.flops"]})")}));
}

TEST(TestFeatureExtractor, EncodedContractChangesWhenCodesChange)
{
    const FeatureExtractor first({"$kernel.dtype"}, {{"$kernel.dtype", {{"fp16", 0}}}});
    const FeatureExtractor second({"$kernel.dtype"}, {{"$kernel.dtype", {{"fp16", 9}}}});
    EXPECT_NE(first.getSignatureHash(), second.getSignatureHash());
    EXPECT_EQ(FeatureExtractor::computeHash({"$q.batch"}, {}), "sha256:611513da8e8614b2");
}

TEST(TestFeatureExtractor, QuotedExpressionsAreNotAdmitted)
{
    EXPECT_THROW(FeatureExtractor({"\"$q.batch\""}), JsonLogicError);
    EXPECT_THROW(FeatureExtractor({R"({"+":["$q.batch",1]})"}), JsonLogicError);
}

TEST(TestFeatureExtractor, HashRejectsUnsafeNumericLiteralsInsideExpressions)
{
    EXPECT_THROW(
        FeatureExtractor::computeHash({json::parse(R"({"log2":[{"+":["$q.batch",1e15]}]})")}),
        JsonLogicError);
    const json infinite
        = {{"+", json::array({"$q.batch", std::numeric_limits<double>::infinity()})}};
    EXPECT_THROW(FeatureExtractor::computeHash({infinite}), JsonLogicError);
    const FeatureExtractor bounded({json::parse(R"({"+":["$q.batch",999999999999999.0]})")});
    FeatureExtractionContext ctx;
    ctx.bind("q.batch", 1.0);
    EXPECT_DOUBLE_EQ(bounded.extract(ctx).at(0), 1e15);
}

TEST(TestFeatureExtractor, EmptySignatureNeedsNoBindings)
{
    const FeatureExtractor extractor({});
    const FeatureExtractionContext ctx;
    EXPECT_EQ(extractor.extract(ctx), std::vector<double>{});
}

TEST(TestFeatureExtractor, MissingKernelMetadataNeverLeaksFromPreviousCandidate)
{
    const FeatureExtractor extractor(
        {"$query.batch", json::parse(R"({"value_or_default":["$kernel.tile_m",7]})")});
    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"query.batch", int64_t{8}}});
    auto work = extractor.prepare(ctx);
    ctx.bindKernelVars({{"tile_m", int64_t{64}}});
    extractor.extractKernelInto(ctx, work);
    EXPECT_EQ(work.values, (std::vector<double>{8, 64}));
    ctx.clearKernelVars();
    extractor.extractKernelInto(ctx, work);
    EXPECT_EQ(work.values, (std::vector<double>{8, 7}));
    const FeatureExtractor required({"$kernel.tile_m"});
    EXPECT_THROW(required.extract(ctx), JsonLogicError);
}

TEST(TestFeatureExtractor, SharedErrorsAreRaisedOnlyWhenTheCandidateSelectsTheirBranch)
{
    const FeatureExtractor extractor(
        {json::parse(R"({"if":["$kernel.enabled",{"/":["$query.batch",0]},7]})")});
    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"query.batch", int64_t{8}}});
    auto work = extractor.prepare(ctx);
    ctx.bindKernelVars({{"enabled", false}});
    extractor.extractKernelInto(ctx, work);
    EXPECT_DOUBLE_EQ(work.values.at(0), 7);
    ctx.clearKernelVars();
    ctx.bindKernelVars({{"enabled", true}});
    EXPECT_THROW(extractor.extractKernelInto(ctx, work), JsonLogicError);
}

TEST(TestFeatureExtractor, DefaultDoesNotHideTypeOrArithmeticErrors)
{
    const FeatureExtractor extractor(
        {json::parse(R"({"value_or_default":[{"/":["$query.batch",0]},7]})")});
    FeatureExtractionContext ctx;
    ctx.bind("query.batch", int64_t{2});
    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
    ctx.bind("query.batch", std::string("two"));
    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
}

TEST(TestFeatureExtractor, ShapeAndRankLowerToCanonicalPublishedBindings)
{
    const FeatureExtractor extractor(
        {json::parse(R"({"shape":["$input",2]})"), json::parse(R"({"rank":["$input"]})")});
    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"input.dims[2]", int64_t{128}}, {"input.rank", int64_t{4}}});
    EXPECT_EQ(extractor.extract(ctx), (std::vector<double>{128, 4}));
    ctx.bind("input.dims[2]", std::string("not an extent"));
    EXPECT_THROW(extractor.extract(ctx), JsonLogicError);
    EXPECT_THROW(FeatureExtractor({json::parse(R"({"shape":["$input","$kernel.axis"]})")}),
                 JsonLogicError);
}

TEST(TestFeatureExtractor, CompiledKernelReferencesDriveMetadataCoverage)
{
    const FeatureExtractor extractor(
        {json::parse(R"({"shape":["$kernel.tensor",0]})"), "$node.batch"});
    EXPECT_FALSE(extractor.validateAgainstKmdFields({"tensor"}));
    EXPECT_TRUE(extractor.validateAgainstKmdFields({"tensor.dims[0]"}));
    FeatureExtractionContext ctx;
    ctx.bind("node.batch", int64_t{2});
    EXPECT_EQ(extractor.getMissingVariables(ctx),
              (std::vector<std::string>{"$kernel.tensor.dims[0]"}));
}

TEST(TestFeatureExtractor, WorkspacesCannotBeUsedWithAnotherSignature)
{
    const FeatureExtractor first({"$kernel.a"});
    const FeatureExtractor second({"$kernel.b"});
    FeatureExtractionContext ctx;
    ctx.bindKernelVars({{"a", int64_t{1}}, {"b", int64_t{2}}});
    auto work = first.prepare(ctx);
    EXPECT_THROW(second.extractKernelInto(ctx, work), JsonLogicError);
}

} // namespace
