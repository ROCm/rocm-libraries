// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{
using hipdnn_plugin_sdk::uhd::CategoricalEncoding;
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;
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

    // Permuted operands compute reciprocals, so the hash must not sort operand arrays
    // (RFC 0019 §6.5).
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

TEST(TestFeatureExtractor, KernelOnlyEntriesAreCachedPerKernelAndMixedOnesRunPerCandidate)
{
    const FeatureExtractor extractor(
        {"$query.batch",
         "$kernel.tile_m",
         json::parse(R"({"*":["$kernel.tile_m","$kernel.priority"]})"),
         json::parse(R"({"ceil_div":["$query.batch","$kernel.tile_m"]})"),
         json::parse(R"({"+":["$kernel.tile[1]","$query.batch"]})")});
    EXPECT_EQ(extractor.kernelDependentCount(), 4u);
    EXPECT_EQ(extractor.mixedCount(), 2u);
    // Only what mixed entries read is bound per candidate; kernel-only reads are cached.
    EXPECT_EQ(extractor.mixedKernelVariables(),
              (std::vector<std::string>{"kernel.tile_m", "kernel.tile[1]"}));

    FeatureExtractionContext ctx;
    ctx.bindQueryVars({{"query.batch", int64_t{100}}});
    ctx.bindKernelVars(
        {{"tile_m", int64_t{32}}, {"priority", int64_t{3}}, {"tile[1]", int64_t{5}}});
    const auto reference = extractor.extract(ctx);

    FeatureExtractionContext kernelOnly;
    kernelOnly.bindKernelVars(
        {{"tile_m", int64_t{32}}, {"priority", int64_t{3}}, {"tile[1]", int64_t{5}}});
    const auto cached = extractor.kernelOnlyValues(kernelOnly);
    EXPECT_EQ(cached, (std::vector<double>{32, 96}));

    auto work = extractor.prepare(ctx);
    auto row = work.values;
    extractor.applyKernelOnly(row.data(), cached);
    for(size_t j = 0; j < extractor.mixedCount(); ++j)
    {
        row[extractor.mixedFeatureIndex(j)] = extractor.evaluateMixed(j, ctx);
    }
    EXPECT_EQ(row, reference);
    EXPECT_EQ(row, (std::vector<double>{100, 32, 96, 4, 105}));
}

/// The graph and device features every engine shares are bound once, in a base the engine's own
/// context reads through; the signature's references and expressions must see them there.
std::shared_ptr<const FeatureExtractionContext> sharedGraphFeatures()
{
    auto shared = std::make_shared<FeatureExtractionContext>();
    shared->bind("graph.flops", 10.0);
    shared->bind("graph.nodes", int64_t{3});
    shared->bind("kernel.dtype", std::string("fp16"));
    return shared;
}

TEST(TestFeatureExtractor, ReferencesAndExpressionsReadTheSharedBase)
{
    FeatureExtractionContext own(sharedGraphFeatures());
    own.bind("constraint.tile", int64_t{4});

    const FeatureExtractor extractor({json::parse(R"({"*":["$graph.flops",2]})"),
                                      "$graph.flops",
                                      "$constraint.tile",
                                      json::parse(R"({"+":["$graph.nodes","$constraint.tile"]})")});
    EXPECT_EQ(extractor.extract(own), (std::vector<double>{20, 10, 4, 7}));
}

TEST(TestFeatureExtractor, AnOwnBindingShadowsTheSharedBaseOnEveryRead)
{
    const auto shared = sharedGraphFeatures();
    FeatureExtractionContext own(shared);
    own.bind("graph.flops", 99.0);

    const FeatureExtractor extractor({json::parse(R"({"*":["$graph.flops",2]})"), "$graph.flops"});
    EXPECT_EQ(extractor.extract(own), (std::vector<double>{198, 99}));
    EXPECT_EQ(std::get<double>(*own.getContext().find("graph.flops")), 99.0);
    EXPECT_EQ(std::get<double>(*own.getContext().find("$graph.flops")), 99.0);

    // A shadowed name is one name: visited once, with the value that reads.
    std::map<std::string, int> visits;
    own.getContext().forEachBinding(
        [&visits](const std::string& name, const auto& /*value*/) { ++visits[name]; });
    EXPECT_EQ(
        visits,
        (std::map<std::string, int>{{"graph.flops", 1}, {"graph.nodes", 1}, {"kernel.dtype", 1}}));
    const auto exported = own.toJson();
    EXPECT_EQ(exported.size(), 3u);
    EXPECT_EQ(exported.at("graph.flops"), 99.0);
    EXPECT_EQ(exported.at("graph.nodes"), 3);

    // The base is shared and never written through.
    EXPECT_EQ(std::get<double>(*shared->getContext().find("graph.flops")), 10.0);

    // Removing the shadow uncovers the base's value; the base's own name is not removable.
    own.unbindSlot("graph.flops");
    EXPECT_EQ(extractor.extract(own), (std::vector<double>{20, 10}));
    own.unbindSlot("graph.nodes");
    EXPECT_TRUE(own.getContext().has("$graph.nodes"));
}

TEST(TestFeatureExtractor, ClearingOwnBindingsLeavesTheSharedBaseReadable)
{
    const auto shared = sharedGraphFeatures();
    FeatureExtractionContext own(shared);
    own.bindKernelVars({{"dtype", std::string("fp32")}, {"tile_m", int64_t{64}}});
    own.bind("constraint.tile", int64_t{4});
    ASSERT_EQ(std::get<std::string>(*own.getContext().find("kernel.dtype")), "fp32");

    // The namespace's own bindings go, including the one that shadowed the base's.
    own.clearKernelVars();
    EXPECT_FALSE(own.getContext().has("kernel.tile_m"));
    ASSERT_TRUE(own.getContext().has("kernel.dtype"));
    EXPECT_EQ(std::get<std::string>(*own.getContext().find("kernel.dtype")), "fp16");
    EXPECT_TRUE(own.getContext().has("graph.flops"));
    EXPECT_TRUE(own.getContext().has("constraint.tile"));

    // clear() leaves nothing readable, and the base itself is untouched.
    own.clear();
    EXPECT_FALSE(own.getContext().has("graph.flops"));
    EXPECT_FALSE(own.getContext().has("constraint.tile"));
    EXPECT_TRUE(own.toJson().empty());
    EXPECT_EQ(shared->toJson().size(), 3u);
}

} // namespace
