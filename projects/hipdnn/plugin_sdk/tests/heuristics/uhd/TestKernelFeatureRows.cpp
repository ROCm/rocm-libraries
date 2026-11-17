// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestKernelFeatureRows.cpp
 * @brief The catalog row builder must produce, for every kernel, exactly the row the
 *        per-candidate reference path does: bind kernelVarsFrom(kernel) and extract().
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include <hipdnn_plugin_sdk/ingestor/KernelFeatureRows.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

namespace
{
using hipdnn_plugin_sdk::ingestor::DescriptorId;
using hipdnn_plugin_sdk::ingestor::KernelDefinition;
using hipdnn_plugin_sdk::ingestor::detail::KernelRowBuilder;
using hipdnn_plugin_sdk::ingestor::detail::kernelVarsFrom;
using hipdnn_plugin_sdk::uhd::CategoricalEncoding;
using hipdnn_plugin_sdk::uhd::FeatureExtractionContext;
using hipdnn_plugin_sdk::uhd::FeatureExtractor;
using hipdnn_plugin_sdk::uhd::JsonLogicError;
using nlohmann::json;

DescriptorId idOf(uint32_t value)
{
    DescriptorId id{};
    for(size_t i = 0; i < 4; ++i)
    {
        id[i] = static_cast<uint8_t>(value >> (8 * i));
    }
    return id;
}

/// A double's bits, so equality is exact for NaN payloads and signed zeros (C++17: no
/// std::bit_cast).
uint64_t bitsOf(double value)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

/// Every kind of KMD field the language can read, some of them optional so a candidate can
/// lack one a neighbour has.
KernelDefinition randomKernel(std::mt19937& rng, uint32_t index)
{
    const auto pick = [&](int64_t low, int64_t high) {
        return std::uniform_int_distribution<int64_t>(low, high)(rng);
    };
    const auto chance = [&](double p) { return std::bernoulli_distribution(p)(rng); };

    KernelDefinition kernel;
    kernel.kernelId = idOf(index + 1);
    kernel.priority = pick(-3, 3);
    // Few distinct tile values, so kernels share mixed-entry keys.
    kernel.metadata["tile_m"] = int64_t{16} << pick(0, 3);
    kernel.metadata["tile_n"] = int64_t{16} << pick(0, 3);
    kernel.metadata["zero"] = int64_t{0};
    kernel.metadata["scale"] = 0.5 + (static_cast<double>(pick(0, 700)) / 200.0);
    kernel.metadata["layout"]
        = std::vector<std::string>{"nhwc", "nchw", "chwn"}[static_cast<size_t>(pick(0, 2))];
    kernel.metadata["flag"] = chance(0.5);
    kernel.metadata["tile"] = std::vector<int64_t>{pick(1, 3), pick(1, 3), pick(1, 3)};
    if(chance(0.5))
    {
        kernel.metadata["opt"] = pick(1, 3);
    }
    if(chance(0.5))
    {
        kernel.metadata["optd"] = static_cast<double>(pick(1, 90)) / 7.0;
    }
    if(chance(0.5))
    {
        kernel.metadata["optarr"] = std::vector<int64_t>{pick(1, 3), pick(1, 3)};
    }
    return kernel;
}

FeatureExtractionContext randomProblem(std::mt19937& rng)
{
    const auto pick = [&](int64_t low, int64_t high) {
        return std::uniform_int_distribution<int64_t>(low, high)(rng);
    };
    FeatureExtractionContext ctx;
    ctx.bind("graph.n", pick(1, 4096));
    ctx.bind("graph.h", pick(1, 512));
    ctx.bind("graph.alpha", static_cast<double>(pick(1, 1000)) / 13.0);
    ctx.bind("device.cu_count", pick(32, 304));
    return ctx;
}

const CategoricalEncoding LAYOUT_ENCODING
    = {{"$kernel.layout", {{"nhwc", 0}, {"nchw", 1}, {"chwn", 2}}}};

/// Signature entries by kind: problem/device only, kernel only, and mixed. Each is
/// computable for every kernel the generator makes (optional fields only behind a default).
const std::vector<json>& entryPool()
{
    static const std::vector<json> s_pool = {
        // Problem and device only.
        "$graph.n",
        "$graph.h",
        "$graph.alpha",
        "$device.cu_count",
        json::parse(R"({"*":["$graph.n","$graph.h"]})"),
        // Kernel only: bare references of every type, then expressions.
        "$kernel.tile_m",
        "$kernel.tile_n",
        "$kernel.zero",
        "$kernel.scale",
        "$kernel.flag",
        "$kernel.tile[1]",
        "$kernel.priority",
        "$kernel.layout",
        json::parse(R"({"*":["$kernel.tile_m","$kernel.tile_n"]})"),
        json::parse(R"({"value_or_default":["$kernel.opt",7]})"),
        json::parse(R"({"value_or_default":["$kernel.optd",-1.5]})"),
        json::parse(R"({"value_or_default":["$kernel.optarr[1]",3]})"),
        json::parse(R"({"if":["$kernel.flag","$kernel.priority",{"-":["$kernel.tile_m",1]}]})"),
        json::parse(R"({"/":["$kernel.tile_m","$kernel.scale"]})"),
        // Mixed with problem or device symbols.
        json::parse(R"({"ceil_div":["$graph.n","$kernel.tile_m"]})"),
        json::parse(R"({"ceil_div":["$device.cu_count","$kernel.tile_n"]})"),
        json::parse(R"({"value_or_default":[{"/":["$graph.n","$kernel.zero"]},-1]})"),
        json::parse(R"({"value_or_default":[{"+":["$graph.n","$kernel.opt"]},99]})"),
        json::parse(R"({"if":[{"==":["$kernel.layout","nhwc"]},"$graph.n",)"
                    R"({"*":["$graph.h","$kernel.priority"]}]})"),
        json::parse(R"({"if":["$kernel.flag",{"*":["$graph.alpha","$kernel.scale"]},0]})"),
        json::parse(R"({"+":[{"*":["$graph.h","$kernel.tile[0]"]},"$kernel.tile[2]"]})"),
        json::parse(R"({"value_or_default":[{"-":["$graph.n","$kernel.optarr[1]"]},-7]})"),
        json::parse(R"({"min":["$device.cu_count","$kernel.tile_m"]})"),
    };
    return s_pool;
}

std::vector<json> randomSignature(std::mt19937& rng)
{
    std::vector<json> signature;
    for(const auto& entry : entryPool())
    {
        if(std::bernoulli_distribution(0.5)(rng))
        {
            signature.push_back(entry);
        }
    }
    if(signature.empty())
    {
        signature.push_back(entryPool().front());
    }
    std::shuffle(signature.begin(), signature.end(), rng);
    return signature;
}

/// The row today's per-candidate path produces.
std::vector<double> referenceRow(const FeatureExtractor& extractor,
                                 const FeatureExtractionContext& problem,
                                 const KernelDefinition& kernel)
{
    auto ctx = problem;
    ctx.clearKernelVars();
    ctx.bindKernelVars(kernelVarsFrom(kernel));
    return extractor.extract(ctx);
}

std::vector<const KernelDefinition*> pointersTo(const std::vector<KernelDefinition>& kernels)
{
    std::vector<const KernelDefinition*> pointers;
    pointers.reserve(kernels.size());
    for(const auto& kernel : kernels)
    {
        pointers.push_back(&kernel);
    }
    return pointers;
}

std::vector<double> buildRows(const KernelRowBuilder& builder,
                              const FeatureExtractor& extractor,
                              const FeatureExtractionContext& problem,
                              const std::vector<const KernelDefinition*>& kernels)
{
    auto ctx = problem;
    const auto prefix = extractor.prepare(ctx);
    return builder.build(ctx, prefix, kernels);
}

/// Bit-for-bit, so a NaN, a -0 or a one-ulp difference fails.
void expectBitIdentical(const FeatureExtractor& extractor,
                        const FeatureExtractionContext& problem,
                        const std::vector<const KernelDefinition*>& kernels,
                        const std::vector<double>& rows)
{
    const size_t width = extractor.featureCount();
    ASSERT_EQ(rows.size(), kernels.size() * width);
    for(size_t i = 0; i < kernels.size(); ++i)
    {
        const auto expected = referenceRow(extractor, problem, *kernels[i]);
        ASSERT_EQ(expected.size(), width);
        for(size_t slot = 0; slot < width; ++slot)
        {
            ASSERT_EQ(bitsOf(rows[(i * width) + slot]), bitsOf(expected[slot]))
                << "kernel " << i << " slot " << slot << ": " << rows[(i * width) + slot]
                << " vs reference " << expected[slot];
        }
    }
}

TEST(TestKernelFeatureRows, RandomCatalogsAndSignaturesMatchThePerCandidateReference)
{
    for(uint32_t seed = 0; seed < 60; ++seed)
    {
        SCOPED_TRACE("seed " + std::to_string(seed));
        std::mt19937 rng(seed);
        const auto extractorPtr
            = std::make_shared<const FeatureExtractor>(randomSignature(rng), LAYOUT_ENCODING);
        const auto& extractor = *extractorPtr;
        const auto problem = randomProblem(rng);
        std::vector<KernelDefinition> kernels;
        const auto count = std::uniform_int_distribution<uint32_t>(1, 40)(rng);
        kernels.reserve(count);
        for(uint32_t i = 0; i < count; ++i)
        {
            kernels.push_back(randomKernel(rng, i));
        }
        auto catalog = pointersTo(kernels);

        const KernelRowBuilder builder(extractorPtr);
        expectBitIdentical(
            extractor, problem, catalog, buildRows(builder, extractor, problem, catalog));
        EXPECT_EQ(builder.cachedKernelCount(), kernels.size());

        // A second ranking reads every kernel from the cache, in another order and as a
        // subset, so a candidate's bindings never leak from the one before it.
        std::shuffle(catalog.begin(), catalog.end(), rng);
        catalog.resize((catalog.size() + 1) / 2);
        expectBitIdentical(
            extractor, problem, catalog, buildRows(builder, extractor, problem, catalog));
        EXPECT_EQ(builder.cachedKernelCount(), kernels.size());
    }
}

// The pool signature uses every entry at once, so each kind is covered even if the random
// subsets above never drew a given combination.
TEST(TestKernelFeatureRows, EveryEntryKindTogetherMatchesTheReference)
{
    std::mt19937 rng(7);
    const auto extractorPtr
        = std::make_shared<const FeatureExtractor>(entryPool(), LAYOUT_ENCODING);
    const auto& extractor = *extractorPtr;
    ASSERT_TRUE(extractor.mixedCount() > 0);
    const auto problem = randomProblem(rng);
    std::vector<KernelDefinition> kernels;
    kernels.reserve(64);
    for(uint32_t i = 0; i < 64; ++i)
    {
        kernels.push_back(randomKernel(rng, i));
    }
    const auto catalog = pointersTo(kernels);
    const KernelRowBuilder builder(extractorPtr);
    expectBitIdentical(
        extractor, problem, catalog, buildRows(builder, extractor, problem, catalog));
}

TEST(TestKernelFeatureRows, AFieldTheKernelLacksReadsAsUnboundAndNeverLeaksFromANeighbour)
{
    // n + opt resolves only for kernels that have `opt`; the others take the default.
    const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{
        json::parse(R"({"value_or_default":[{"+":["$graph.n","$kernel.opt"]},99]})")});
    FeatureExtractionContext problem;
    problem.bind("graph.n", int64_t{1000});

    std::vector<KernelDefinition> kernels(5);
    for(uint32_t i = 0; i < kernels.size(); ++i)
    {
        kernels[i].kernelId = idOf(i + 1);
    }
    kernels[0].metadata["opt"] = int64_t{1};
    kernels[2].metadata["opt"] = int64_t{3};
    kernels[3].metadata["opt"] = int64_t{1}; // same as kernel 0: shares its value
    // kernels 1 and 4 lack `opt`.

    const KernelRowBuilder builder(extractor);
    const auto rows = buildRows(builder, *extractor, problem, pointersTo(kernels));
    EXPECT_EQ(rows, (std::vector<double>{1001, 99, 1003, 1001, 99}));
}

TEST(TestKernelFeatureRows, AContextAnEarlierBuildUsedLeaksNoKernelValue)
{
    const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{
        json::parse(R"({"value_or_default":[{"+":["$graph.n","$kernel.opt"]},99]})")});
    FeatureExtractionContext ctx;
    ctx.bind("graph.n", int64_t{1000});
    const auto prefix = extractor->prepare(ctx);

    KernelDefinition withOpt;
    withOpt.kernelId = idOf(1);
    withOpt.metadata["opt"] = int64_t{5};
    KernelDefinition withoutOpt;
    withoutOpt.kernelId = idOf(2);

    const KernelRowBuilder builder(extractor);
    EXPECT_EQ(builder.build(ctx, prefix, {&withOpt}), std::vector<double>{1005});
    // The same context again: the first build left `kernel.opt` bound in it.
    EXPECT_EQ(builder.build(ctx, prefix, {&withoutOpt}), std::vector<double>{99});
}

TEST(TestKernelFeatureRows, AnUnresolvableRowFailsLikeTheReferenceAndCachesNothingForThatKernel)
{
    // A bare reference to a missing field throws; so does a mixed expression with no default.
    for(const json& entry :
        {json("$kernel.tile_m"), json::parse(R"({"ceil_div":["$graph.n","$kernel.tile_m"]})")})
    {
        const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{entry});
        FeatureExtractionContext problem;
        problem.bind("graph.n", int64_t{1000});

        KernelDefinition good;
        good.kernelId = idOf(1);
        good.metadata["tile_m"] = int64_t{64};
        KernelDefinition bad;
        bad.kernelId = idOf(2);

        EXPECT_THROW(referenceRow(*extractor, problem, bad), JsonLogicError);
        const KernelRowBuilder builder(extractor);
        EXPECT_THROW(buildRows(builder, *extractor, problem, {&good, &bad}), JsonLogicError);

        // The failure left the cache usable for the kernels that do resolve.
        EXPECT_NO_THROW(buildRows(builder, *extractor, problem, {&good}));
        EXPECT_THROW(buildRows(builder, *extractor, problem, {&good, &bad}), JsonLogicError);
        EXPECT_EQ(builder.cachedKernelCount(), entry.is_string() ? 1u : 2u);
    }
}

TEST(TestKernelFeatureRows, ADefinitionReusedUnderAnotherIdIsNotServedTheOldValues)
{
    const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{
        "$kernel.tile_m", json::parse(R"({"ceil_div":["$graph.n","$kernel.tile_m"]})")});
    FeatureExtractionContext problem;
    problem.bind("graph.n", int64_t{1000});
    const KernelRowBuilder builder(extractor);

    // The same object re-purposed as a different kernel: its id changed, so it is a new key.
    KernelDefinition slot;
    slot.kernelId = idOf(1);
    slot.metadata["tile_m"] = int64_t{64};
    EXPECT_EQ(buildRows(builder, *extractor, problem, {&slot}), (std::vector<double>{64, 16}));

    slot.kernelId = idOf(2);
    slot.metadata["tile_m"] = int64_t{100};
    EXPECT_EQ(buildRows(builder, *extractor, problem, {&slot}), (std::vector<double>{100, 10}));
    EXPECT_EQ(builder.cachedKernelCount(), 2u);
}

TEST(TestKernelFeatureRows, TwoDefinitionsSharingAnIdKeepTheirOwnValues)
{
    // One id may name differently-parameterized definitions on disjoint arches, and one
    // heuristic can rank catalogs of both; each must read its own metadata and priority.
    const auto extractor = std::make_shared<const FeatureExtractor>(
        std::vector<json>{"$kernel.tile_m",
                          "$kernel.priority",
                          json::parse(R"({"ceil_div":["$graph.n","$kernel.tile_m"]})")});
    FeatureExtractionContext problem;
    problem.bind("graph.n", int64_t{1000});
    const KernelRowBuilder builder(extractor);

    KernelDefinition first;
    first.kernelId = idOf(1);
    first.priority = 3;
    first.metadata["tile_m"] = int64_t{64};
    KernelDefinition second = first;
    second.priority = 8;
    second.metadata["tile_m"] = int64_t{100};

    EXPECT_EQ(buildRows(builder, *extractor, problem, {&first}), (std::vector<double>{64, 3, 16}));
    EXPECT_EQ(buildRows(builder, *extractor, problem, {&second}),
              (std::vector<double>{100, 8, 10}));
    EXPECT_EQ(buildRows(builder, *extractor, problem, {&first, &second}),
              (std::vector<double>{64, 3, 16, 100, 8, 10}));
    EXPECT_EQ(builder.cachedKernelCount(), 2u);
}

TEST(TestKernelFeatureRows, SignatureWithoutKernelEntriesYieldsTheSharedRowForEveryKernel)
{
    const auto extractor = std::make_shared<const FeatureExtractor>(
        std::vector<json>{"$graph.n", json::parse(R"({"*":["$graph.n",2]})")});
    FeatureExtractionContext problem;
    problem.bind("graph.n", int64_t{21});
    std::vector<KernelDefinition> kernels(3);
    for(uint32_t i = 0; i < kernels.size(); ++i)
    {
        kernels[i].kernelId = idOf(i + 1);
    }
    const KernelRowBuilder builder(extractor);
    EXPECT_EQ(buildRows(builder, *extractor, problem, pointersTo(kernels)),
              (std::vector<double>{21, 42, 21, 42, 21, 42}));
}

TEST(TestKernelFeatureRows, EmptyCatalogAndEmptySignatureYieldNoRows)
{
    const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{});
    const FeatureExtractionContext problem;
    const KernelRowBuilder builder(extractor);
    KernelDefinition kernel;
    kernel.kernelId = idOf(1);
    EXPECT_TRUE(buildRows(builder, *extractor, problem, {}).empty());
    EXPECT_TRUE(buildRows(builder, *extractor, problem, {&kernel}).empty());
}

TEST(TestKernelFeatureRows, WorkspaceOfAnotherSignatureIsRefused)
{
    const auto extractor = std::make_shared<const FeatureExtractor>(std::vector<json>{"$graph.n"});
    const FeatureExtractor other({"$graph.n"});
    FeatureExtractionContext problem;
    problem.bind("graph.n", int64_t{1});
    const auto foreign = other.prepare(problem);
    const KernelRowBuilder builder(extractor);
    EXPECT_THROW(builder.build(problem, foreign, {}), JsonLogicError);
}

TEST(TestKernelFeatureRows, ConcurrentRankingsOnOneBuilderMatchTheSerialReference)
{
    std::mt19937 rng(11);
    const auto extractor = std::make_shared<const FeatureExtractor>(entryPool(), LAYOUT_ENCODING);
    const auto problem = randomProblem(rng);
    std::vector<KernelDefinition> kernels;
    kernels.reserve(96);
    for(uint32_t i = 0; i < 96; ++i)
    {
        kernels.push_back(randomKernel(rng, i));
    }
    std::vector<std::vector<double>> expected;
    expected.reserve(kernels.size());
    for(const auto& kernel : kernels)
    {
        expected.push_back(referenceRow(*extractor, problem, kernel));
    }

    // A fresh builder per round so the threads race on the first fill, not only on reads.
    for(int round = 0; round < 8; ++round)
    {
        const KernelRowBuilder builder(extractor);
        std::atomic<int> mismatches{0};
        std::vector<std::thread> threads;
        threads.reserve(6);
        for(unsigned t = 0; t < 6; ++t)
        {
            threads.emplace_back([&, t] {
                std::mt19937 local(1000 + t);
                for(int repeat = 0; repeat < 4; ++repeat)
                {
                    std::vector<size_t> order(kernels.size());
                    for(size_t i = 0; i < order.size(); ++i)
                    {
                        order[i] = i;
                    }
                    std::shuffle(order.begin(), order.end(), local);
                    order.resize(48 + (t * 8));
                    std::vector<const KernelDefinition*> catalog;
                    catalog.reserve(order.size());
                    for(const auto index : order)
                    {
                        catalog.push_back(&kernels[index]);
                    }
                    const auto rows = buildRows(builder, *extractor, problem, catalog);
                    const size_t width = extractor->featureCount();
                    for(size_t i = 0; i < order.size(); ++i)
                    {
                        if(!std::equal(rows.begin() + static_cast<std::ptrdiff_t>(i * width),
                                       rows.begin() + static_cast<std::ptrdiff_t>((i + 1) * width),
                                       expected[order[i]].begin()))
                        {
                            ++mismatches;
                        }
                    }
                }
            });
        }
        for(auto& thread : threads)
        {
            thread.join();
        }
        EXPECT_EQ(mismatches.load(), 0);
    }
}

} // namespace

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
