// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file BenchUhdRanking.cpp
 * @brief Disabled microbenchmark of UHD kernel ranking at catalog scale (RFC 0019 §9.4).
 *        Run with --gtest_also_run_disabled_tests --gtest_filter='*UhdRankingBenchmark*'.
 *        Sizes come from UHD_BENCH_{TREES,LEAVES,KERNELS,KFEATS,PFEATS,EXTRA,REPS}.
 */

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <gtest/gtest.h>

#include "../KernelIngestorTestFixtures.hpp"

#include <hipdnn_plugin_sdk/heuristics/uhd/FeatureExtractor.hpp>
#include <hipdnn_plugin_sdk/heuristics/uhd/adapters/TreeDataAdapter.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelHeuristicFactory.hpp>
#include <hipdnn_plugin_sdk/ingestor/UhdKernelHeuristic.hpp>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>

#include <hipdnn_test_sdk/utilities/FileUtilities.hpp>
#include <hipdnn_test_sdk/utilities/GbdtModelTestBuilder.hpp>
#include <hipdnn_test_sdk/utilities/LogRecorder.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace hipdnn_plugin_sdk::ingestor
{
namespace
{

using Builder = hipdnn_test_sdk::utilities::GbdtModelTestBuilder;
using Clock = std::chrono::steady_clock;

size_t envSize(const char* name, size_t fallback)
{
    const std::string value = hipdnn_data_sdk::utilities::getEnv(name);
    return value.empty() ? fallback
                         : static_cast<size_t>(std::strtoull(value.c_str(), nullptr, 10));
}

/// A LightGBM-shaped tree: grow by splitting a random leaf until @p leaves leaves.
Builder::TreeSpec randomTree(std::mt19937_64& rng, size_t leaves, int32_t features)
{
    Builder::TreeSpec tree;
    std::vector<int32_t> openLeaves = {0};
    tree.featureIndices = {0};
    tree.thresholds = {0.0};
    tree.leftChildren = {-1};
    tree.rightChildren = {-1};
    tree.leafValues = {0.0};
    tree.defaultLeft = {1};
    std::uniform_int_distribution<int32_t> feature(0, features - 1);
    std::uniform_real_distribution<double> threshold(0.0, 256.0);
    std::uniform_real_distribution<double> leaf(-0.1, 0.1);
    while(openLeaves.size() < leaves)
    {
        const size_t pick = std::uniform_int_distribution<size_t>(0, openLeaves.size() - 1)(rng);
        const int32_t node = openLeaves[pick];
        const auto left = static_cast<int32_t>(tree.featureIndices.size());
        for(int c = 0; c < 2; ++c)
        {
            tree.featureIndices.push_back(0);
            tree.thresholds.push_back(0.0);
            tree.leftChildren.push_back(-1);
            tree.rightChildren.push_back(-1);
            tree.leafValues.push_back(0.0);
            tree.defaultLeft.push_back(1);
        }
        tree.featureIndices[static_cast<size_t>(node)] = feature(rng);
        tree.thresholds[static_cast<size_t>(node)] = threshold(rng);
        tree.leftChildren[static_cast<size_t>(node)] = left;
        tree.rightChildren[static_cast<size_t>(node)] = left + 1;
        tree.defaultLeft[static_cast<size_t>(node)] = static_cast<uint8_t>(rng() & 1U);
        openLeaves[pick] = left;
        openLeaves.push_back(left + 1);
    }
    for(const auto open : openLeaves)
    {
        tree.leafValues[static_cast<size_t>(open)] = leaf(rng);
    }
    return tree;
}

struct Stats
{
    double minUs;
    double medianUs;
};

template <typename F>
Stats timeIt(size_t reps, F&& body)
{
    std::vector<double> samples;
    samples.reserve(reps);
    for(size_t r = 0; r < reps; ++r)
    {
        const auto start = Clock::now();
        body();
        samples.push_back(std::chrono::duration<double, std::micro>(Clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    return {samples.front(), samples[samples.size() / 2]};
}

} // namespace

TEST(TestIngestorUhdRankingBenchmark, DISABLED_CatalogScale)
{
    const size_t trees = envSize("UHD_BENCH_TREES", 500);
    const size_t leaves = envSize("UHD_BENCH_LEAVES", 63);
    const size_t kernels = envSize("UHD_BENCH_KERNELS", 800);
    const size_t kernelFeatures = envSize("UHD_BENCH_KFEATS", 20);
    const size_t problemFeatures = envSize("UHD_BENCH_PFEATS", 8);
    const size_t extraFields = envSize("UHD_BENCH_EXTRA", 15);
    const size_t reps = envSize("UHD_BENCH_REPS", 30);
    std::mt19937_64 rng(20261029);

    // Signature: kernel fields, a 3-element array field, problem tokens, two expressions.
    std::vector<nlohmann::json> signature;
    std::vector<std::string> knobs;
    std::unordered_set<std::string> fields;
    for(size_t i = 0; i < kernelFeatures; ++i)
    {
        signature.emplace_back("$kernel.k" + std::to_string(i));
        knobs.push_back("k" + std::to_string(i));
        fields.insert("k" + std::to_string(i));
    }
    for(size_t i = 0; i < 3; ++i)
    {
        signature.emplace_back("$kernel.tile[" + std::to_string(i) + "]");
    }
    knobs.emplace_back("tile");
    fields.insert("tile");
    for(size_t i = 0; i < problemFeatures; ++i)
    {
        signature.emplace_back("$bench.p" + std::to_string(i));
    }
    signature.push_back(nlohmann::json::parse(R"({"ceil_div":["$bench.p0","$kernel.k0"]})"));
    signature.push_back(nlohmann::json::parse(R"({"*":["$kernel.k1","$kernel.tile[0]"]})"));
    const auto featureCount = static_cast<int32_t>(signature.size());

    Builder model;
    model.setFeaturesHash(uhd::FeatureExtractor::computeHash(signature))
        .setNumFeatures(featureCount)
        .setBaseScore(1.0)
        .setTrainingArches({"gfx942"});
    for(size_t t = 0; t < trees; ++t)
    {
        model.addTree(randomTree(rng, leaves, featureCount));
    }
    const hipdnn_test_sdk::utilities::ScopedDirectory dir(
        std::filesystem::temp_directory_path()
        / ("uhd_bench_" + std::to_string(Clock::now().time_since_epoch().count())));
    const auto modelPath = dir.path() / "model.bin";
    ASSERT_TRUE(model.buildToFile(modelPath.string()));

    // Kernels: every referenced field plus unread metadata, as a real KMD carries.
    std::uniform_int_distribution<int64_t> value(1, 255);
    std::vector<KernelDefinition> definitions(kernels);
    for(size_t k = 0; k < kernels; ++k)
    {
        auto& kernel = definitions[k];
        kernel.kernelId.fill(0);
        kernel.kernelId[0] = static_cast<uint8_t>(k & 0xFFU);
        kernel.kernelId[1] = static_cast<uint8_t>((k >> 8) & 0xFFU);
        kernel.priority = static_cast<int64_t>(k % 7);
        for(size_t i = 0; i < kernelFeatures; ++i)
        {
            kernel.metadata["k" + std::to_string(i)] = value(rng);
        }
        kernel.metadata["tile"] = std::vector<int64_t>{value(rng), value(rng), value(rng)};
        for(size_t i = 0; i < extraFields; ++i)
        {
            kernel.metadata["unread_field_" + std::to_string(i)] = value(rng);
        }
    }
    Catalog catalog;
    for(const auto& definition : definitions)
    {
        catalog.entries.push_back(&definition);
    }
    for(size_t i = 0; i < problemFeatures; ++i)
    {
        catalog.bound["bench.p" + std::to_string(i)] = value(rng);
    }

    HeuristicDescriptor descriptor;
    descriptor.id.fill(0);
    descriptor.id[0] = 0xEE;
    descriptor.name = "bench";
    descriptor.adapter = UhdAdapter::TREE_DATA;
    descriptor.featuresSignature = signature;
    descriptor.featuresHash = uhd::FeatureExtractor::computeHash(signature);
    descriptor.objective = "max";
    descriptor.score.metric = "tflops";
    descriptor.score.calibrated = false;
    descriptor.score.transform = "identity";
    descriptor.modelArtifactPath = "model.bin";
    descriptor.baseDir = dir.path();

    const testing::TestGraph graph;
    auto properties = testing::testDeviceProperties();
    properties.gcnArchName = "gfx942";
    const MatchContext context{graph, 0, properties};

    // Load: parse, verify and flatten the artifact.
    std::vector<uint8_t> bytes(std::filesystem::file_size(modelPath));
    std::ifstream(modelPath, std::ios::binary)
        .read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    const auto load = timeIt(std::max<size_t>(reps / 3, 3), [&] {
        auto adapter = uhd::TreeDataAdapter::loadFromBuffer(
            bytes.data(), bytes.size(), descriptor.featuresHash);
        ASSERT_NE(adapter, nullptr);
    });
    const auto adapter
        = uhd::TreeDataAdapter::loadFromBuffer(bytes.data(), bytes.size(), descriptor.featuresHash);
    ASSERT_NE(adapter, nullptr);

    const auto heuristic = makeKernelHeuristic(descriptor, {}, knobs, fields);
    ASSERT_NE(heuristic, nullptr);
    {
        // A resolver's traceDecidedBy() is context-free; the per-ranking trace is authoritative.
        const auto recorder
            = hipdnn_test_sdk::utilities::SharedLogRecorder::withOverrideLevel(HIPDNN_SEV_INFO);
        ASSERT_EQ(heuristic->rankScored(catalog, context).size(), kernels);
        ASSERT_TRUE(recorder.hasLogContaining("decided_by=model"))
            << recorder.getRecordedLogsAsString();
    }

    // Full ranking, then its halves: feature rows (binding each kernel's whole metadata, the
    // pre-slot path) and tree scoring alone.
    std::vector<ScoredKernel> ranked;
    const auto rank = timeIt(reps, [&] { ranked = heuristic->rankScored(catalog, context); });
    ASSERT_EQ(ranked.size(), kernels);

    const uhd::FeatureExtractor extractor(signature);
    std::vector<std::vector<double>> rows;
    const auto extract = timeIt(reps, [&] {
        auto ctx = detail::catalogProblemFeatures(context, catalog.bound);
        auto work = extractor.prepare(ctx);
        rows.clear();
        for(const auto* entry : catalog.entries)
        {
            ctx.clearKernelVars();
            ctx.bindKernelVars(detail::kernelVarsFrom(*entry));
            extractor.extractKernelInto(ctx, work);
            rows.push_back(work.values);
        }
    });
    std::vector<double> raw;
    const auto score = timeIt(reps, [&] { raw = adapter->scoreBatch(rows); });

    // A checksum of the order and scores, so a perf change is shown not to change results.
    uint64_t digest = 1469598103934665603ULL;
    for(const auto& candidate : ranked)
    {
        for(const auto byte : candidate.kernelId)
        {
            digest = (digest ^ byte) * 1099511628211ULL;
        }
        uint64_t bits = 0;
        static_assert(sizeof(bits) == sizeof(candidate.score));
        std::memcpy(&bits, &candidate.score, sizeof(bits));
        digest = (digest ^ bits) * 1099511628211ULL;
    }

    std::printf("uhd_bench trees=%zu leaves=%zu kernels=%zu features=%d extra=%zu reps=%zu\n",
                trees,
                leaves,
                kernels,
                featureCount,
                extraFields,
                reps);
    std::printf("uhd_bench model_bytes=%zu load_us min=%.1f med=%.1f\n",
                bytes.size(),
                load.minUs,
                load.medianUs);
    std::printf("uhd_bench rank_us min=%.1f med=%.1f\n", rank.minUs, rank.medianUs);
    std::printf(
        "uhd_bench extract_wholebind_us min=%.1f med=%.1f\n", extract.minUs, extract.medianUs);
    std::printf("uhd_bench score_us min=%.1f med=%.1f\n", score.minUs, score.medianUs);
    std::printf("uhd_bench digest=%016llx top=%.17g\n",
                static_cast<unsigned long long>(digest),
                ranked.front().score);
}

} // namespace hipdnn_plugin_sdk::ingestor

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
