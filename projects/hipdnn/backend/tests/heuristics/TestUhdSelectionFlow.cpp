// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdSelectionFlow.cpp
 * @brief Integration tests for the complete UHD selection flow.
 *
 * Tests the selection flow from RFC §6:
 * 1. Register engines with UHD configs and kernel candidates
 * 2. Extract features from device/kernel/query context
 * 3. Score candidates using adapters
 * 4. Sort by objective with tie-breaking
 * 5. Verify correct winner selection
 */

#include "heuristics/uhd/EngineRegistry.hpp"
#include "heuristics/uhd/SelectionEngine.hpp"

#include <gtest/gtest.h>

#include <cmath>

using hipdnn_backend::heuristics::uhd::EngineEntry;
using hipdnn_backend::heuristics::uhd::EngineRegistry;
using hipdnn_backend::heuristics::uhd::FeatureExtractionContext;
using hipdnn_backend::heuristics::uhd::KernelCandidate;
using hipdnn_backend::heuristics::uhd::SelectionEngine;
using hipdnn_backend::heuristics::uhd::SelectionResult;
using hipdnn_backend::heuristics::uhd::UhdConfig;

namespace ScoreXform = hipdnn_backend::heuristics::uhd::ScoreTransform;

namespace
{

// Helper to create a KernelCandidate
KernelCandidate makeCandidate(int64_t kernelId,
                              int64_t priority,
                              const std::unordered_map<std::string, double>& metadata = {})
{
    KernelCandidate k;
    k.kernelId = kernelId;
    k.priority = priority;
    k.metadata = metadata;
    return k;
}

class TestUhdSelectionFlow : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear registry before each test
        EngineRegistry::instance().clear();
    }

    void TearDown() override { EngineRegistry::instance().clear(); }

    // Helper to create a simple engine with static_order adapter
    static EngineEntry createStaticOrderEngine(int64_t engineId,
                                               const std::vector<KernelCandidate>& candidates)
    {
        EngineEntry entry;
        entry.engineId = engineId;
        entry.engineName = "TestEngine_" + std::to_string(engineId);
        entry.uhdConfig.uhdId = "uhd_" + std::to_string(engineId);
        entry.uhdConfig.adapterType = "static_order";
        entry.uhdConfig.featuresSignature = {"\"$kernel.priority\"", "\"$kernel.id\""};
        entry.uhdConfig.staticOrderFields = {"priority", "id"};
        entry.uhdConfig.objective = "max";
        entry.candidates = candidates;
        return entry;
    }

    // Helper to create device vars
    static FeatureExtractionContext::ValueMap defaultDeviceVars()
    {
        return {
            {"cu_count", int64_t{120}},
            {"multi_processor_count", int64_t{120}},
            {"total_global_mem", int64_t{32LL * 1024 * 1024 * 1024}},
            {"device_id", int64_t{0}},
        };
    }

    // Helper to create query vars
    static FeatureExtractionContext::ValueMap defaultQueryVars()
    {
        return {
            {"batch", 32.0},
            {"seqlen_q", 512.0},
            {"seqlen_k", 512.0},
            {"num_heads", 32.0},
            {"head_dim", 128.0},
        };
    }
};

// ========== Basic Selection Tests ==========

TEST_F(TestUhdSelectionFlow, SelectWithNoEnginesReturnsNotApplied)
{
    // No engines registered
    auto result = SelectionEngine::select(999, defaultDeviceVars(), defaultQueryVars());

    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.sortedKernelIds.empty());
    EXPECT_EQ(result.fallbackReason, "engine not found in registry");
}

TEST_F(TestUhdSelectionFlow, SelectWithEmptyCandidatesReturnsApplied)
{
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.adapterType = "static_order";
    entry.candidates = {}; // Empty

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.sortedKernelIds.empty());
}

TEST_F(TestUhdSelectionFlow, SelectSingleCandidateReturnsIt)
{
    auto k1 = makeCandidate(1, 0, {{"tile_m", 128.0}, {"split_k", 1.0}});

    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 1u);
    EXPECT_EQ(result.sortedKernelIds[0], 1);
    EXPECT_EQ(result.bestKernelId, 1);
}

// ========== Priority-Based Selection (Static Order) ==========

TEST_F(TestUhdSelectionFlow, StaticOrderSelectsByPriorityFirst)
{
    // Lower priority value = higher priority (selected first)
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);
    auto k3 = makeCandidate(3, 15);

    auto entry = createStaticOrderEngine(100, {k1, k2, k3});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 3u);
    // Sorted by priority (ascending): k2(5) < k1(10) < k3(15)
    // But static_order adapter uses negate, so lower priority gets higher score
    // With objective=max, higher score wins → lowest priority first
    EXPECT_EQ(result.sortedKernelIds[0], 2); // priority 5
    EXPECT_EQ(result.sortedKernelIds[1], 1); // priority 10
    EXPECT_EQ(result.sortedKernelIds[2], 3); // priority 15
}

TEST_F(TestUhdSelectionFlow, StaticOrderTieBreaksByIdWhenPriorityEqual)
{
    auto k1 = makeCandidate(100, 5);
    auto k2 = makeCandidate(50, 5);
    auto k3 = makeCandidate(200, 5);

    auto entry = createStaticOrderEngine(100, {k1, k2, k3});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 3u);
    // Same priority, tie-break by id (ascending)
    EXPECT_EQ(result.sortedKernelIds[0], 50);
    EXPECT_EQ(result.sortedKernelIds[1], 100);
    EXPECT_EQ(result.sortedKernelIds[2], 200);
}

// ========== Objective Tests ==========

TEST_F(TestUhdSelectionFlow, MinObjectiveSelectsLowestScore)
{
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.priority\"", "\"$kernel.id\""};
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.uhdConfig.objective = "min"; // Changed to min
    entry.candidates = {k1, k2};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    // With min objective, higher priority value (lower score from negation) wins
    EXPECT_EQ(result.sortedKernelIds[0], 1); // priority 10 → score -10 (lowest)
    EXPECT_EQ(result.sortedKernelIds[1], 2); // priority 5 → score -5
}

// ========== Score Transform Tests ==========

TEST_F(TestUhdSelectionFlow, ScoreTransformLog1pInverseApplied)
{
    // log1p(x) inverse is expm1(x)
    const double rawScore = 2.0;
    const double transformed = ScoreXform::applyInverse(rawScore, "log1p");
    EXPECT_NEAR(transformed, std::expm1(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformLogInverseApplied)
{
    const double rawScore = 2.0;
    const double transformed = ScoreXform::applyInverse(rawScore, "log");
    EXPECT_NEAR(transformed, std::exp(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformIdentityPassthrough)
{
    const double rawScore = 42.0;
    EXPECT_EQ(ScoreXform::applyInverse(rawScore, "identity"), 42.0);
    EXPECT_EQ(ScoreXform::applyInverse(rawScore, ""), 42.0);
    EXPECT_EQ(ScoreXform::applyInverse(rawScore, "unknown"), 42.0);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformRoundTrip)
{
    const double original = 100.0;

    // log1p round-trip
    const double log1pForward = ScoreXform::applyForward(original, "log1p");
    const double log1pBack = ScoreXform::applyInverse(log1pForward, "log1p");
    EXPECT_NEAR(log1pBack, original, 1e-10);

    // sqrt round-trip
    const double sqrtForward = ScoreXform::applyForward(original, "sqrt");
    const double sqrtBack = ScoreXform::applyInverse(sqrtForward, "sqrt");
    EXPECT_NEAR(sqrtBack, original, 1e-10);
}

// ========== Registry Tests ==========

TEST_F(TestUhdSelectionFlow, RegistryStoresAndRetrievesEngines)
{
    auto entry = createStaticOrderEngine(100, {});
    EngineRegistry::instance().registerEngine(entry);

    EXPECT_TRUE(EngineRegistry::instance().hasEngine(100));
    EXPECT_FALSE(EngineRegistry::instance().hasEngine(999));

    auto retrieved = EngineRegistry::instance().getEngine(100);
    ASSERT_TRUE(retrieved.has_value());
    EXPECT_EQ(retrieved->get().engineId, 100);
}

TEST_F(TestUhdSelectionFlow, RegistryClearRemovesAllEngines)
{
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {}));
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(200, {}));

    EXPECT_EQ(EngineRegistry::instance().size(), 2u);

    EngineRegistry::instance().clear();

    EXPECT_EQ(EngineRegistry::instance().size(), 0u);
    EXPECT_FALSE(EngineRegistry::instance().hasEngine(100));
}

// ========== Adapter Creation Tests ==========

TEST_F(TestUhdSelectionFlow, GetOrCreateAdapterCreatesStaticOrderAdapter)
{
    auto entry = createStaticOrderEngine(100, {});
    EngineRegistry::instance().registerEngine(entry);

    auto adapter = EngineRegistry::instance().getOrCreateAdapter(100);
    ASSERT_NE(adapter, nullptr);
    EXPECT_EQ(adapter->type(), hipdnn_backend::heuristics::uhd::UhdAdapterType::STATIC_ORDER);
}

TEST_F(TestUhdSelectionFlow, GetOrCreateAdapterCachesAdapter)
{
    auto entry = createStaticOrderEngine(100, {});
    EngineRegistry::instance().registerEngine(entry);

    auto adapter1 = EngineRegistry::instance().getOrCreateAdapter(100);
    auto adapter2 = EngineRegistry::instance().getOrCreateAdapter(100);

    EXPECT_EQ(adapter1.get(), adapter2.get()); // Same instance
}

// ========== Feature Extraction Integration ==========

TEST_F(TestUhdSelectionFlow, SelectionUsesKernelMetadata)
{
    // Create kernels with different tile_m values
    // Features signature includes tile_m, so scores should differ
    auto k1 = makeCandidate(1, 0, {{"tile_m", 64.0}});
    auto k2 = makeCandidate(2, 0, {{"tile_m", 128.0}});
    auto k3 = makeCandidate(3, 0, {{"tile_m", 256.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.adapterType = "static_order";
    // Include tile_m in signature so it affects scoring
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\"", "\"$kernel.id\""};
    entry.uhdConfig.staticOrderFields = {"tile_m", "id"};
    entry.uhdConfig.objective = "max";
    entry.candidates = {k1, k2, k3};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 3u);
    // Lower tile_m gets higher score (negated), so 64 wins with max objective
    EXPECT_EQ(result.sortedKernelIds[0], 1); // tile_m=64
    EXPECT_EQ(result.sortedKernelIds[1], 2); // tile_m=128
    EXPECT_EQ(result.sortedKernelIds[2], 3); // tile_m=256
}

// ========== Score-Only Mode ==========

TEST_F(TestUhdSelectionFlow, ScoreOnlyReturnsBestScore)
{
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    auto entry = createStaticOrderEngine(100, {k1, k2});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::scoreOnly(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.bestScore.has_value());
    EXPECT_TRUE(result.bestKernelId.has_value());
    EXPECT_EQ(*result.bestKernelId, 2); // Best by priority
}

// ========== Multiple Engines ==========

TEST_F(TestUhdSelectionFlow, MultipleEnginesIndependentSelection)
{
    // Engine 100: k1 wins
    auto k1 = makeCandidate(1, 1);
    auto k2 = makeCandidate(2, 10);
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {k1, k2}));

    // Engine 200: k3 wins
    auto k3 = makeCandidate(3, 2);
    auto k4 = makeCandidate(4, 20);
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(200, {k3, k4}));

    auto result100 = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());
    auto result200 = SelectionEngine::select(200, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result100.applied);
    EXPECT_TRUE(result200.applied);
    EXPECT_EQ(result100.bestKernelId, 1);
    EXPECT_EQ(result200.bestKernelId, 3);
}

} // namespace
