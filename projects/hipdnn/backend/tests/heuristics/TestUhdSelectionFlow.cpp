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

#include "GbdtModelTestBuilder.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>

using hipdnn_backend::heuristics::uhd::EngineEntry;
using hipdnn_backend::heuristics::uhd::EngineRegistry;
using hipdnn_backend::heuristics::uhd::FeatureExtractionContext;
using hipdnn_backend::heuristics::uhd::KernelCandidate;
using hipdnn_backend::heuristics::uhd::SelectionEngine;

namespace score_xform = hipdnn_backend::heuristics::uhd::score_transform;

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

    void TearDown() override
    {
        EngineRegistry::instance().clear();
    }

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
    entry.uhdConfig.uhdId = "uhd_empty_test";
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

// ========== Score Transform Tests ==========

TEST_F(TestUhdSelectionFlow, ScoreTransformLog1pInverseApplied)
{
    // log1p(x) inverse is expm1(x)
    const double rawScore = 2.0;
    const double transformed = score_xform::applyInverse(rawScore, "log1p");
    EXPECT_NEAR(transformed, std::expm1(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformLogInverseApplied)
{
    const double rawScore = 2.0;
    const double transformed = score_xform::applyInverse(rawScore, "log");
    EXPECT_NEAR(transformed, std::exp(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformIdentityPassthrough)
{
    const double rawScore = 42.0;
    EXPECT_EQ(score_xform::applyInverse(rawScore, "identity"), 42.0);
    EXPECT_EQ(score_xform::applyInverse(rawScore, ""), 42.0);
    EXPECT_EQ(score_xform::applyInverse(rawScore, "unknown"), 42.0);
}

TEST_F(TestUhdSelectionFlow, ScoreTransformRoundTrip)
{
    const double original = 100.0;

    // log1p round-trip
    const double log1pForward = score_xform::applyForward(original, "log1p");
    const double log1pBack = score_xform::applyInverse(log1pForward, "log1p");
    EXPECT_NEAR(log1pBack, original, 1e-10);

    // sqrt round-trip
    const double sqrtForward = score_xform::applyForward(original, "sqrt");
    const double sqrtBack = score_xform::applyInverse(sqrtForward, "sqrt");
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
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->engineId, 100);
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
    entry.uhdConfig.uhdId = "uhd_kernel_metadata_test";
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

// ========== Fallback Behavior Tests (RFC §6) ==========

TEST_F(TestUhdSelectionFlow, AdapterWithoutHashSkipsMismatchGuard)
{
    // Renamed from FeaturesHashMismatchFallsBackToStaticOrder, which promised coverage
    // it no longer provides. The mismatch guard in SelectionEngine needs *both* hashes
    // non-empty and differing, and no adapter can reach that state today:
    // StaticOrderAdapter reports an empty hash, and TreeDataAdapter::load already
    // rejects a mismatched model and returns nullptr, so selection degrades earlier at
    // "adapter creation failed". The guard is kept as defense for a future adapter
    // that does not self-check; what this test actually pins is the empty-hash path.
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_hash_skip_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.priority\"", "\"$kernel.id\""};
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.uhdConfig.objective = "max";
    // The config hash must describe its own signature or registration rejects it, so
    // compute it rather than inventing one. StaticOrderAdapter reports an empty hash,
    // and the mismatch guard only fires when both sides are non-empty — so this
    // exercises the "adapter carries no hash" path, which must still apply the model.
    entry.uhdConfig.featuresHash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(
        entry.uhdConfig.featuresSignature);
    entry.candidates = {k1, k2};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    // StaticOrderAdapter has empty hash, config has non-empty hash
    // But our current impl only checks when BOTH are non-empty
    // So this should still apply (no mismatch when adapter hash is empty)
    EXPECT_TRUE(result.applied);
}

TEST_F(TestUhdSelectionFlow, UnknownAdapterTypeFallsBackToStaticOrder)
{
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_unknown_adapter_test";
    entry.uhdConfig.adapterType = "unknown_adapter_type";
    entry.uhdConfig.featuresSignature = {"\"$kernel.priority\""};
    entry.uhdConfig.objective = "max";
    entry.candidates = {k1, k2};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    // Should fall back to static ordering since adapter creation fails
    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.fallbackReason.empty());
    // Fallback still sorts by priority
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2); // priority 5 < priority 10
    EXPECT_EQ(result.sortedKernelIds[1], 1);
}

// ========== Registry Edge Cases ==========

TEST_F(TestUhdSelectionFlow, RegistryReregistrationOverwrites)
{
    // Register engine with one set of candidates
    auto k1 = makeCandidate(1, 10);
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {k1}));

    // Re-register with different candidates
    auto k2 = makeCandidate(2, 5);
    auto k3 = makeCandidate(3, 15);
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {k2, k3}));

    // Should have the new candidates
    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());
    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    // k2 (priority 5) should win
    EXPECT_EQ(result.bestKernelId, 2);
}

TEST_F(TestUhdSelectionFlow, GetAdapterForUnregisteredEngineReturnsNull)
{
    auto adapter = EngineRegistry::instance().getOrCreateAdapter(999);
    EXPECT_EQ(adapter, nullptr);
}

TEST_F(TestUhdSelectionFlow, GetEngineForUnregisteredReturnsNull)
{
    auto engine = EngineRegistry::instance().getEngine(999);
    EXPECT_EQ(engine, nullptr);
}

TEST_F(TestUhdSelectionFlow, SnapshotSurvivesReregistration)
{
    // RFC 0019 §9.2 supports dropping a replacement descriptor set in place. A holder
    // of the previous snapshot must keep reading a consistent entry rather than having
    // it assigned over mid-use.
    EngineRegistry::instance().registerEngine(
        createStaticOrderEngine(100, {makeCandidate(1, 10), makeCandidate(2, 20)}));

    auto snapshot = EngineRegistry::instance().getEngine(100);
    ASSERT_NE(snapshot, nullptr);
    ASSERT_EQ(snapshot->candidates.size(), 2u);

    // Replace the engine with a differently-shaped one.
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {makeCandidate(9, 1)}));

    // The old snapshot is unchanged and still readable.
    ASSERT_EQ(snapshot->candidates.size(), 2u);
    EXPECT_EQ(snapshot->candidates[0].kernelId, 1);

    // A fresh lookup sees the replacement.
    auto current = EngineRegistry::instance().getEngine(100);
    ASSERT_NE(current, nullptr);
    ASSERT_EQ(current->candidates.size(), 1u);
    EXPECT_EQ(current->candidates[0].kernelId, 9);
}

TEST_F(TestUhdSelectionFlow, NullSnapshotYieldsNullAdapterAndExtractor)
{
    const std::shared_ptr<const hipdnn_backend::heuristics::uhd::EngineEntry> none;

    EXPECT_EQ(EngineRegistry::instance().getOrCreateAdapter(none), nullptr);
    EXPECT_EQ(EngineRegistry::instance().getOrCreateExtractor(none), nullptr);
}

TEST_F(TestUhdSelectionFlow, SnapshotSurvivesRegistryClear)
{
    EngineRegistry::instance().registerEngine(createStaticOrderEngine(100, {makeCandidate(1, 10)}));

    auto snapshot = EngineRegistry::instance().getEngine(100);
    ASSERT_NE(snapshot, nullptr);

    EngineRegistry::instance().clear();

    // Reading through the snapshot after a clear must stay well-defined.
    ASSERT_EQ(snapshot->candidates.size(), 1u);
    EXPECT_EQ(snapshot->engineId, 100);
    EXPECT_EQ(EngineRegistry::instance().getEngine(100), nullptr);
}

// ========== Calibrated Score Transform Tests ==========

TEST_F(TestUhdSelectionFlow, RegisterRejectsUnsupportedScoreTransform)
{
    // applyInverse cannot signal "unknown" at the point of use — it returns the score
    // unchanged. So a transform this runtime cannot invert has to be caught at load,
    // or the model's transformed output gets reported as if it were in the declared
    // score.units and silently corrupts cross-engine comparison (RFC §12.3).
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.scoreTransform = "boxcox";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterRejectsMisspelledScoreTransform)
{
    // The failure mode that motivates fail-closed: a near-miss that would otherwise
    // pass through as a no-op.
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.scoreTransform = "log1P";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterAcceptsEverySupportedScoreTransform)
{
    int64_t engineId = 100;
    for(const auto* transform :
        hipdnn_backend::heuristics::uhd::score_transform::SUPPORTED_TRANSFORMS)
    {
        auto k1 = makeCandidate(1, 5);
        auto entry = createStaticOrderEngine(engineId, {k1});
        entry.uhdConfig.scoreTransform = transform;

        EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry))
            << "transform '" << transform << "' is listed as supported but was rejected";
        ++engineId;
    }
}

TEST_F(TestUhdSelectionFlow, SupportedTransformsCoverTheSchemaVocabulary)
{
    // flatbuffers_sdk/schemas/uhd.fbs documents the transform field as
    // (e.g., "identity", "log1p", "exp"). A name the schema advertises but the runtime
    // rejects is a descriptor that passes schema review and then fails to load.
    namespace xform = hipdnn_backend::heuristics::uhd::score_transform;

    for(const auto* documented : {"identity", "log1p", "exp"})
    {
        EXPECT_TRUE(xform::isSupported(documented)) << "uhd.fbs documents transform '" << documented
                                                    << "' but the runtime cannot invert it";
    }
}

TEST_F(TestUhdSelectionFlow, ExpTransformInvertsToLog)
{
    namespace xform = hipdnn_backend::heuristics::uhd::score_transform;

    // Round-trip: a target pushed through the forward transform comes back out.
    const double target = 4.5;
    EXPECT_NEAR(xform::applyInverse(xform::applyForward(target, "exp"), "exp"), target, 1e-9);
}

// ========== Mixed Valid/Invalid Candidates ==========

TEST_F(TestUhdSelectionFlow, AllCandidatesFailScoringFallsBack)
{
    // If all candidates fail scoring, we should fall back to static ordering
    // This is hard to test with StaticOrderAdapter since it doesn't fail
    // But we can verify the fallback path exists by checking the behavior
    // when adapter creation fails (tested above in UnknownAdapterType)

    // For now, just verify that when scoring works, we get applied=true
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());
    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.fallbackReason.empty());
}

// ========== Large Candidate Sets ==========

TEST_F(TestUhdSelectionFlow, LargeCandidateSetPerformance)
{
    // Test with many candidates to ensure reasonable performance
    std::vector<KernelCandidate> candidates;
    candidates.reserve(100);
    for(int i = 0; i < 100; ++i)
    {
        candidates.push_back(makeCandidate(i, i % 10)); // priority cycles 0-9
    }

    auto entry = createStaticOrderEngine(100, candidates);
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 100u);
    // Best candidates should be those with priority 0 (ids 0, 10, 20, ...)
    // First one should be id 0 (lowest id among priority 0)
    EXPECT_EQ(result.sortedKernelIds[0], 0);
}

// ========== Deterministic Ordering ==========

TEST_F(TestUhdSelectionFlow, SelectionIsDeterministic)
{
    // Same inputs should always produce same outputs
    auto k1 = makeCandidate(1, 5);
    auto k2 = makeCandidate(2, 5);
    auto k3 = makeCandidate(3, 5);

    auto entry = createStaticOrderEngine(100, {k3, k1, k2}); // Intentionally unordered
    EngineRegistry::instance().registerEngine(entry);

    // Run selection multiple times
    auto result1 = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());
    auto result2 = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());
    auto result3 = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    // Results should be identical
    EXPECT_EQ(result1.sortedKernelIds, result2.sortedKernelIds);
    EXPECT_EQ(result2.sortedKernelIds, result3.sortedKernelIds);

    // And deterministically ordered by id when priorities are equal
    ASSERT_EQ(result1.sortedKernelIds.size(), 3u);
    EXPECT_EQ(result1.sortedKernelIds[0], 1);
    EXPECT_EQ(result1.sortedKernelIds[1], 2);
    EXPECT_EQ(result1.sortedKernelIds[2], 3);
}

// ========== Empty Feature Vars ==========

TEST_F(TestUhdSelectionFlow, SelectionWithEmptyDeviceVars)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    // Empty device vars - should still work since static_order only uses $kernel.*
    const FeatureExtractionContext::ValueMap emptyDeviceVars;
    auto result = SelectionEngine::select(100, emptyDeviceVars, defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.bestKernelId, 1);
}

TEST_F(TestUhdSelectionFlow, SelectionWithEmptyQueryVars)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    // Empty query vars - should still work since static_order only uses $kernel.*
    const FeatureExtractionContext::ValueMap emptyQueryVars;
    auto result = SelectionEngine::select(100, defaultDeviceVars(), emptyQueryVars);

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.bestKernelId, 1);
}

// ========== RFC 0019 §13: Selection Trace (Observability) ==========

TEST_F(TestUhdSelectionFlow, TraceContainsUhdId)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.uhdId = "test-uhd-id-12345";
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.trace.uhdId, "test-uhd-id-12345");
}

TEST_F(TestUhdSelectionFlow, TraceContainsAdapterType)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.trace.adapterType, "static_order");
}

TEST_F(TestUhdSelectionFlow, TraceMarksUsedModelOnSuccess)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.trace.usedModel);
}

TEST_F(TestUhdSelectionFlow, TraceContainsFeaturesHashFromConfig)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    const auto hash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(
        entry.uhdConfig.featuresSignature);
    entry.uhdConfig.featuresHash = hash;
    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.trace.featuresHashConfig, hash);
}

TEST_F(TestUhdSelectionFlow, TraceRecordsFallbackReasonOnEngineNotFound)
{
    // Don't register any engine
    auto result = SelectionEngine::select(999, defaultDeviceVars(), defaultQueryVars());

    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.trace.fallbackReason.empty());
    EXPECT_NE(result.trace.fallbackReason.find("not found"), std::string::npos);
}

TEST_F(TestUhdSelectionFlow, TraceArchWasTrainedDefaultsToTrue)
{
    // Static order adapter returns true for isTrainedForArch (no restriction)
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    EngineRegistry::instance().registerEngine(entry);

    const FeatureExtractionContext::ValueMap deviceVars = {
        {"arch", std::string("gfx942")},
        {"cu_count", 120.0},
    };

    auto result = SelectionEngine::select(100, deviceVars, defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.trace.archWasTrained);
}

// ========== KMD Field Validation Tests (RFC §7.3) ==========

TEST_F(TestUhdSelectionFlow, RegisterEngineValidatesKmdFieldCoverage)
{
    // Create engine with features_signature referencing $kernel.tile_m
    // but candidates have no "tile_m" metadata
    auto k1 = makeCandidate(1, 5, {{"other_field", 128.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\""};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    // Should throw because tile_m is referenced but not in candidate metadata
    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterEngineAcceptsValidKmdFields)
{
    // Create engine with features_signature referencing $kernel.tile_m
    // and candidates have "tile_m" in metadata
    auto k1 = makeCandidate(1, 5, {{"tile_m", 128.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\""};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    // Should not throw
    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    EXPECT_TRUE(EngineRegistry::instance().hasEngine(100));
}

TEST_F(TestUhdSelectionFlow, RegisterEngineAcceptsImplicitPriorityAndIdFields)
{
    // $kernel.priority and $kernel.id are implicitly added by SelectionEngine
    auto k1 = makeCandidate(1, 5); // No explicit metadata

    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.priority\"", "\"$kernel.id\""};
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.candidates = {k1};

    // Should not throw because priority and id are implicit
    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
}

TEST_F(TestUhdSelectionFlow, RegisterEngineValidationErrorMessageListsMissingFields)
{
    auto k1 = makeCandidate(1, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\"", "\"$kernel.split_k\""};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    try
    {
        EngineRegistry::instance().registerEngine(entry);
        FAIL() << "Expected std::invalid_argument";
    }
    catch(const std::invalid_argument& e)
    {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("tile_m"), std::string::npos)
            << "Message should mention tile_m: " << msg;
        EXPECT_NE(msg.find("split_k"), std::string::npos)
            << "Message should mention split_k: " << msg;
    }
}

TEST_F(TestUhdSelectionFlow, RegisterEngineSkipsValidationWithEmptySignature)
{
    // Empty features_signature should skip validation
    auto k1 = makeCandidate(1, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {}; // Empty
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
}

TEST_F(TestUhdSelectionFlow, RegisterEngineSkipsValidationWithNoCandidates)
{
    // No candidates should skip validation (nothing to validate against)
    EngineEntry entry;
    entry.engineId = 100;
    entry.engineName = "TestEngine";
    entry.uhdConfig.uhdId = "test-uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\""};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {}; // Empty

    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
}

// ========== Cached Extractor Tests ==========

TEST_F(TestUhdSelectionFlow, CachedExtractorProducesConsistentResults)
{
    // Register an engine with a features signature
    auto k1 = makeCandidate(1, 10, {{"tile_m", 64.0}, {"tile_n", 64.0}});
    auto k2 = makeCandidate(2, 5, {{"tile_m", 128.0}, {"tile_n", 128.0}});
    auto k3 = makeCandidate(3, 1, {{"tile_m", 256.0}, {"tile_n", 256.0}});

    EngineEntry entry;
    entry.engineId = 200;
    entry.engineName = "CacheTestEngine";
    entry.uhdConfig.uhdId = "cache-test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\"", "\"$kernel.tile_n\""};
    entry.uhdConfig.staticOrderFields = {"tile_m"};
    entry.candidates = {k1, k2, k3};

    EngineRegistry::instance().registerEngine(entry);

    // Run selection 100 times and verify consistent results
    FeatureExtractionContext::ValueMap deviceVars;
    deviceVars["arch"] = std::string("gfx942");

    FeatureExtractionContext::ValueMap queryVars;
    queryVars["m"] = 1024.0;

    std::optional<int64_t> firstBestKernel;

    for(int i = 0; i < 100; ++i)
    {
        auto result = SelectionEngine::select(200, deviceVars, queryVars);
        ASSERT_TRUE(result.applied) << "Selection failed on iteration " << i;
        ASSERT_TRUE(result.bestKernelId.has_value()) << "No best kernel on iteration " << i;

        if(!firstBestKernel.has_value())
        {
            firstBestKernel = result.bestKernelId;
        }
        else
        {
            EXPECT_EQ(result.bestKernelId.value(), firstBestKernel.value())
                << "Inconsistent result on iteration " << i;
        }
    }
}

TEST_F(TestUhdSelectionFlow, GetOrCreateExtractorReturnsNullForUnknownEngine)
{
    auto extractor = EngineRegistry::instance().getOrCreateExtractor(999999);
    EXPECT_EQ(extractor, nullptr);
}

TEST_F(TestUhdSelectionFlow, GetOrCreateExtractorReturnsSameInstance)
{
    // Register an engine
    auto k1 = makeCandidate(1, 10, {{"tile_m", 64.0}});

    EngineEntry entry;
    entry.engineId = 300;
    entry.engineName = "ExtractorCacheEngine";
    entry.uhdConfig.uhdId = "extractor-cache";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"\"$kernel.tile_m\""};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    EngineRegistry::instance().registerEngine(entry);

    // Get extractor twice and verify same instance
    auto extractor1 = EngineRegistry::instance().getOrCreateExtractor(300);
    auto extractor2 = EngineRegistry::instance().getOrCreateExtractor(300);

    ASSERT_NE(extractor1, nullptr);
    ASSERT_NE(extractor2, nullptr);
    EXPECT_EQ(extractor1.get(), extractor2.get())
        << "Extractor should be cached and return same instance";
}

// ========== features_hash contract at registration (RFC §7.3) ==========

TEST_F(TestUhdSelectionFlow, RegisterRejectsHashThatDoesNotDescribeSignature)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.featuresHash = "sha256:0000000000000000";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterRejectsHashOfPermutedSignature)
{
    // The hash must pin order, not just membership.
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});

    std::vector<std::string> permuted = entry.uhdConfig.featuresSignature;
    std::reverse(permuted.begin(), permuted.end());
    entry.uhdConfig.featuresHash
        = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(permuted);

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterAcceptsMatchingHash)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.featuresHash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(
        entry.uhdConfig.featuresSignature);

    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
}

TEST_F(TestUhdSelectionFlow, RegisterAcceptsAbsentHashForStaticOrder)
{
    // features_hash is optional for feature-less adapters (RFC §7.3).
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.featuresHash.clear();

    EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
}

TEST_F(TestUhdSelectionFlow, RegisterAcceptsBareReferenceSignature)
{
    // The RFC-canonical spelling must survive registration end to end.
    auto k1 = makeCandidate(1, 10, {{"tile_m", 64.0}});
    auto k2 = makeCandidate(2, 5, {{"tile_m", 128.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "bare-ref";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"$kernel.priority", "$kernel.id"};
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.uhdConfig.objective = "max";
    entry.uhdConfig.featuresHash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(
        entry.uhdConfig.featuresSignature);
    entry.candidates = {k1, k2};

    ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2); // lower priority wins
}

// ========== Fail-open and trace preservation (RFC §6 step 6, §13) ==========

TEST_F(TestUhdSelectionFlow, UnknownAdapterStillProducesOrdering)
{
    // RFC §6 step 6: a model that cannot be built degrades to priority+id; the engine
    // must stay rankable rather than drop out.
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "unknown-adapter-uhd";
    entry.uhdConfig.adapterType = "no_such_adapter";
    entry.uhdConfig.featuresSignature = {"$kernel.priority"};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1, k2};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_FALSE(result.applied) << "the model did not run";
    EXPECT_TRUE(result.hasOrdering()) << "but a usable ordering must still come back";
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2); // priority 5 beats priority 10
}

TEST_F(TestUhdSelectionFlow, FallbackPreservesModelProvenanceInTrace)
{
    // Provenance set before the failure must survive into the returned trace —
    // otherwise the failures worth diagnosing are the ones that report an empty uhdId.
    auto k1 = makeCandidate(1, 10);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "provenance-uhd";
    entry.uhdConfig.adapterType = "no_such_adapter";
    entry.uhdConfig.featuresSignature = {"$kernel.priority"};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_FALSE(result.applied);
    EXPECT_EQ(result.trace.uhdId, "provenance-uhd");
    EXPECT_EQ(result.trace.adapterType, "no_such_adapter");
    EXPECT_FALSE(result.trace.usedModel);
    EXPECT_FALSE(result.trace.fallbackReason.empty());
}

TEST_F(TestUhdSelectionFlow, HasOrderingIsFalseWhenEngineMissing)
{
    auto result = SelectionEngine::select(999, defaultDeviceVars(), defaultQueryVars());

    EXPECT_FALSE(result.applied);
    EXPECT_FALSE(result.hasOrdering());
}

TEST_F(TestUhdSelectionFlow, MalformedSignatureEntryThrowsInvalidArgument)
{
    // registerEngine documents std::invalid_argument; a malformed entry makes the
    // extractor throw JsonLogicError, which a caller catching invalid_argument would
    // miss.
    auto k1 = makeCandidate(1, 5, {{"tile_m", 64.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "malformed";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"{not valid json"};
    entry.uhdConfig.staticOrderFields = {"priority"};
    entry.candidates = {k1};

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterRejectsHashDeclaredOverEmptySignature)
{
    // A hash over no features is self-inconsistent; skipping the check when the
    // signature is empty would let it through unvalidated.
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "hash-no-signature";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {};
    entry.uhdConfig.featuresHash = "sha256:0000000000000000";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, EngineWithNoCandidatesIsNotRuledOut)
{
    // Nothing to rank is not the same as selection failing — the engine keeps its
    // place rather than being dropped.
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "no-candidates";
    entry.uhdConfig.adapterType = "static_order";
    entry.candidates = {};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.sortedKernelIds.empty());
    EXPECT_TRUE(result.hasOrdering());
}

// ========== Per-candidate binding hygiene ==========

// ========== Untrained architecture degrades (RFC §9.3) ==========

/// Fixture that writes a real tree_data artifact so arch metadata can be exercised
/// end to end through the selection flow.
class TestUhdSelectionFlowTreeData : public TestUhdSelectionFlow
{
protected:
    void SetUp() override
    {
        TestUhdSelectionFlow::SetUp();
        // Name the artifact after the running test so parallel suites don't collide.
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        _modelPath = (std::filesystem::temp_directory_path()
                      / ("uhd_arch_test_" + std::string(info != nullptr ? info->name() : "unknown")
                         + ".bin"))
                         .string();
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove(_modelPath, ec);
        TestUhdSelectionFlow::TearDown();
    }

    /// Register a tree_data engine whose model was trained only on `trainedArches`.
    void registerTreeDataEngine(int64_t engineId, const std::vector<std::string>& trainedArches)
    {
        namespace uhd_test = hipdnn_backend::heuristics::uhd::testing;

        const std::vector<std::string> signature = {"$kernel.priority"};
        const auto hash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(signature);

        uhd_test::GbdtModelTestBuilder builder;
        builder.setNumFeatures(static_cast<int32_t>(signature.size()))
            .setFeaturesHash(hash)
            .setTrainingArches(trainedArches)
            .setModelVersion("v1.2.3")
            .addTree(uhd_test::makeLeafTreeSpec(1.0));
        ASSERT_TRUE(builder.buildToFile(_modelPath));

        EngineEntry entry;
        entry.engineId = engineId;
        entry.uhdConfig.uhdId = "tree-uhd";
        entry.uhdConfig.adapterType = "tree_data";
        entry.uhdConfig.featuresSignature = signature;
        entry.uhdConfig.featuresHash = hash;
        entry.uhdConfig.staticOrderFields = {"priority"};
        entry.uhdConfig.objective = "max";
        entry.uhdConfig.modelArtifactPath = _modelPath;
        entry.candidates = {makeCandidate(1, 10), makeCandidate(2, 5)};

        ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    }

    /// Register a tree_data engine that scores by $kernel.priority.
    ///
    /// The tree splits on priority: at or below `threshold` yields `leftLeaf`,
    /// otherwise `rightLeaf`. That gives a real per-candidate score, which is what
    /// objective, transform and score-validity behaviour need — static_order is a
    /// declared precedence and produces no score at all.
    void registerScoringEngine(int64_t engineId,
                               const std::vector<KernelCandidate>& candidates,
                               double threshold = 7.0,
                               double leftLeaf = 2.0,
                               double rightLeaf = 1.0,
                               const std::string& objective = "max",
                               const std::string& transform = "",
                               bool calibrated = false)
    {
        namespace uhd_test = hipdnn_backend::heuristics::uhd::testing;

        const std::vector<std::string> signature = {"$kernel.priority"};
        const auto hash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(signature);

        uhd_test::GbdtModelTestBuilder::TreeSpec split;
        split.featureIndices = {0, 0, 0};
        split.thresholds = {threshold, 0.0, 0.0};
        split.leftChildren = {1, -1, -1};
        split.rightChildren = {2, -1, -1};
        split.leafValues = {0.0, leftLeaf, rightLeaf};
        split.defaultLeft = {1, 1, 1};

        uhd_test::GbdtModelTestBuilder builder;
        builder.setNumFeatures(static_cast<int32_t>(signature.size()))
            .setFeaturesHash(hash)
            .setModelVersion("v1.2.3")
            .addTree(split);
        ASSERT_TRUE(builder.buildToFile(_modelPath));

        EngineEntry entry;
        entry.engineId = engineId;
        entry.uhdConfig.uhdId = "scoring-uhd";
        entry.uhdConfig.adapterType = "tree_data";
        entry.uhdConfig.featuresSignature = signature;
        entry.uhdConfig.featuresHash = hash;
        entry.uhdConfig.objective = objective;
        entry.uhdConfig.scoreTransform = transform;
        entry.uhdConfig.scoreCalibrated = calibrated;
        entry.uhdConfig.modelArtifactPath = _modelPath;
        entry.candidates = candidates;

        ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    }

    /// Register a tree_data engine over a caller-supplied signature, for the tests
    /// that need a specific feature reference rather than $kernel.priority.
    void registerScoringEngineWithSignature(int64_t engineId,
                                            const std::vector<std::string>& signature,
                                            const std::vector<KernelCandidate>& candidates)
    {
        namespace uhd_test = hipdnn_backend::heuristics::uhd::testing;
        const auto hash = hipdnn_backend::heuristics::uhd::FeatureExtractor::computeHash(signature);

        uhd_test::GbdtModelTestBuilder builder;
        builder.setNumFeatures(static_cast<int32_t>(signature.size()))
            .setFeaturesHash(hash)
            .addTree(uhd_test::makeLeafTreeSpec(1.0));
        ASSERT_TRUE(builder.buildToFile(_modelPath));

        EngineEntry entry;
        entry.engineId = engineId;
        entry.uhdConfig.uhdId = "sig-uhd";
        entry.uhdConfig.adapterType = "tree_data";
        entry.uhdConfig.featuresSignature = signature;
        entry.uhdConfig.featuresHash = hash;
        entry.uhdConfig.objective = "max";
        entry.uhdConfig.modelArtifactPath = _modelPath;
        entry.candidates = candidates;

        ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    }

    static FeatureExtractionContext::ValueMap deviceVarsForArch(const std::string& arch)
    {
        auto vars = defaultDeviceVars();
        vars[hipdnn_backend::heuristics::uhd::DEVICE_ARCH_KEY] = arch;
        return vars;
    }

    std::string _modelPath;
};

TEST_F(TestUhdSelectionFlowTreeData, TrainedArchUsesModel)
{
    registerTreeDataEngine(100, {"gfx942", "gfx950"});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx942"), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.trace.usedModel);
    EXPECT_TRUE(result.trace.archWasTrained);
    EXPECT_EQ(result.trace.deviceArch, "gfx942");
}

TEST_F(TestUhdSelectionFlowTreeData, UntrainedArchDegradesToStaticOrder)
{
    // RFC §9.3: a model trained only on gfx942 has no basis for ranking gfx1100, so it
    // must degrade rather than extrapolate.
    registerTreeDataEngine(100, {"gfx942"});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx1100"), defaultQueryVars());

    EXPECT_FALSE(result.applied) << "the model must not score an untrained arch";
    EXPECT_FALSE(result.trace.usedModel);
    EXPECT_FALSE(result.trace.archWasTrained);
    EXPECT_EQ(result.trace.deviceArch, "gfx1100");
    EXPECT_NE(result.fallbackReason.find("gfx1100"), std::string::npos);

    // Fail open: still ranked, by priority then id.
    EXPECT_TRUE(result.hasOrdering());
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2);
}

TEST_F(TestUhdSelectionFlowTreeData, UntrainedArchFallbackKeepsProvenance)
{
    registerTreeDataEngine(100, {"gfx942"});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx1100"), defaultQueryVars());

    EXPECT_EQ(result.trace.uhdId, "tree-uhd");
    EXPECT_EQ(result.trace.adapterType, "tree_data");
    EXPECT_EQ(result.trace.modelVersion, "v1.2.3");
    ASSERT_EQ(result.trace.trainingArches.size(), 1u);
    EXPECT_EQ(result.trace.trainingArches[0], "gfx942");
}

TEST_F(TestUhdSelectionFlowTreeData, EmptyTrainingArchesMeansNoRestriction)
{
    registerTreeDataEngine(100, {});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx1100"), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.trace.archWasTrained);
}

TEST_F(TestUhdSelectionFlowTreeData, MissingArchKeySkipsTheCheck)
{
    // Device properties without an arch cannot be checked; selection
    // proceeds rather than degrading on missing information.
    registerTreeDataEngine(100, {"gfx942"});

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_TRUE(result.trace.deviceArch.empty());
}

TEST_F(TestUhdSelectionFlow, CandidateMissingMetadataDoesNotInheritPreviousValue)
{
    // The context is reused across candidates for speed (RFC §6 step 2). A candidate
    // that omits a referenced field must fail to score rather than silently pick up
    // the previous candidate's value.
    auto withField = makeCandidate(1, 10, {{"tile_m", 64.0}});
    auto withoutField = makeCandidate(2, 5); // no tile_m

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "staleness";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.featuresSignature = {"$kernel.tile_m"};
    entry.uhdConfig.staticOrderFields = {"tile_m"};
    entry.candidates = {withField, withoutField};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_EQ(result.scoredCandidates.size(), 2u);

    // Exactly the candidate carrying tile_m scores; the other is marked invalid.
    const auto scoredWithout = std::find_if(result.scoredCandidates.begin(),
                                            result.scoredCandidates.end(),
                                            [](const auto& sc) { return sc.kernelId == 2; });
    ASSERT_NE(scoredWithout, result.scoredCandidates.end());
    EXPECT_FALSE(scoredWithout->scoreValid)
        << "candidate without tile_m must not inherit the previous candidate's value";
}

// ========== objective vocabulary (RFC 0019 §5, §6 step 4) ==========

TEST_F(TestUhdSelectionFlow, RegisterRejectsUnrecognizedObjective)
{
    // sortByObjective computes maximize = (objective != "min"), so a typo silently
    // maximizes. For a min-objective model that inverts the ranking with no diagnostic.
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.objective = "minimize";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterRejectsWrongCaseObjective)
{
    auto k1 = makeCandidate(1, 5);
    auto entry = createStaticOrderEngine(100, {k1});
    entry.uhdConfig.objective = "MIN";

    EXPECT_THROW(EngineRegistry::instance().registerEngine(entry), std::invalid_argument);
}

TEST_F(TestUhdSelectionFlow, RegisterAcceptsBothObjectiveValuesAndEmpty)
{
    int64_t engineId = 100;
    for(const auto* objective : {"max", "min", ""})
    {
        auto k1 = makeCandidate(1, 5);
        auto entry = createStaticOrderEngine(engineId, {k1});
        entry.uhdConfig.objective = objective;
        EXPECT_NO_THROW(EngineRegistry::instance().registerEngine(entry))
            << "objective '" << objective << "' should be accepted";
        ++engineId;
    }
}

// ========== static_order packing stays exact (RFC 0019 §6 step 5) ==========

// ========== static_order is a declared precedence, not a score (RFC 0019 §5) ==========

TEST_F(TestUhdSelectionFlow, StaticOrderNeedsNoFeaturesSignature)
{
    // The RFC's canonical descriptor: {"adapter":"static_order","order":["priority","id"]}
    // — no features, no model, no hash. This previously failed adapter creation and
    // limped in through the degrade path.
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "canonical-static";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.candidates = {makeCandidate(1, 10), makeCandidate(2, 5)};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied) << "the declared order is the model, not a fallback";
    EXPECT_TRUE(result.trace.usedModel);
    EXPECT_TRUE(result.fallbackReason.empty());
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2); // priority 5 before 10
}

TEST_F(TestUhdSelectionFlow, StaticOrderRanksLargeIdsCorrectly)
{
    // The packed-scalar encoding this replaces let a kernel id above the radix
    // outweigh a whole unit of priority, so priority 1 beat priority 0. A pairwise
    // comparator has no encoding to overflow.
    auto bestPriorityHugeId = makeCandidate(20000000000L, 0);
    auto worsePrioritySmallId = makeCandidate(1, 1);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_large_ids_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};
    entry.candidates = {bestPriorityHugeId, worsePrioritySmallId};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 20000000000L) << "priority 0 must win regardless of id";
}

TEST_F(TestUhdSelectionFlow, StaticOrderRanksByArbitraryKmdField)
{
    auto small = makeCandidate(1, 0, {{"tile_m", 64.0}});
    auto large = makeCandidate(2, 0, {{"tile_m", 256.0}});

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_kmd_field_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"tile_m"};
    entry.candidates = {large, small};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 1) << "lower tile_m ranks first";
}

TEST_F(TestUhdSelectionFlow, StaticOrderAcceptsNamespacedFieldNames)
{
    auto k1 = makeCandidate(1, 10);
    auto k2 = makeCandidate(2, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_namespaced_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"$kernel.priority"};
    entry.candidates = {k1, k2};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2);
}

TEST_F(TestUhdSelectionFlow, StaticOrderSortsMissingFieldLast)
{
    // A candidate that does not carry the ordering field still gets ranked, after the
    // ones that do, so the order stays total and no candidate is silently dropped.
    auto has = makeCandidate(1, 0, {{"tile_m", 64.0}});
    auto lacks = makeCandidate(2, 0);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_missing_field_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"tile_m"};
    entry.candidates = {lacks, has};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 1);
    EXPECT_EQ(result.sortedKernelIds[1], 2);
}

TEST_F(TestUhdSelectionFlow, StaticOrderReportsNoScore)
{
    // A declared order carries no score, so a static_order engine cannot take part in
    // the §12.3 cross-engine comparison. Reporting one would be inventing a number.
    auto k1 = makeCandidate(1, 5);

    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "uhd_no_score_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.candidates = {k1};

    EngineRegistry::instance().registerEngine(entry);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    EXPECT_EQ(result.bestKernelId, 1);
    EXPECT_FALSE(result.bestScore.has_value());
}

// ========== Scoring behaviour, on a real tree_data model ==========
//
// These previously used static_order as a stand-in for a scorer. That conflation is
// what the packed-key bug grew out of: static_order produces no score, so objective,
// transform and score-validity have nothing to act on. They need an actual model.

TEST_F(TestUhdSelectionFlowTreeData, MaxObjectiveSelectsHighestScore)
{
    // priority <= 7 scores 2.0, above scores 1.0.
    registerScoringEngine(100, {makeCandidate(1, 10), makeCandidate(2, 5)});

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2); // scored 2.0
    ASSERT_TRUE(result.bestScore.has_value());
    EXPECT_DOUBLE_EQ(*result.bestScore, 2.0);
}

TEST_F(TestUhdSelectionFlowTreeData, MinObjectiveSelectsLowestScore)
{
    registerScoringEngine(100, {makeCandidate(1, 10), makeCandidate(2, 5)}, 7.0, 2.0, 1.0, "min");

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    EXPECT_TRUE(result.applied);
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 1) << "priority 10 scored 1.0, the lower score";
}

TEST_F(TestUhdSelectionFlowTreeData, ScoreOnlyReturnsBestScore)
{
    registerScoringEngine(100, {makeCandidate(1, 10), makeCandidate(2, 5)});

    auto result = SelectionEngine::scoreOnly(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_TRUE(result.bestScore.has_value());
    EXPECT_DOUBLE_EQ(*result.bestScore, 2.0);
    EXPECT_EQ(result.bestKernelId, 2);
}

TEST_F(TestUhdSelectionFlowTreeData, TransformIsInvertedToRecoverDeclaredUnits)
{
    // Trained on log1p(target), so the reported score is expm1 of the raw leaf value.
    registerScoringEngine(100, {makeCandidate(1, 5)}, 7.0, 2.0, 1.0, "max", "log1p", true);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_TRUE(result.bestScore.has_value());
    EXPECT_NEAR(*result.bestScore, std::expm1(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlowTreeData, TransformIsInvertedEvenWhenUncalibrated)
{
    // `transform` and `calibrated` are orthogonal: the first recovers the declared
    // units, the second states cross-engine comparability.
    registerScoringEngine(100, {makeCandidate(1, 5)}, 7.0, 2.0, 1.0, "max", "log1p", false);

    auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

    ASSERT_TRUE(result.bestScore.has_value());
    EXPECT_NEAR(*result.bestScore, std::expm1(2.0), 1e-10);
}

TEST_F(TestUhdSelectionFlowTreeData, IdentityAndEmptyTransformLeaveScoreUnchanged)
{
    for(const auto* transform : {"", "identity"})
    {
        EngineRegistry::instance().clear();
        registerScoringEngine(100, {makeCandidate(1, 5)}, 7.0, 2.0, 1.0, "max", transform, true);

        auto result = SelectionEngine::select(100, defaultDeviceVars(), defaultQueryVars());

        ASSERT_TRUE(result.bestScore.has_value()) << "transform '" << transform << "'";
        EXPECT_DOUBLE_EQ(*result.bestScore, 2.0) << "transform '" << transform << "'";
    }
}

TEST_F(TestUhdSelectionFlowTreeData, TraceRecordsDeviceArchFromDeviceVars)
{
    registerScoringEngine(100, {makeCandidate(1, 5)});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx942"), defaultQueryVars());

    EXPECT_EQ(result.trace.deviceArch, "gfx942");
}

TEST_F(TestUhdSelectionFlowTreeData, StringDevicePropertyInSignatureDegrades)
{
    // $device.arch is bound as a string. RFC §7.2 makes that a type error rather
    // than a silent NaN, which a GBDT would otherwise route down default_left and
    // score as ordinary data.
    registerScoringEngineWithSignature(
        100, {"$device.arch"}, {makeCandidate(1, 10), makeCandidate(2, 5)});

    auto result = SelectionEngine::select(100, deviceVarsForArch("gfx942"), defaultQueryVars());

    EXPECT_FALSE(result.applied);
    EXPECT_TRUE(result.hasOrdering());
    ASSERT_EQ(result.sortedKernelIds.size(), 2u);
    EXPECT_EQ(result.sortedKernelIds[0], 2) << "degrades to priority order";
}

TEST_F(TestUhdSelectionFlowTreeData, SnapshotResolvesItsOwnAdapter)
{
    // A snapshot is only self-consistent if the adapter comes from it too; resolving
    // by id would pair a new model with the old snapshot's config and candidates.
    registerScoringEngine(100, {makeCandidate(1, 5)});
    auto snapshot = EngineRegistry::instance().getEngine(100);
    ASSERT_NE(snapshot, nullptr);

    auto fromSnapshot = EngineRegistry::instance().getOrCreateAdapter(snapshot);
    auto fromId = EngineRegistry::instance().getOrCreateAdapter(100);

    ASSERT_NE(fromSnapshot, nullptr);
    ASSERT_NE(fromId, nullptr);
    EXPECT_EQ(fromSnapshot.get(), fromId.get()) << "same entry, so the cache is shared";
}

} // namespace
