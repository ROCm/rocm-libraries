// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestUhdTraceRetrieval.cpp
 * @brief Tests for UHD trace retrieval API (RFC 0019 §12).
 *
 * Verifies that SelectionTrace can be retrieved programmatically after
 * finalize via hipdnnHeuristicPolicyGetTrace.
 */

#include "heuristics/uhd/EngineRegistry.hpp"
#include "heuristics/uhd/FeatureExtractor.hpp"
#include "heuristics/uhd/SelectionEngine.hpp"
#include "heuristics/uhd/UhdBuiltIn.hpp"
#include "plugin/HeuristicPlugin.hpp"

#include "GbdtModelTestBuilder.hpp"

#include <hipdnn_data_sdk/utilities/PolicyNames.hpp>
#include <hipdnn_plugin_sdk/HeuristicsPluginApi.h>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <system_error>
#include <vector>

using namespace hipdnn_backend::heuristics::uhd;
using hipdnn_backend::plugin::HeuristicPlugin;
using hipdnn_backend::plugin::HeuristicPluginFunctionTable;

namespace
{

const int64_t UHD_POLICY_ID = hipdnn_data_sdk::utilities::policyNameToId("SelectionHeuristic::UHD");

// Convenience: grab the raw function table for direct C-ABI tests.
const HeuristicPluginFunctionTable& uhdAbi()
{
    static const HeuristicPluginFunctionTable s_funcs = populateFunctionTable();
    return s_funcs;
}

class TestUhdTraceRetrieval : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Clear registry before each test
        EngineRegistry::instance().clear();

        // Create plugin wrapper (proper way to register the UHD policy)
        _plugin = HeuristicPlugin::createBuiltIn(populateFunctionTable(), "built-in:UHD-trace-test");
    }

    void TearDown() override
    {
        EngineRegistry::instance().clear();
    }

    std::shared_ptr<HeuristicPlugin> _plugin;
};

TEST_F(TestUhdTraceRetrieval, GetTraceAfterFinalize)
{
    // Register a simple static_order engine
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "test_uhd_trace";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};

    // Add one candidate
    KernelCandidate candidate;
    candidate.kernelId = 1;
    candidate.metadata["priority"] = 10.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle and policy descriptor via plugin wrapper
    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);

    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs
    std::array<int64_t, 1> engineIds = {100};
    _plugin->setEngineIds(desc, engineIds.data(), 1);

    // Finalize (might return false if no device properties set, but trace should still work)
    _plugin->finalize(desc);

    // Retrieve trace (use raw function table - getTrace not wrapped)
    const char* traceJson = nullptr;
    auto status = uhdAbi().policyGetTrace(desc, 100, &traceJson);

    // If finalize didn't apply (common without device props), trace won't be available
    if(status == HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED)
    {
        GTEST_SKIP() << "UHD finalize declined (no device properties); trace not available";
    }

    ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson, nullptr);

    // Parse JSON
    nlohmann::json trace = nlohmann::json::parse(traceJson);

    // Verify trace fields
    EXPECT_EQ(trace["uhd_id"], "test_uhd_trace");
    EXPECT_EQ(trace["adapter_type"], "static_order");
    // Note: static_order reports used_model=true because it created an adapter instance
    // even though it doesn't use ML. This is acceptable trace behavior.
    EXPECT_TRUE(trace.contains("used_model"));

    // Cleanup
    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdTraceRetrieval, GetTraceNotFinalizedReturnsError)
{
    // Create handle and descriptor
    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);

    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    // Try to get trace before finalize (use raw function table)
    const char* traceJson = nullptr;
    EXPECT_EQ(uhdAbi().policyGetTrace(desc, 100, &traceJson),
              HIPDNN_PLUGIN_STATUS_NOT_INITIALIZED);

    // Cleanup
    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdTraceRetrieval, GetTraceForUnknownEngineReturnsNotSupported)
{
    // Register engine 100 with candidates
    EngineEntry entry;
    entry.engineId = 100;
    entry.uhdConfig.uhdId = "test_uhd";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};

    KernelCandidate candidate;
    candidate.kernelId = 1;
    candidate.metadata["priority"] = 5.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle and descriptor
    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);

    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    // Set engine IDs and finalize
    std::array<int64_t, 1> engineIds = {100};
    _plugin->setEngineIds(desc, engineIds.data(), 1);

    _plugin->finalize(desc);

    // Try to get trace for engine that wasn't finalized (engine 200)
    const char* traceJson = nullptr;
    EXPECT_EQ(uhdAbi().policyGetTrace(desc, 200, &traceJson), HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);

    // Cleanup
    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdTraceRetrieval, GetTraceWithModelAdapter)
{
    // This test demonstrates trace retrieval for a model-based adapter
    // when model artifacts become available.
    //
    // For now, we test with static_order since tree_data/onnx/custom_library
    // require actual model files.

    EngineEntry entry;
    entry.engineId = 200;
    entry.uhdConfig.uhdId = "model_uhd_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"priority", "id"};

    KernelCandidate candidate;
    candidate.kernelId = 5;
    candidate.metadata["priority"] = 20.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    // Create handle and descriptor
    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);

    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    std::array<int64_t, 1> engineIds = {200};
    _plugin->setEngineIds(desc, engineIds.data(), 1);

    _plugin->finalize(desc);

    // Get trace (use raw function table)
    const char* traceJson = nullptr;
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, 200, &traceJson), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson, nullptr);

    // Parse and verify
    nlohmann::json trace = nlohmann::json::parse(traceJson);
    EXPECT_EQ(trace["uhd_id"], "model_uhd_test");
    EXPECT_TRUE(trace.contains("adapter_type"));

    // Cleanup
    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdTraceRetrieval, TraceJsonCachedAcrossMultipleCalls)
{
    // Verify that the JSON string is cached and remains valid

    EngineEntry entry;
    entry.engineId = 300;
    entry.uhdConfig.uhdId = "cached_trace_test";
    entry.uhdConfig.adapterType = "static_order";
    entry.uhdConfig.staticOrderFields = {"id"};

    KernelCandidate candidate;
    candidate.kernelId = 10;
    candidate.metadata["id"] = 100.0;
    entry.candidates.push_back(candidate);

    EngineRegistry::instance().registerEngine(std::move(entry));

    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);

    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    std::array<int64_t, 1> engineIds = {300};
    _plugin->setEngineIds(desc, engineIds.data(), 1);

    _plugin->finalize(desc);

    // Get trace first time (use raw function table)
    const char* traceJson1 = nullptr;
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, 300, &traceJson1), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson1, nullptr);

    // Get trace second time (should be cached)
    const char* traceJson2 = nullptr;
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, 300, &traceJson2), HIPDNN_PLUGIN_STATUS_SUCCESS);
    ASSERT_NE(traceJson2, nullptr);

    // Pointers should be identical (same cached string)
    EXPECT_EQ(traceJson1, traceJson2);

    // Content should be identical
    EXPECT_STREQ(traceJson1, traceJson2);

    // Cleanup
    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

// ========== Kernel ranking published through the trace (RFC 0019 §2, §13) ==========
//
// The kernel ranking is what the UHD actually computes, and the heuristic plugin ABI
// returns engine IDs only. It therefore rides the trace channel as `ranked_kernel_ids`.

/// Kernel every declared order puts first: lowest priority value, lowest id, first
/// registered. Only a model that disagrees with priority can rank it second.
constexpr int64_t KERNEL_DECLARED_FIRST = 7;

/// Kernel the model puts first: highest priority value, so it lands on the tree's
/// high-scoring side.
constexpr int64_t KERNEL_MODEL_FIRST = 9;

constexpr int64_t ENGINE_WITH_MODEL = 400;
constexpr int64_t ENGINE_WITHOUT_UHD = 401;
constexpr int64_t ENGINE_UNREGISTERED = 402;

KernelCandidate makeCandidate(int64_t kernelId, int64_t priority)
{
    KernelCandidate k;
    k.kernelId = kernelId;
    k.priority = priority;
    return k;
}

/// Fixture backed by a real tree_data artifact. static_order would not do: it ranks by
/// declared precedence, so a ranking it produced could not distinguish "the model's
/// order was preserved" from "the fallback's order was preserved".
class TestUhdRankedKernelIds : public TestUhdTraceRetrieval
{
protected:
    void SetUp() override
    {
        TestUhdTraceRetrieval::SetUp();
        // Name the artifact after the running test so parallel suites don't collide.
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        _modelPath = (std::filesystem::temp_directory_path()
                      / ("uhd_ranking_test_" + std::string(info != nullptr ? info->name() : "unknown")
                         + ".bin"))
                         .string();
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove(_modelPath, ec);
        TestUhdTraceRetrieval::TearDown();
    }

    /// Register a tree_data engine whose model inverts the priority order: the tree
    /// splits on $kernel.priority and scores the high-priority-value side best, so with
    /// objective=max the model's ranking is the reverse of the priority, id and
    /// registration orders, which all agree with each other here.
    void registerPriorityInvertingEngine(int64_t engineId)
    {
        namespace uhd_test = hipdnn_backend::heuristics::uhd::testing;

        const std::vector<std::string> signature = {"$kernel.priority"};
        const auto hash = FeatureExtractor::computeHash(signature);

        uhd_test::GbdtModelTestBuilder::TreeSpec split;
        split.featureIndices = {0, 0, 0};
        split.thresholds = {7.0, 0.0, 0.0};
        split.leftChildren = {1, -1, -1};
        split.rightChildren = {2, -1, -1};
        split.leafValues = {0.0, 1.0, 5.0}; // priority <= 7 scores 1.0, above scores 5.0
        split.defaultLeft = {1, 1, 1};

        uhd_test::GbdtModelTestBuilder builder;
        builder.setNumFeatures(static_cast<int32_t>(signature.size()))
            .setFeaturesHash(hash)
            .setModelVersion("v1.2.3")
            .addTree(split);
        ASSERT_TRUE(builder.buildToFile(_modelPath));

        EngineEntry entry;
        entry.engineId = engineId;
        entry.uhdConfig.uhdId = "ranking-uhd";
        entry.uhdConfig.adapterType = "tree_data";
        entry.uhdConfig.featuresSignature = signature;
        entry.uhdConfig.featuresHash = hash;
        entry.uhdConfig.objective = "max";
        entry.uhdConfig.modelArtifactPath = _modelPath;
        entry.candidates = {makeCandidate(KERNEL_DECLARED_FIRST, 1),
                            makeCandidate(KERNEL_MODEL_FIRST, 20)};

        ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    }

    /// Register the same two candidates under a static_order UHD, which ranks by
    /// priority then id — the exact opposite of the model above.
    static void registerPriorityOrderedEngine(int64_t engineId)
    {
        EngineEntry entry;
        entry.engineId = engineId;
        entry.uhdConfig.uhdId = "declared-uhd";
        entry.uhdConfig.adapterType = "static_order";
        entry.uhdConfig.staticOrderFields = {"priority", "id"};
        entry.uhdConfig.objective = "max";
        entry.candidates = {makeCandidate(KERNEL_DECLARED_FIRST, 1),
                            makeCandidate(KERNEL_MODEL_FIRST, 20)};

        ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(entry));
    }

    static std::vector<int64_t> rankedKernelIds(const nlohmann::json& trace)
    {
        return trace.at("ranked_kernel_ids").get<std::vector<int64_t>>();
    }

    std::string _modelPath;
};

TEST_F(TestUhdRankedKernelIds, TraceCarriesRankingInModelOrder)
{
    registerPriorityInvertingEngine(ENGINE_WITH_MODEL);

    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);
    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    const std::array<int64_t, 1> engineIds = {ENGINE_WITH_MODEL};
    _plugin->setEngineIds(desc, engineIds.data(), engineIds.size());
    _plugin->finalize(desc);

    const char* traceJson = nullptr;
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_WITH_MODEL, &traceJson),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    const auto trace = nlohmann::json::parse(traceJson);

    // Without this the next assertion could pass on a degraded selection that merely
    // happened to agree with the model.
    ASSERT_TRUE(trace.at("used_model").get<bool>())
        << "selection degraded, so the order below would be the fallback's: "
        << trace.dump();

    const std::vector<int64_t> modelOrder = {KERNEL_MODEL_FIRST, KERNEL_DECLARED_FIRST};
    EXPECT_EQ(rankedKernelIds(trace), modelOrder)
        << "priority, id and registration order all rank " << KERNEL_DECLARED_FIRST
        << " first; only the model ranks " << KERNEL_MODEL_FIRST << " first";

    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdRankedKernelIds, EnginesWithoutARankingStayInThePlanAndCarryNone)
{
    registerPriorityInvertingEngine(ENGINE_WITH_MODEL);

    // Registered, but with no UHD at all: selection resolves nothing to rank with, so
    // it produces no ordering (RFC 0019 §6 step 6 fails open — the engine survives).
    EngineEntry bare;
    bare.engineId = ENGINE_WITHOUT_UHD;
    bare.candidates = {makeCandidate(11, 3)};
    ASSERT_NO_THROW(EngineRegistry::instance().registerEngine(bare));

    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);
    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    // ENGINE_UNREGISTERED is absent from the registry entirely.
    const std::vector<int64_t> engineIds
        = {ENGINE_UNREGISTERED, ENGINE_WITH_MODEL, ENGINE_WITHOUT_UHD};
    _plugin->setEngineIds(desc, engineIds.data(), engineIds.size());
    _plugin->finalize(desc);

    EXPECT_EQ(_plugin->getSortedEngineIds(desc), engineIds)
        << "storing a ranking must not change which engines execute, or in what order";

    const char* traceJson = nullptr;
    EXPECT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_UNREGISTERED, &traceJson),
              HIPDNN_PLUGIN_STATUS_NOT_APPLICABLE);

    ASSERT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_WITHOUT_UHD, &traceJson),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    const auto bareTrace = nlohmann::json::parse(traceJson);
    EXPECT_FALSE(bareTrace.contains("ranked_kernel_ids"))
        << "no ordering was produced, so none may be reported: " << bareTrace.dump();

    // The ranked engine still reports its ranking, so the two above were skipped
    // individually rather than the whole store going missing.
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_WITH_MODEL, &traceJson),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::vector<int64_t> modelOrder = {KERNEL_MODEL_FIRST, KERNEL_DECLARED_FIRST};
    EXPECT_EQ(rankedKernelIds(nlohmann::json::parse(traceJson)), modelOrder);

    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

TEST_F(TestUhdRankedKernelIds, RefinalizeDoesNotServeTheRankingFromTheCachedTrace)
{
    // The trace JSON is cached per engine id, and a descriptor may be finalized again
    // after its engine's UHD is replaced (RFC 0019 §9.2). A cache entry serialized
    // before that would report the previous run's ranking for the current one.
    registerPriorityInvertingEngine(ENGINE_WITH_MODEL);

    auto handle = _plugin->createHandle();
    ASSERT_NE(handle, nullptr);
    auto desc = _plugin->createPolicyDescriptor(handle, UHD_POLICY_ID);
    ASSERT_NE(desc, nullptr);

    const std::array<int64_t, 1> engineIds = {ENGINE_WITH_MODEL};
    _plugin->setEngineIds(desc, engineIds.data(), engineIds.size());
    _plugin->finalize(desc);

    const char* traceJson = nullptr;
    ASSERT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_WITH_MODEL, &traceJson),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::vector<int64_t> modelOrder = {KERNEL_MODEL_FIRST, KERNEL_DECLARED_FIRST};
    ASSERT_EQ(rankedKernelIds(nlohmann::json::parse(traceJson)), modelOrder);

    EngineRegistry::instance().clear();
    registerPriorityOrderedEngine(ENGINE_WITH_MODEL);

    _plugin->setEngineIds(desc, engineIds.data(), engineIds.size());
    _plugin->finalize(desc);

    ASSERT_EQ(uhdAbi().policyGetTrace(desc, ENGINE_WITH_MODEL, &traceJson),
              HIPDNN_PLUGIN_STATUS_SUCCESS);
    const std::vector<int64_t> declaredOrder = {KERNEL_DECLARED_FIRST, KERNEL_MODEL_FIRST};
    EXPECT_EQ(rankedKernelIds(nlohmann::json::parse(traceJson)), declaredOrder)
        << "the second finalize ranked by declared priority; the cache served the first";

    _plugin->destroyPolicyDescriptor(desc);
    _plugin->destroyHandle(handle);
}

} // namespace
