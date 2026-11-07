// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for EXHAUSTIVE autotune mode.
// Verifies that EXHAUSTIVE mode primes engine caches via the global.benchmarking
// knob and that STANDARD mode does not.

#include <algorithm>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "AutotuneIntegrationFixture.hpp"
#include "test_plugins/TestPluginEngineIdMap.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

using IntegrationAutotuneExhaustive = hipdnn_tests::AutotuneIntegrationFixture;

// Test: EXHAUSTIVE mode with the BENCHMARK_UNPRIMED policy verifies per-engine behavior.
//
// This consolidated test verifies autotune behavior for all engine types:
// - Engine A (main): has benchmarking knob, priming succeeds, benchmark succeeds
//   → ranExhaustive=true, succeeded=true
// - Engine B: no benchmarking knob, priming skipped, benchmark succeeds
//   → ranExhaustive=false, succeeded=true
// - Engine C: no benchmarking knob, priming skipped, benchmark succeeds
//   → ranExhaustive=false, succeeded=true
// - EngineFails: has benchmarking knob, fails UNCONDITIONALLY (both priming AND benchmark)
//   → ranExhaustive=false, succeeded=false
// - EnginePrimingOnlyFails: has benchmarking knob, priming fails, benchmark succeeds
//   → ranExhaustive=false, succeeded=true, exhaustiveNotRunReason contains priming failure
//
// With the BENCHMARK_UNPRIMED policy and priming fails, the engine is still
// benchmarked (unprimed): AutotuneResult::ranExhaustive is false, and errorMessage
// notes the priming failure even though succeeded may be true.
TEST_F(IntegrationAutotuneExhaustive, ExhaustiveModeWithBenchmarkUnprimedPolicy)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;
    config.primingFailurePolicy = PrimingFailurePolicy::BENCHMARK_UNPRIMED;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify we have results
    ASSERT_FALSE(results.empty()) << "Expected at least one autotune result";

    // Engine ID constants
    constexpr int64_t ENGINE_A_ID = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
    constexpr int64_t ENGINE_B_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
    constexpr int64_t ENGINE_C_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineC>();
    constexpr int64_t ENGINE_FAILS_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineFails>();
    constexpr int64_t ENGINE_PRIMING_ONLY_FAILS_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEnginePrimingOnlyFails>();

    // Track which engines we found
    bool foundEngineA = false;
    bool foundEngineFails = false;
    bool foundEnginePrimingOnlyFails = false;
    bool anySucceeded = false;
    bool anyRanExhaustive = false;

    for(const auto& r : results)
    {
        if(r.succeeded)
        {
            anySucceeded = true;
        }
        if(r.ranExhaustive)
        {
            anyRanExhaustive = true;
        }

        if(r.engineId == ENGINE_A_ID)
        {
            foundEngineA = true;
            // Engine A: has benchmarking knob, priming succeeds, benchmark succeeds
            EXPECT_TRUE(r.succeeded) << "Engine A should succeed";
            EXPECT_TRUE(r.ranExhaustive) << "Engine A should have run exhaustive priming";
            EXPECT_TRUE(r.supportsExhaustive)
                << "Engine A exposes the benchmarking knob, so it supports exhaustive";
        }
        else if(r.engineId == ENGINE_B_ID || r.engineId == ENGINE_C_ID)
        {
            // Engines B and C: no benchmarking knob, priming skipped
            EXPECT_TRUE(r.succeeded) << "Engines B/C should succeed";
            EXPECT_FALSE(r.ranExhaustive)
                << "Engines B/C have no benchmarking knob, should not run exhaustive";
            EXPECT_FALSE(r.supportsExhaustive)
                << "Engines B/C lack the benchmarking knob, so they do not support exhaustive";
        }
        else if(r.engineId == ENGINE_FAILS_ID)
        {
            foundEngineFails = true;
            // EngineFails: fails UNCONDITIONALLY (both priming AND benchmark)
            EXPECT_FALSE(r.succeeded) << "EngineFails must not succeed";
            EXPECT_FALSE(r.ranExhaustive)
                << "EngineFails priming failed, so ranExhaustive should be false";
            EXPECT_TRUE(r.supportsExhaustive)
                << "EngineFails exposes the benchmarking knob, so it supports exhaustive";
        }
        else if(r.engineId == ENGINE_PRIMING_ONLY_FAILS_ID)
        {
            foundEnginePrimingOnlyFails = true;
            // EnginePrimingOnlyFails: priming fails, but benchmark succeeds
            // This tests the case where succeeded may be true despite a priming failure
            EXPECT_TRUE(r.succeeded)
                << "EnginePrimingOnlyFails benchmark should succeed despite priming failure";
            EXPECT_FALSE(r.ranExhaustive)
                << "EnginePrimingOnlyFails priming failed, so ranExhaustive should be false";
            EXPECT_TRUE(r.supportsExhaustive) << "EnginePrimingOnlyFails exposes the benchmarking "
                                                 "knob, so it supports exhaustive";
            EXPECT_TRUE(r.errorMessage.empty())
                << "EnginePrimingOnlyFails benchmark succeeded, so errorMessage is empty; the "
                   "priming failure belongs in exhaustiveNotRunReason. Got: "
                << r.errorMessage;
            EXPECT_FALSE(r.exhaustiveNotRunReason.empty())
                << "EnginePrimingOnlyFails should have exhaustiveNotRunReason noting priming "
                   "failure";
        }
    }

    // Verify we found the key engines we're testing
    EXPECT_TRUE(foundEngineA) << "Engine A not found in results";
    EXPECT_TRUE(foundEngineFails) << "EngineFails not found in results";
    EXPECT_TRUE(foundEnginePrimingOnlyFails) << "EnginePrimingOnlyFails not found in results";

    // Overall checks
    EXPECT_TRUE(anySucceeded) << "At least one engine should succeed";
    EXPECT_TRUE(anyRanExhaustive)
        << "At least one engine should have run exhaustive priming (Engine A)";
}

// Test: STANDARD mode does not set ranExhaustive on any engine
TEST_F(IntegrationAutotuneExhaustive, AutoModeDoesNotRunCachePriming)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    constexpr int64_t ENGINE_A_ID = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();
    constexpr int64_t ENGINE_B_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
    constexpr int64_t ENGINE_C_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineC>();

    ASSERT_FALSE(results.empty());
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.ranExhaustive)
            << "Engine " << r.engineId << " should not have ran exhaustive in STANDARD mode";

        // supportsExhaustive reflects engine capability, not whether priming ran,
        // so it is reported even in STANDARD mode where priming never runs.
        if(r.engineId == ENGINE_B_ID || r.engineId == ENGINE_C_ID)
        {
            EXPECT_FALSE(r.supportsExhaustive)
                << "Engine " << r.engineId << " lacks the benchmarking knob in STANDARD mode";
        }
        else if(r.engineId == ENGINE_A_ID)
        {
            EXPECT_TRUE(r.supportsExhaustive) << "Engine A exposes the benchmarking knob, so "
                                                 "supportsExhaustive holds in STANDARD "
                                                 "mode";
        }
    }
}

// Test: EXHAUSTIVE priming is SKIPPED (not failed) when the priming plan's
// compiled workspace exceeds the provided budget; the plan is still benchmarked
// unprimed.
//
// AutotunePluginEngineWorkspaceGrows reports 8192 for its priming plan (benchmarking knob
// enabled) and for the estimate of that configuration, but only 1024 for its non-exhaustive
// real plan. With an allocated workspace size of 4096:
//   - priming compiled size (8192) > 4096    -> priming skipped
//   - non-exhaustive compiled (1024) <= 4096 -> benchmarked normally -> succeeded
//
// A workspace skip during priming always continues, allowing the plan to run through
// the benchmarking loop un-primed, regardless of primingFailurePolicy. The skipped
// plan's result has ranExhaustive = false and the skip in exhaustiveNotRunReason.
TEST_F(IntegrationAutotuneExhaustive, ExhaustivePrimingWorkspaceSkipBenchmarksUnprimed)
{
    constexpr int64_t ENGINE_WORKSPACE_GROWS_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineWorkspaceGrows>();

    // Budget below the engine's priming compiled workspace (8192) and >= its real
    // compiled workspace (1024).
    constexpr int64_t WORKSPACE_BUDGET = 4096;

    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    auto result = bundle.graph->add_engines({ENGINE_WORKSPACE_GROWS_ID});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t estimatedWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(estimatedWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_GT(estimatedWs, WORKSPACE_BUDGET)
        << "The estimate covers the priming configuration, so it exceeds the budget here";

    const Workspace workspace(static_cast<size_t>(WORKSPACE_BUDGET));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;
    // A workspace skip is not a priming failure: even with the ABORT_ON_PRIMING_FAILURE
    // policy, the run must continue and benchmark the plan unprimed rather than aborting.
    config.primingFailurePolicy = PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), WORKSPACE_BUDGET, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bool foundWorkspaceGrows = false;
    for(const auto& r : results)
    {
        if(r.engineId != ENGINE_WORKSPACE_GROWS_ID)
        {
            continue;
        }
        foundWorkspaceGrows = true;
        EXPECT_TRUE(r.succeeded) << "WorkspaceGrows real plan fits the budget and should benchmark";
        EXPECT_FALSE(r.ranExhaustive)
            << "WorkspaceGrows priming was skipped for workspace, so ranExhaustive must be false";
        EXPECT_TRUE(r.supportsExhaustive)
            << "WorkspaceGrows exposes the benchmarking knob, so it supports exhaustive even "
               "though priming was skipped: supportsExhaustive and ranExhaustive are independent";
        EXPECT_TRUE(r.errorMessage.empty())
            << "A benchmarked-unprimed plan that succeeded carries no benchmark error; the "
               "priming skip belongs in exhaustiveNotRunReason. Got: "
            << r.errorMessage;
        EXPECT_FALSE(r.exhaustiveNotRunReason.empty())
            << "WorkspaceGrows priming compiled workspace exceeds the budget, so an "
               "exhaustiveNotRunReason describing the workspace skip should be attached";
        EXPECT_NE(r.exhaustiveNotRunReason.find("workspace"), std::string::npos)
            << "exhaustiveNotRunReason should mention the workspace skip; got: "
            << r.exhaustiveNotRunReason;
    }

    EXPECT_TRUE(foundWorkspaceGrows) << "WorkspaceGrows engine not found in results";
}

// Test: a workspace sized from get_estimated_max_workspace_size() is large enough for EXHAUSTIVE
// priming, whichever add_*() entry point collected the engine. The estimate covers the
// benchmarking configuration of an engine whose priming plan needs more workspace than its
// default plan.
enum class AddEnginePath
{
    ADD_ENGINES,
    ADD_ENGINE_VARIANTS,
    ADD_ENGINE_SWEEP,
    ADD_ENGINE_CONFIGS,
};

class IntegrationAutotuneExhaustiveEstimate : public IntegrationAutotuneExhaustive,
                                              public ::testing::WithParamInterface<AddEnginePath>
{
};

TEST_P(IntegrationAutotuneExhaustiveEstimate, PrimingRunsWhenWorkspaceSizedFromEstimate)
{
    constexpr int64_t ENGINE_WORKSPACE_GROWS_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineWorkspaceGrows>();

    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    Error result{ErrorCode::OK, ""};
    const auto path = GetParam();
    if(path == AddEnginePath::ADD_ENGINES)
    {
        result = bundle.graph->add_engines({ENGINE_WORKSPACE_GROWS_ID});
    }
    else if(path == AddEnginePath::ADD_ENGINE_VARIANTS)
    {
        result = bundle.graph->add_engine_variants({EngineVariant{ENGINE_WORKSPACE_GROWS_ID, {}}});
    }
    else if(path == AddEnginePath::ADD_ENGINE_SWEEP)
    {
        result
            = bundle.graph->add_engine_sweep({EngineSweepSpec{ENGINE_WORKSPACE_GROWS_ID, {}, {}}});
    }
    else
    {
        std::vector<EngineConfigInfo> configs;
        result = bundle.graph->get_engine_configs(configs);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        configs.erase(std::remove_if(configs.begin(),
                                     configs.end(),
                                     [&](const EngineConfigInfo& config) {
                                         return config.engineId != ENGINE_WORKSPACE_GROWS_ID;
                                     }),
                      configs.end());
        result = bundle.graph->add_engine_configs(configs);
    }
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t estimatedWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(estimatedWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(estimatedWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), estimatedWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    bool foundWorkspaceGrows = false;
    for(const auto& r : results)
    {
        if(r.engineId != ENGINE_WORKSPACE_GROWS_ID)
        {
            continue;
        }
        foundWorkspaceGrows = true;
        EXPECT_TRUE(r.ranExhaustive) << r.exhaustiveNotRunReason;
    }

    EXPECT_TRUE(foundWorkspaceGrows) << "WorkspaceGrows engine not found in results";
}

INSTANTIATE_TEST_SUITE_P(AddEnginePaths,
                         IntegrationAutotuneExhaustiveEstimate,
                         ::testing::Values(AddEnginePath::ADD_ENGINES,
                                           AddEnginePath::ADD_ENGINE_VARIANTS,
                                           AddEnginePath::ADD_ENGINE_SWEEP,
                                           AddEnginePath::ADD_ENGINE_CONFIGS));

// Test: the ABORT_ON_PRIMING_FAILURE policy hard-fails when an engine fails priming.
//
// The test plugin's AutotunePluginEngineFails (-21) fails executeGraph()
// UNCONDITIONALLY so both priming AND benchmark fail and succeeded==false holds.
// With the ABORT_ON_PRIMING_FAILURE policy, the entire autotune() call fails.
TEST_F(IntegrationAutotuneExhaustive, AbortPolicyHardFailsOnPrimingFailure)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;
    config.primingFailurePolicy = PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);

    // The unconditionally-failing engine's priming execution genuinely fails,
    // so EXHAUSTIVE priming returns HIPDNN_BACKEND_ERROR with no winner selected.
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;
}

// Test: a default-constructed config aborts on a recoverable priming failure.
//
// AutotuneConfig's default primingFailurePolicy is ABORT_ON_PRIMING_FAILURE.
// AutotunePluginEnginePrimingOnlyFails (-22) fails execution ONLY during priming
// (benchmarking knob enabled) and would succeed when benchmarked unprimed, so the
// BENCHMARK_UNPRIMED policy tolerates it and returns OK. Isolating this engine makes
// the outcome a true discriminator: under the default policy the priming failure must
// abort the whole autotune() call with HIPDNN_BACKEND_ERROR, proving the default aborts
// on a failure that the alternative policy would have recovered from.
TEST_F(IntegrationAutotuneExhaustive, DefaultPolicyAbortsOnPrimingFailure)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("autotune_exhaustive_test_conv", bundle);

    constexpr int64_t ENGINE_PRIMING_ONLY_FAILS_ID
        = hipdnn_tests::plugin_constants::engineId<AutotunePluginEnginePrimingOnlyFails>();
    auto result = bundle.graph->add_engine(ENGINE_PRIMING_ONLY_FAILS_ID);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);

    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;
}

} // namespace
