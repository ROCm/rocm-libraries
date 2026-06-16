// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration test for EXHAUSTIVE autotune mode.
// Verifies that EXHAUSTIVE mode primes engine caches via the global.benchmarking
// knob and that AUTO mode does not.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "../test_plugins/TestPluginEngineIdMap.hpp"
#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

using IntegrationAutotuneExhaustive = hipdnn_tests::AutotuneIntegrationFixture;

// Test: EXHAUSTIVE mode with continueOnPrimingFailure=true verifies per-engine behavior.
//
// This consolidated test verifies the RFC's autotune behavior for all engine types:
// - Engine A (main): has benchmarking knob, priming succeeds, benchmark succeeds
//   → ranExhaustive=true, succeeded=true
// - Engine B: no benchmarking knob, priming skipped, benchmark succeeds
//   → ranExhaustive=false, succeeded=true
// - Engine C: no benchmarking knob, priming skipped, benchmark succeeds
//   → ranExhaustive=false, succeeded=true
// - EngineFails: has benchmarking knob, fails UNCONDITIONALLY (both priming AND benchmark)
//   → ranExhaustive=false, succeeded=false
// - EnginePrimingOnlyFails: has benchmarking knob, priming fails, benchmark succeeds
//   → ranExhaustive=false, succeeded=true, errorMessage contains priming failure
//
// RFC § 6.4: "When continueOnPrimingFailure is true and priming fails, the engine
// is still benchmarked (unprimed). Its AutotuneResult::ranExhaustive is false,
// and errorMessage notes the priming failure even though succeeded may be true."
TEST_F(IntegrationAutotuneExhaustive, ExhaustiveModeWithContinueOnPrimingFailure)
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
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.continueOnPrimingFailure = true;

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
        }
        else if(r.engineId == ENGINE_B_ID || r.engineId == ENGINE_C_ID)
        {
            // Engines B and C: no benchmarking knob, priming skipped
            EXPECT_TRUE(r.succeeded) << "Engines B/C should succeed";
            EXPECT_FALSE(r.ranExhaustive)
                << "Engines B/C have no benchmarking knob, should not run exhaustive";
        }
        else if(r.engineId == ENGINE_FAILS_ID)
        {
            foundEngineFails = true;
            // EngineFails: fails UNCONDITIONALLY (both priming AND benchmark)
            EXPECT_FALSE(r.succeeded) << "EngineFails must not succeed";
            EXPECT_FALSE(r.ranExhaustive)
                << "EngineFails priming failed, so ranExhaustive should be false";
        }
        else if(r.engineId == ENGINE_PRIMING_ONLY_FAILS_ID)
        {
            foundEnginePrimingOnlyFails = true;
            // EnginePrimingOnlyFails: priming fails, but benchmark succeeds
            // This tests the RFC case "succeeded may be true"
            EXPECT_TRUE(r.succeeded)
                << "EnginePrimingOnlyFails benchmark should succeed despite priming failure";
            EXPECT_FALSE(r.ranExhaustive)
                << "EnginePrimingOnlyFails priming failed, so ranExhaustive should be false";
            EXPECT_FALSE(r.errorMessage.empty())
                << "EnginePrimingOnlyFails should have errorMessage noting priming failure";
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

// Test: AUTO mode does not set ranExhaustive on any engine
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
    config.mode = TuneMode::AUTO;
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    ASSERT_FALSE(results.empty());
    for(const auto& r : results)
    {
        EXPECT_FALSE(r.ranExhaustive)
            << "Engine " << r.engineId << " should not have ran exhaustive in AUTO mode";
    }
}

// Test: continueOnPrimingFailure=false hard-fails when an engine fails priming.
//
// The test plugin's AutotunePluginEngineFails (-21) fails executeGraph()
// UNCONDITIONALLY so both priming AND benchmark fail and succeeded==false holds.
// With continueOnPrimingFailure=false, the entire autotune() call fails.
TEST_F(IntegrationAutotuneExhaustive, ContinueOnPrimingFailureFalseHardFails)
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
    config.strategy = AutotuneStrategy::SINGLE_SHOT;
    config.warmupIterations = 1;
    config.continueOnPrimingFailure = false;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);

    // The unconditionally-failing engine's priming execution genuinely fails,
    // so EXHAUSTIVE priming returns HIPDNN_BACKEND_ERROR with no winner selected.
    EXPECT_EQ(result.code, ErrorCode::HIPDNN_BACKEND_ERROR) << result.err_msg;
}

} // namespace
