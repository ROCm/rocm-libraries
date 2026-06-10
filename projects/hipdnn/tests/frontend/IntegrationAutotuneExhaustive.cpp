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

#include "AutotuneIntegrationFixture.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_data_sdk::utilities;

namespace
{

using IntegrationAutotuneExhaustive = hipdnn_tests::AutotuneIntegrationFixture;

// Test: EXHAUSTIVE autotune sets ranExhaustive = true for engines with benchmarking knob
TEST_F(IntegrationAutotuneExhaustive, ExhaustiveModeRunsCachePriming)
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

    // Verify at least one engine succeeded
    ASSERT_FALSE(results.empty());
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
    }
    ASSERT_TRUE(anySucceeded) << "No engine succeeded during EXHAUSTIVE autotune";
    EXPECT_TRUE(anyRanExhaustive) << "No engine ran exhaustive priming (expected at least one "
                                     "since the test plugin has global.benchmarking knob)";
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
// A priming-only failure would leave succeeded==true ("...even though succeeded
// may be true") — a different scenario.
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

// Test: continueOnPrimingFailure=true tolerates the failing engine; it appears
// as a result entry with succeeded==false AND ranExhaustive==false, while a
// non-failing engine wins.
TEST_F(IntegrationAutotuneExhaustive, ContinueOnPrimingFailureTrueToleratesFailingEngine)
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

    bool anySucceeded = false;
    bool failingEngineFoundAndFailed = false;
    constexpr int64_t FAILING_ENGINE_ID = -21;
    for(const auto& r : results)
    {
        if(r.succeeded)
        {
            anySucceeded = true;
        }
        if(r.engineId == FAILING_ENGINE_ID)
        {
            EXPECT_FALSE(r.succeeded) << "Failing engine must not succeed";
            EXPECT_FALSE(r.ranExhaustive) << "Failing engine benchmarked-and-failed, not skipped";
            failingEngineFoundAndFailed = !r.succeeded;
        }
    }
    EXPECT_TRUE(anySucceeded) << "A non-failing engine should win";
    EXPECT_TRUE(failingEngineFoundAndFailed)
        << "The unconditionally-failing engine should appear as a failed result";
}

} // namespace
