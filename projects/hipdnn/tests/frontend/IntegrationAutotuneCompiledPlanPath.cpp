// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Integration tests for the compiled-plan autotune path:
// create_execution_plans() -> build_plans(ALL) -> autotune() -> execute().
// Also tests the manual benchmark loop using plan-indexed access APIs.

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

using IntegrationAutotuneCompiledPlanPath = hipdnn_tests::AutotuneIntegrationFixture;

// Test: compiled-plan path autotune end-to-end workflow
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneEndToEnd)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    ASSERT_GE(maxWs, 0);

    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    assertAnySucceeded(results, "No engine succeeded during compiled-plan autotune");

    buildWorkspaceAndExecute(bundle);
}

// Test: EXHAUSTIVE mode is rejected on the compiled-plan path
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneExhaustiveBlocked)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    EXPECT_EQ(result.code, ErrorCode::INVALID_VALUE) << result.err_msg;
    EXPECT_NE(result.err_msg.find("EXHAUSTIVE mode is not supported"), std::string::npos)
        << result.err_msg;
}

// Test: manual benchmark loop using plan-indexed access APIs
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopEndToEnd)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GT(planCount, 0);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    int64_t bestIndex = -1;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            bestIndex = i;
            break;
        }
    }

    ASSERT_GE(bestIndex, 0) << "No plan succeeded during manual loop";

    result = bundle.graph->build_plan_at_index(bestIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    buildWorkspaceAndExecute(bundle);
}

// Test: multi-engine plugin provides multiple plans and autotune benchmarks them
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // With multi-engine plugin, should have multiple plans
    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Multiple engines should produce multiple results
    ASSERT_GE(results.size(), 2u) << "Autotune should benchmark >= 2 engines";

    assertAnySucceeded(results, "At least one engine should succeed");

    buildWorkspaceAndExecute(bundle);
}

// Test: build_plans(ALL) compiles multiple engines and they are executable
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllCompilesMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Verify multiple plans are actually executable
    int successCount = 0;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            ++successCount;
        }
    }
    ASSERT_GE(successCount, 2) << "build_plans(ALL) should compile multiple executable plans";
}

// Test: build_plans(ALL) compiles more plans than HEURISTICS_CHOICE
TEST_F(IntegrationAutotuneCompiledPlanPath, BuildPlansAllVsHeuristicsChoice)
{
    // Graph A: build_plans(HEURISTICS_CHOICE) — only compiles the active plan
    ConvGraphBundle bundleA;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundleA);
    auto result = bundleA.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleA.graph->build_plans(BuildPlanPolicy::HEURISTICS_CHOICE);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCountA = bundleA.graph->get_execution_plan_count();
    ASSERT_GE(planCountA, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWsA = bundleA.graph->get_autotune_workspace_size();
    const Workspace workspaceA(static_cast<size_t>(maxWsA));

    int successCountA = 0;
    for(int64_t i = 0; i < planCountA; ++i)
    {
        auto execResult = bundleA.graph->execute_plan_at_index(
            _handle, bundleA.variantPack, workspaceA.get(), i);
        if(execResult.is_good())
        {
            ++successCountA;
        }
    }

    // Graph B: build_plans(ALL) — compiles all plans
    ConvGraphBundle bundleB;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundleB);
    result = bundleB.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    result = bundleB.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCountB = bundleB.graph->get_execution_plan_count();
    const int64_t maxWsB = bundleB.graph->get_autotune_workspace_size();
    const Workspace workspaceB(static_cast<size_t>(maxWsB));

    int successCountB = 0;
    for(int64_t i = 0; i < planCountB; ++i)
    {
        auto execResult = bundleB.graph->execute_plan_at_index(
            _handle, bundleB.variantPack, workspaceB.get(), i);
        if(execResult.is_good())
        {
            ++successCountB;
        }
    }

    // ALL should compile more plans than HEURISTICS_CHOICE
    ASSERT_GT(successCountB, successCountA)
        << "build_plans(ALL) should compile more plans than HEURISTICS_CHOICE";
}

// Test: manual benchmark loop iterates over multiple engines
TEST_F(IntegrationAutotuneCompiledPlanPath, ManualLoopMultipleEngines)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    const int64_t planCount = bundle.graph->get_execution_plan_count();
    ASSERT_GE(planCount, 2) << "Multi-engine plugin should provide >= 2 plans";

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Iterate all plans, track all successes
    std::vector<int64_t> successIndices;
    for(int64_t i = 0; i < planCount; ++i)
    {
        auto execResult
            = bundle.graph->execute_plan_at_index(_handle, bundle.variantPack, workspace.get(), i);
        if(execResult.is_good())
        {
            successIndices.push_back(i);
        }
    }

    ASSERT_GE(successIndices.size(), 2u) << "At least 2 plans should succeed during manual loop";

    // Select a non-zero index if possible
    const int64_t selectedIndex = successIndices.back();

    result = bundle.graph->build_plan_at_index(selectedIndex);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    buildWorkspaceAndExecute(bundle);
}

// Test: compiled-plan path autotune with engine ID filter selects specific engine
TEST_F(IntegrationAutotuneCompiledPlanPath, CompiledPlanAutotuneWithEngineIdFilter)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    auto result = bundle.graph->create_execution_plans({HeuristicMode::FALLBACK});
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = bundle.graph->build_plans(BuildPlanPolicy::ALL);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Verify we have multiple plans before filtering
    ASSERT_GE(bundle.graph->get_execution_plan_count(), 2);

    const int64_t maxWs = bundle.graph->get_autotune_workspace_size();
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Filter to only engine B
    const int64_t engineBId = hipdnn_tests::plugin_constants::engineId<AutotunePluginEngineB>();
    // Engine A exposes the global.benchmarking knob; engine B does not. The
    // compiled-plan path computes supportsExhaustive per plan from the engine's
    // knobs, so it must be reported on both the benchmarked winner (B) and the
    // filtered-out candidates (A) alike.
    const int64_t engineAId = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;
    config.engineIdFilter = {engineBId};

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Every candidate plan produces a result. The filtered-in engine (B) is the
    // only one benchmarked and ranked; the excluded plans surface as filtered
    // failed results carrying the engineIdFilter reason.
    int benchmarkedEngineB = 0;
    int excludedByFilter = 0;
    bool sawFilteredEngineA = false;
    for(const auto& r : results)
    {
        // Priming never runs on the compiled-plan path (EXHAUSTIVE is rejected),
        // so every result reports ranExhaustive=false with no priming reason.
        EXPECT_FALSE(r.ranExhaustive);
        EXPECT_TRUE(r.exhaustiveNotRunReason.empty()) << r.exhaustiveNotRunReason;

        if(r.engineId == engineBId && r.succeeded)
        {
            ++benchmarkedEngineB;
            // Engine B exposes no global.benchmarking knob.
            EXPECT_FALSE(r.supportsExhaustive)
                << "Engine B has no benchmarking knob, so supportsExhaustive is false";
        }
        else
        {
            ++excludedByFilter;
            EXPECT_FALSE(r.succeeded);
            EXPECT_NE(r.engineId, engineBId) << "Engine B should not be filtered out";
            EXPECT_NE(r.errorMessage.find("engineIdFilter"), std::string::npos) << r.errorMessage;
            if(r.engineId == engineAId)
            {
                sawFilteredEngineA = true;
                // A filtered-out engine still reports its real benchmarking
                // capability; engine A exposes global.benchmarking.
                EXPECT_TRUE(r.supportsExhaustive)
                    << "Filtered-out engine A still reports supportsExhaustive=true";
            }
        }
    }
    EXPECT_EQ(benchmarkedEngineB, 1) << "Filter should benchmark exactly 1 engine (B)";
    EXPECT_GT(excludedByFilter, 0) << "Excluded plans should still surface as filtered results";
    EXPECT_TRUE(sawFilteredEngineA)
        << "Engine A should appear as a filtered result carrying its capability";

    buildWorkspaceAndExecute(bundle);
}

// Test: plan-spec path autotune with engine ID filter selects specific engine
TEST_F(IntegrationAutotuneCompiledPlanPath, PlanSpecAutotuneWithEngineIdFilter)
{
    ConvGraphBundle bundle;
    createBuiltConvGraph("compiled_plan_path_test_conv", bundle);

    // Use plan-spec path (add_all_engines) instead of compiled-plan path
    auto result = bundle.graph->add_all_engines();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    int64_t maxWs = 0;
    result = bundle.graph->get_estimated_max_workspace_size(maxWs);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    const Workspace workspace(static_cast<size_t>(maxWs));

    // Filter to only engine A (the original autotune engine)
    const int64_t engineAId = hipdnn_tests::plugin_constants::engineId<AutotunePlugin>();

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = AutotuneStrategy::FIXED_AVERAGE;
    config.timedIterations = 1;
    config.warmupIterations = 1;
    config.engineIdFilter = {engineAId};

    std::vector<AutotuneResult> results;
    result = bundle.graph->autotune(
        _handle, bundle.variantPack, workspace.get(), maxWs, config, {}, &results);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_FALSE(results.empty()) << "No results from engine A found";

    // Only engine A's plan specs are benchmarked (succeeded); any plan spec from
    // another engine is excluded by the filter and surfaces as a filtered failed
    // result carrying the engineIdFilter reason.
    bool sawSucceededEngineA = false;
    for(const auto& r : results)
    {
        if(r.engineId == engineAId)
        {
            if(r.succeeded)
            {
                sawSucceededEngineA = true;
            }
        }
        else
        {
            EXPECT_FALSE(r.succeeded) << "Filtered-out engine should not succeed";
            EXPECT_NE(r.errorMessage.find("engineIdFilter"), std::string::npos) << r.errorMessage;
        }
    }
    EXPECT_TRUE(sawSucceededEngineA) << "Engine A should be benchmarked and succeed";

    buildWorkspaceAndExecute(bundle);
}

} // namespace
