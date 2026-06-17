// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/BenchmarkStatistics.hpp>
#include <hipdnn_frontend/autotune/CartesianProduct.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace hipdnn_frontend;

// ============================================================================
// PlanSpec Deduplication Tests
// ============================================================================

TEST(TestAutotuneTypes, PlanSpecEqualSameOrder)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    a.workspaceSize = 1024;

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    b.workspaceSize = 2048; // Different workspace — should still be equal

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecEqualDifferentKnobOrder)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentEngineId)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    PlanSpec b;
    b.engineId = 99;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobValues)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{4});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobCount)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecEqualEmptyKnobs)
{
    PlanSpec a;
    a.engineId = 42;

    PlanSpec b;
    b.engineId = 42;

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobNames)
{
    PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("KNOB_A", int64_t{1});

    PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("KNOB_B", int64_t{1});

    EXPECT_NE(a, b);
}

// ============================================================================
// CartesianProduct Tests
// ============================================================================

TEST(TestAutotuneTypes, CartesianProductEmptyAxes)
{
    const std::vector<KnobSweepAxis> axes;
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    // Cartesian product of zero sets is one empty tuple
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].empty());
}

TEST(TestAutotuneTypes, CartesianProductSingleAxis)
{
    const std::vector<KnobSweepAxis> axes = {{"SPLIT_K", {int64_t{1}, int64_t{2}, int64_t{4}}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    ASSERT_EQ(result.size(), 3u);

    EXPECT_EQ(result[0].size(), 1u);
    EXPECT_EQ(result[0][0].knobId(), "SPLIT_K");
    EXPECT_EQ(std::get<int64_t>(result[0][0].value()), 1);

    EXPECT_EQ(std::get<int64_t>(result[1][0].value()), 2);
    EXPECT_EQ(std::get<int64_t>(result[2][0].value()), 4);
}

TEST(TestAutotuneTypes, CartesianProductTwoAxes)
{
    const std::vector<KnobSweepAxis> axes
        = {{"SPLIT_K", {int64_t{1}, int64_t{2}}}, {"TILE_SIZE", {int64_t{64}, int64_t{128}}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    ASSERT_EQ(result.size(), 4u);

    // Each combination should have 2 knob settings
    for(const auto& combo : result)
    {
        EXPECT_EQ(combo.size(), 2u);
    }
}

TEST(TestAutotuneTypes, CartesianProductThreeAxesCorrectCount)
{
    const std::vector<KnobSweepAxis> axes = {{"A", {int64_t{1}, int64_t{2}, int64_t{3}}},
                                             {"B", {int64_t{10}, int64_t{20}}},
                                             {"C", {int64_t{100}, int64_t{200}, int64_t{300}}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    // 3 * 2 * 3 = 18
    EXPECT_EQ(result.size(), 18u);
}

TEST(TestAutotuneTypes, CartesianProductEmptyAxisProducesEmptyResult)
{
    const std::vector<KnobSweepAxis> axes = {
        {"SPLIT_K", {int64_t{1}, int64_t{2}}}, {"TILE_SIZE", {}} // empty values
    };
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_TRUE(result.empty());
}

TEST(TestAutotuneTypes, CartesianProductErrorAtLimit)
{
    // Create axes that would produce > 10,000 combinations
    // 101 * 100 = 10,100 > 10,000
    std::vector<KnobValueVariant> values101;
    values101.reserve(101);
    for(int64_t i = 0; i < 101; ++i)
    {
        values101.emplace_back(i);
    }
    std::vector<KnobValueVariant> values100;
    values100.reserve(100);
    for(int64_t i = 0; i < 100; ++i)
    {
        values100.emplace_back(i);
    }

    const std::vector<KnobSweepAxis> axes = {{"A", values101}, {"B", values100}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.empty());
}

TEST(TestAutotuneTypes, CartesianProductAtExactLimit)
{
    // 100 * 100 = 10,000 — should succeed (limit is >10,000)
    std::vector<KnobValueVariant> values100;
    values100.reserve(100);
    for(int64_t i = 0; i < 100; ++i)
    {
        values100.emplace_back(i);
    }

    const std::vector<KnobSweepAxis> axes = {{"A", values100}, {"B", values100}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(result.size(), 10000u);
}

TEST(TestAutotuneTypes, CartesianProductPreservesMixedValueTypesAndAxisOrder)
{
    const std::vector<KnobSweepAxis> axes = {{"INT_AXIS", {int64_t{1}}},
                                             {"DOUBLE_AXIS", {0.25}},
                                             {"STRING_AXIS", {std::string("fast")}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    ASSERT_EQ(result.size(), 1u);
    ASSERT_EQ(result[0].size(), 3u);
    EXPECT_EQ(result[0][0].knobId(), "INT_AXIS");
    EXPECT_EQ(std::get<int64_t>(result[0][0].value()), 1);
    EXPECT_EQ(result[0][1].knobId(), "DOUBLE_AXIS");
    EXPECT_DOUBLE_EQ(std::get<double>(result[0][1].value()), 0.25);
    EXPECT_EQ(result[0][2].knobId(), "STRING_AXIS");
    EXPECT_EQ(std::get<std::string>(result[0][2].value()), "fast");
}

TEST(TestAutotuneTypes, CartesianProductAboveWarningThresholdStillProducesAllCombinations)
{
    std::vector<KnobValueVariant> values40;
    values40.reserve(40);
    for(int64_t i = 0; i < 40; ++i)
    {
        values40.emplace_back(i);
    }
    std::vector<KnobValueVariant> values30;
    values30.reserve(30);
    for(int64_t i = 0; i < 30; ++i)
    {
        values30.emplace_back(i);
    }

    const std::vector<KnobSweepAxis> axes = {{"A", values40}, {"B", values30}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::computeCartesianProduct(axes, result);

    ASSERT_EQ(error.code, ErrorCode::OK) << error.err_msg;
    ASSERT_EQ(result.size(), 1200u);
    EXPECT_EQ(std::get<int64_t>(result.front()[0].value()), 0);
    EXPECT_EQ(std::get<int64_t>(result.front()[1].value()), 0);
    EXPECT_EQ(std::get<int64_t>(result.back()[0].value()), 39);
    EXPECT_EQ(std::get<int64_t>(result.back()[1].value()), 29);
}

// ============================================================================
// BenchmarkStatistics Tests
// ============================================================================

TEST(TestAutotuneTypes, MeanSingleValue)
{
    const std::vector<float> values = {5.0f};
    EXPECT_FLOAT_EQ(autotune::computeMean(values), 5.0f);
}

TEST(TestAutotuneTypes, MeanMultipleValues)
{
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(autotune::computeMean(values), 3.0f);
}

TEST(TestAutotuneTypes, MeanDoubleValues)
{
    const std::vector<double> values = {2.0, 4.0, 6.0};
    EXPECT_DOUBLE_EQ(autotune::computeMean(values), 4.0);
}

TEST(TestAutotuneTypes, MeanThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::computeMean(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevUniformValues)
{
    // All identical values should have zero standard deviation
    const std::vector<float> values = {3.0f, 3.0f, 3.0f, 3.0f};
    EXPECT_FLOAT_EQ(autotune::computeStddev(values), 0.0f);
}

TEST(TestAutotuneTypes, StddevKnownValues)
{
    // Population stddev of {2, 4, 4, 4, 5, 5, 7, 9}
    // Mean = 40/8 = 5.0
    // Variance = ((2-5)^2 + (4-5)^2 + (4-5)^2 + (4-5)^2 + (5-5)^2 + (5-5)^2 + (7-5)^2 +
    // (9-5)^2) / 8
    //          = (9+1+1+1+0+0+4+16)/8 = 32/8 = 4.0
    // Stddev = sqrt(4) = 2.0
    const std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(autotune::computeStddev(values), 2.0);
}

TEST(TestAutotuneTypes, StddevThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::computeStddev(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, CoVKnownValues)
{
    // Mean = 5.0, Stddev = 2.0, CoV = 2.0/5.0 = 0.4
    const std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(autotune::computeCoefficientOfVariation(values), 0.4);
}

TEST(TestAutotuneTypes, CoVUniformValuesIsZero)
{
    const std::vector<float> values = {7.0f, 7.0f, 7.0f};
    EXPECT_FLOAT_EQ(autotune::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVAllZerosIsZero)
{
    // When mean is 0, CoV returns 0 to avoid division by zero
    const std::vector<float> values = {0.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(autotune::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVThrowsOnEmpty)
{
    const std::vector<double> values;
    EXPECT_THROW(autotune::computeCoefficientOfVariation(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevSingleValue)
{
    // Single value: stddev = 0
    const std::vector<float> values = {42.0f};
    EXPECT_FLOAT_EQ(autotune::computeStddev(values), 0.0f);
}

// ============================================================================
// AutotuneConfig Default Values Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneConfigCustomValues)
{
    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
    config.warmupIterations = 5;
    config.timedIterations = 20;
    config.maxIterations = 200;
    config.windowSize = 10;
    config.stabilityThreshold = 0.02f;
    config.engineIdFilter = {1, 2, 3};
    config.continueOnPrimingFailure = true;

    EXPECT_EQ(config.mode, TuneMode::EXHAUSTIVE);
    EXPECT_EQ(config.strategy, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_EQ(config.warmupIterations, 5);
    EXPECT_EQ(config.timedIterations, 20);
    EXPECT_EQ(config.maxIterations, 200);
    EXPECT_EQ(config.windowSize, 10);
    EXPECT_FLOAT_EQ(config.stabilityThreshold, 0.02f);
    EXPECT_EQ(config.engineIdFilter.size(), 3u);
    EXPECT_TRUE(config.continueOnPrimingFailure);
}

// ============================================================================
// AutotuneResult Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneResultDefaults)
{
    const AutotuneResult result;
    EXPECT_EQ(result.engineId, -1);
    EXPECT_TRUE(result.engineName.empty());
    EXPECT_TRUE(result.knobSettings.empty());
    EXPECT_EQ(result.rank, -1);
    EXPECT_FLOAT_EQ(result.minTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.avgTimeMs, 0.0f);
    EXPECT_FLOAT_EQ(result.stddevMs, 0.0f);
    EXPECT_EQ(result.iterationsRun, 0);
    EXPECT_FALSE(result.succeeded);
    EXPECT_EQ(result.modeUsed, TuneMode::AUTO);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.workspaceSize, 0);
    EXPECT_FALSE(result.ranExhaustive);
    EXPECT_TRUE(result.errorMessage.empty());
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
}

TEST(TestAutotuneTypes, AutotuneResultPopulated)
{
    AutotuneResult result;
    result.engineId = 42;
    result.engineName = "MIOpen_ConvFwd";
    result.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    result.rank = 0;
    result.minTimeMs = 1.5f;
    result.avgTimeMs = 1.8f;
    result.stddevMs = 0.3f;
    result.iterationsRun = 10;
    result.succeeded = true;
    result.modeUsed = TuneMode::EXHAUSTIVE;
    result.converged = true;
    result.workspaceSize = 4096;
    result.ranExhaustive = true;
    result.strategyUsed = AutotuneStrategy::SINGLE_SHOT;

    EXPECT_EQ(result.engineId, 42);
    EXPECT_EQ(result.engineName, "MIOpen_ConvFwd");
    EXPECT_EQ(result.knobSettings.size(), 1u);
    EXPECT_EQ(result.rank, 0);
    EXPECT_FLOAT_EQ(result.minTimeMs, 1.5f);
    EXPECT_FLOAT_EQ(result.avgTimeMs, 1.8f);
    EXPECT_FLOAT_EQ(result.stddevMs, 0.3f);
    EXPECT_EQ(result.iterationsRun, 10);
    EXPECT_TRUE(result.succeeded);
    EXPECT_EQ(result.modeUsed, TuneMode::EXHAUSTIVE);
    EXPECT_TRUE(result.converged);
    EXPECT_EQ(result.workspaceSize, 4096);
    EXPECT_TRUE(result.ranExhaustive);
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::SINGLE_SHOT);
}

// ============================================================================
// AutotuneStorageConfig Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneStorageConfigCustomValues)
{
    AutotuneStorageConfig config;
    config.filePath = "/tmp/autotune_results.json";
    config.deleteAllExistingFileContent = true;

    EXPECT_EQ(config.filePath.string(), "/tmp/autotune_results.json");
    EXPECT_TRUE(config.deleteAllExistingFileContent);
}

// ============================================================================
// AutotuneRankingFn Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneRankingFnIsCallable)
{
    auto sortByAvgTime = [](std::vector<AutotuneResult>& results) {
        std::sort(
            results.begin(), results.end(), [](const AutotuneResult& a, const AutotuneResult& b) {
                return a.avgTimeMs < b.avgTimeMs;
            });
    };
    const AutotuneRankingFn fn = sortByAvgTime;
    EXPECT_NE(fn, nullptr);

    std::vector<AutotuneResult> results(2);
    results[0].avgTimeMs = 2.0f;
    results[1].avgTimeMs = 1.0f;

    fn(results);
    EXPECT_LT(results[0].avgTimeMs, results[1].avgTimeMs);
}

// ============================================================================
// TuneMode and AutotuneStrategy Enum Tests
// ============================================================================

TEST(TestAutotuneTypes, TuneModeValues)
{
    EXPECT_NE(TuneMode::AUTO, TuneMode::EXHAUSTIVE);
}

TEST(TestAutotuneTypes, AutotuneStrategyValues)
{
    EXPECT_NE(AutotuneStrategy::SINGLE_SHOT, AutotuneStrategy::FIXED_AVERAGE);
    EXPECT_NE(AutotuneStrategy::FIXED_AVERAGE, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_NE(AutotuneStrategy::SINGLE_SHOT, AutotuneStrategy::RUN_UNTIL_STABLE);
}

TEST(TestAutotuneTypes, TuneModeToStringUnknownReturnsUnknown)
{
    EXPECT_EQ(tuneModeToString(static_cast<TuneMode>(999)), "UNKNOWN");
}

TEST(TestAutotuneTypes, AutotuneStrategyToStringUnknownReturnsUnknown)
{
    EXPECT_EQ(strategyToString(static_cast<AutotuneStrategy>(999)), "UNKNOWN");
}

// ============================================================================
// EngineVariant Tests
// ============================================================================

TEST(TestAutotuneTypes, EngineVariantConstruction)
{
    EngineVariant variant;
    variant.engineId = 42;
    variant.knobSettings["SPLIT_K"] = int64_t{2};
    variant.knobSettings["TILE_SIZE"] = int64_t{128};

    EXPECT_EQ(variant.engineId, 42);
    EXPECT_EQ(variant.knobSettings.size(), 2u);
    EXPECT_EQ(std::get<int64_t>(variant.knobSettings.at("SPLIT_K")), 2);
}

// ============================================================================
// KnobSweepAxis Tests
// ============================================================================

TEST(TestAutotuneTypes, KnobSweepAxisConstruction)
{
    KnobSweepAxis axis;
    axis.knobId = "SPLIT_K";
    axis.values = {int64_t{1}, int64_t{2}, int64_t{4}};

    EXPECT_EQ(axis.knobId, "SPLIT_K");
    EXPECT_EQ(axis.values.size(), 3u);
}

// ============================================================================
// EngineSweepSpec Tests
// ============================================================================

TEST(TestAutotuneTypes, EngineSweepSpecConstruction)
{
    EngineSweepSpec spec;
    spec.engineId = 42;
    spec.axes = {{"SPLIT_K", {int64_t{1}, int64_t{2}}}};
    spec.fixedSettings["REDUCTION_MODE"] = int64_t{1};

    EXPECT_EQ(spec.engineId, 42);
    EXPECT_EQ(spec.axes.size(), 1u);
    EXPECT_EQ(spec.fixedSettings.size(), 1u);
}
