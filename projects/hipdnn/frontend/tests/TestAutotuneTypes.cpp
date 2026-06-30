// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_frontend/autotune/AutotuneTypes.hpp>
#include <hipdnn_frontend/autotune/BenchmarkStatistics.hpp>
#include <hipdnn_frontend/autotune/CartesianProduct.hpp>
#include <hipdnn_frontend/autotune/KnobConstants.hpp>
#include <hipdnn_frontend/autotune/PlanSpec.hpp>

#include <cmath>
#include <string>
#include <vector>

using namespace hipdnn_frontend;

// ============================================================================
// PlanSpec Deduplication Tests
// ============================================================================

TEST(TestAutotuneTypes, PlanSpecEqualSameOrder)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    a.workspaceSize = 1024;

    autotune::detail::PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    b.workspaceSize = 2048; // Different workspace - should still be equal

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecEqualDifferentKnobOrder)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    autotune::detail::PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    b.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentEngineId)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    autotune::detail::PlanSpec b;
    b.engineId = 99;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobValues)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    autotune::detail::PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{4});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobCount)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("SPLIT_K", int64_t{2});
    a.knobSettings.emplace_back("TILE_SIZE", int64_t{128});

    autotune::detail::PlanSpec b;
    b.engineId = 42;
    b.knobSettings.emplace_back("SPLIT_K", int64_t{2});

    EXPECT_NE(a, b);
}

TEST(TestAutotuneTypes, PlanSpecEqualEmptyKnobs)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;

    autotune::detail::PlanSpec b;
    b.engineId = 42;

    EXPECT_EQ(a, b);
}

TEST(TestAutotuneTypes, PlanSpecNotEqualDifferentKnobNames)
{
    autotune::detail::PlanSpec a;
    a.engineId = 42;
    a.knobSettings.emplace_back("KNOB_A", int64_t{1});

    autotune::detail::PlanSpec b;
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

    auto error = autotune::detail::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    // Cartesian product of zero sets is one empty tuple
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].empty());
}

TEST(TestAutotuneTypes, CartesianProductSingleAxis)
{
    const std::vector<KnobSweepAxis> axes = {{"SPLIT_K", {int64_t{1}, int64_t{2}, int64_t{4}}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::detail::computeCartesianProduct(axes, result);
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

    auto error = autotune::detail::computeCartesianProduct(axes, result);
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

    auto error = autotune::detail::computeCartesianProduct(axes, result);
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

    auto error = autotune::detail::computeCartesianProduct(axes, result);
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

    auto error = autotune::detail::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(result.empty());
}

TEST(TestAutotuneTypes, CartesianProductAtExactLimit)
{
    // 100 * 100 = 10,000 - should succeed (limit is >10,000)
    std::vector<KnobValueVariant> values100;
    values100.reserve(100);
    for(int64_t i = 0; i < 100; ++i)
    {
        values100.emplace_back(i);
    }

    const std::vector<KnobSweepAxis> axes = {{"A", values100}, {"B", values100}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::detail::computeCartesianProduct(axes, result);
    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(result.size(), 10000u);
}

TEST(TestAutotuneTypes, CartesianProductPreservesMixedValueTypesAndAxisOrder)
{
    const std::vector<KnobSweepAxis> axes = {{"INT_AXIS", {int64_t{1}}},
                                             {"DOUBLE_AXIS", {0.25}},
                                             {"STRING_AXIS", {std::string("fast")}}};
    std::vector<std::vector<KnobSetting>> result;

    auto error = autotune::detail::computeCartesianProduct(axes, result);

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

    auto error = autotune::detail::computeCartesianProduct(axes, result);

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
    EXPECT_FLOAT_EQ(autotune::detail::computeMean(values), 5.0f);
}

TEST(TestAutotuneTypes, MeanMultipleValues)
{
    const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    EXPECT_FLOAT_EQ(autotune::detail::computeMean(values), 3.0f);
}

TEST(TestAutotuneTypes, MeanDoubleValues)
{
    const std::vector<double> values = {2.0, 4.0, 6.0};
    EXPECT_DOUBLE_EQ(autotune::detail::computeMean(values), 4.0);
}

TEST(TestAutotuneTypes, MeanThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::detail::computeMean(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevUniformValues)
{
    // All identical values should have zero standard deviation
    const std::vector<float> values = {3.0f, 3.0f, 3.0f, 3.0f};
    EXPECT_FLOAT_EQ(autotune::detail::computeStddev(values), 0.0f);
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
    EXPECT_DOUBLE_EQ(autotune::detail::computeStddev(values), 2.0);
}

TEST(TestAutotuneTypes, StddevThrowsOnEmpty)
{
    const std::vector<float> values;
    EXPECT_THROW(autotune::detail::computeStddev(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, CoVKnownValues)
{
    // Mean = 5.0, Stddev = 2.0, CoV = 2.0/5.0 = 0.4
    const std::vector<double> values = {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(autotune::detail::computeCoefficientOfVariation(values), 0.4);
}

TEST(TestAutotuneTypes, CoVUniformValuesIsZero)
{
    const std::vector<float> values = {7.0f, 7.0f, 7.0f};
    EXPECT_FLOAT_EQ(autotune::detail::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVAllZerosIsZero)
{
    // When mean is 0, CoV returns 0 to avoid division by zero
    const std::vector<float> values = {0.0f, 0.0f, 0.0f};
    EXPECT_FLOAT_EQ(autotune::detail::computeCoefficientOfVariation(values), 0.0f);
}

TEST(TestAutotuneTypes, CoVThrowsOnEmpty)
{
    const std::vector<double> values;
    EXPECT_THROW(autotune::detail::computeCoefficientOfVariation(values), std::invalid_argument);
}

TEST(TestAutotuneTypes, StddevSingleValue)
{
    // Single value: stddev = 0
    const std::vector<float> values = {42.0f};
    EXPECT_FLOAT_EQ(autotune::detail::computeStddev(values), 0.0f);
}

// ============================================================================
// AutotuneConfig Default Values Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneConfigDefaultValues)
{
    const AutotuneConfig config;

    EXPECT_EQ(config.mode, TuneMode::STANDARD);
    EXPECT_EQ(config.strategy, AutotuneStrategy::RUN_UNTIL_STABLE);
    EXPECT_EQ(config.warmupIterations, 1);
    EXPECT_EQ(config.timedIterations, 10);
    EXPECT_EQ(config.maxIterations, 100);
    EXPECT_EQ(config.windowSize, 3);
    EXPECT_FLOAT_EQ(config.stabilityThreshold, 0.05f);
    EXPECT_TRUE(config.engineIdFilter.empty());
    EXPECT_EQ(config.rankingFn, nullptr);
    EXPECT_EQ(config.primingFailurePolicy, PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE);
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
    EXPECT_EQ(result.modeUsed, TuneMode::STANDARD);
    EXPECT_FALSE(result.converged);
    EXPECT_EQ(result.workspaceSize, 0);
    EXPECT_FALSE(result.supportsExhaustive);
    EXPECT_FALSE(result.ranExhaustive);
    EXPECT_TRUE(result.exhaustiveNotRunReason.empty());
    EXPECT_TRUE(result.errorMessage.empty());
    EXPECT_EQ(result.strategyUsed, AutotuneStrategy::RUN_UNTIL_STABLE);
}

// ============================================================================
// AutotuneStorageConfig Tests
// ============================================================================

TEST(TestAutotuneTypes, AutotuneStorageConfigDefaults)
{
    const AutotuneStorageConfig config;

    EXPECT_TRUE(config.filePath.empty());
    EXPECT_FALSE(config.deleteAllExistingFileContent);
}

// ============================================================================
// TuneMode and AutotuneStrategy Enum Tests
// ============================================================================

TEST(TestAutotuneTypes, TuneModeValues)
{
    EXPECT_NE(TuneMode::STANDARD, TuneMode::EXHAUSTIVE);
}

TEST(TestAutotuneTypes, AutotuneStrategyValues)
{
    EXPECT_NE(AutotuneStrategy::FIXED_AVERAGE, AutotuneStrategy::RUN_UNTIL_STABLE);
}

TEST(TestAutotuneTypes, TuneModeToStringMapsEachValue)
{
    EXPECT_EQ(tuneModeToString(TuneMode::STANDARD), "STANDARD");
    EXPECT_EQ(tuneModeToString(TuneMode::EXHAUSTIVE), "EXHAUSTIVE");
}

TEST(TestAutotuneTypes, TuneModeToStringUnknownReturnsUnknown)
{
    EXPECT_EQ(tuneModeToString(static_cast<TuneMode>(999)), "UNKNOWN");
}

TEST(TestAutotuneTypes, TuneModeToLowerStringMapsEachValue)
{
    EXPECT_EQ(tuneModeToLowerString(TuneMode::STANDARD), "standard");
    EXPECT_EQ(tuneModeToLowerString(TuneMode::EXHAUSTIVE), "exhaustive");
}

TEST(TestAutotuneTypes, TuneModeToLowerStringUnknownReturnsUnknown)
{
    EXPECT_EQ(tuneModeToLowerString(static_cast<TuneMode>(999)), "unknown");
}

TEST(TestAutotuneTypes, AutotuneStrategyToStringMapsEachValue)
{
    EXPECT_EQ(strategyToString(AutotuneStrategy::FIXED_AVERAGE), "FIXED_AVERAGE");
    EXPECT_EQ(strategyToString(AutotuneStrategy::RUN_UNTIL_STABLE), "RUN_UNTIL_STABLE");
}

TEST(TestAutotuneTypes, AutotuneStrategyToStringUnknownReturnsUnknown)
{
    EXPECT_EQ(strategyToString(static_cast<AutotuneStrategy>(999)), "UNKNOWN");
}

TEST(TestAutotuneTypes, AutotuneStrategyToLowerStringMapsEachValue)
{
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::FIXED_AVERAGE), "fixed_average");
    EXPECT_EQ(strategyToLowerString(AutotuneStrategy::RUN_UNTIL_STABLE), "run_until_stable");
}

TEST(TestAutotuneTypes, AutotuneStrategyToLowerStringUnknownReturnsUnknown)
{
    EXPECT_EQ(strategyToLowerString(static_cast<AutotuneStrategy>(999)), "unknown");
}

TEST(TestAutotuneTypes, PrimingFailurePolicyToStringMapsEachValue)
{
    EXPECT_EQ(primingFailurePolicyToString(PrimingFailurePolicy::ABORT_ON_PRIMING_FAILURE),
              "ABORT_ON_PRIMING_FAILURE");
    EXPECT_EQ(primingFailurePolicyToString(PrimingFailurePolicy::BENCHMARK_UNPRIMED),
              "BENCHMARK_UNPRIMED");
}

TEST(TestAutotuneTypes, PrimingFailurePolicyToStringUnknownReturnsUnknown)
{
    EXPECT_EQ(primingFailurePolicyToString(static_cast<PrimingFailurePolicy>(999)), "UNKNOWN");
}
