// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/TimingStatistics.hpp>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace stats = hipdnn_data_sdk::utilities::detail;

class TestTimingStatistics : public ::testing::Test
{
};

// --- mean -------------------------------------------------------------------

TEST_F(TestTimingStatistics, MeanAveragesSamples)
{
    EXPECT_DOUBLE_EQ(stats::mean(std::vector<double>{2.0, 4.0, 6.0}), 4.0);
    EXPECT_DOUBLE_EQ(stats::mean(std::vector<double>{5.0}), 5.0);
}

TEST_F(TestTimingStatistics, MeanStaysWithinObservedRange)
{
    // The documented invariant callers depend on: rounding must never push the mean
    // outside [min, max]. Near-identical samples are the case that stresses it.
    const std::vector<float> nearIdentical(7, 0.1f);
    const auto result = stats::mean(nearIdentical);
    EXPECT_GE(result, 0.1f);
    EXPECT_LE(result, 0.1f);
}

TEST_F(TestTimingStatistics, MeanRejectsEmptyInput)
{
    EXPECT_THROW(stats::mean(std::vector<double>{}), std::invalid_argument);
}

// --- medianOfSorted ---------------------------------------------------------

TEST_F(TestTimingStatistics, MedianOfSortedHandlesOddAndEvenCounts)
{
    EXPECT_DOUBLE_EQ(stats::medianOfSorted(std::vector<double>{1.0, 2.0, 3.0}), 2.0);
    EXPECT_DOUBLE_EQ(stats::medianOfSorted(std::vector<double>{1.0, 2.0, 3.0, 4.0}), 2.5);
    EXPECT_DOUBLE_EQ(stats::medianOfSorted(std::vector<double>{9.0}), 9.0);
}

TEST_F(TestTimingStatistics, MedianOfSortedRejectsEmptyInput)
{
    EXPECT_THROW(stats::medianOfSorted(std::vector<double>{}), std::invalid_argument);
}

// --- stddev / coefficientOfVariation ----------------------------------------

TEST_F(TestTimingStatistics, StddevIsPopulationNotSample)
{
    // Population stddev of {2,4,4,4,5,5,7,9} is exactly 2.0; the sample (N-1) form is not.
    const std::vector<double> values{2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0};
    EXPECT_DOUBLE_EQ(stats::stddev(values), 2.0);
}

TEST_F(TestTimingStatistics, StddevIsZeroForIdenticalSamples)
{
    EXPECT_DOUBLE_EQ(stats::stddev(std::vector<double>{3.0, 3.0, 3.0}), 0.0);
}

TEST_F(TestTimingStatistics, CoefficientOfVariationScalesByMean)
{
    // Same spread, different magnitude: the ratio must fall as the mean rises.
    const auto tight = stats::coefficientOfVariation(std::vector<double>{99.0, 100.0, 101.0});
    const auto loose = stats::coefficientOfVariation(std::vector<double>{9.0, 10.0, 11.0});
    EXPECT_LT(tight, loose);
}

TEST_F(TestTimingStatistics, CoefficientOfVariationReturnsZeroWhenMeanIsZero)
{
    // All-zero samples: the ratio is undefined and 0 is returned rather than NaN.
    const auto result = stats::coefficientOfVariation(std::vector<double>{0.0, 0.0, 0.0});
    EXPECT_DOUBLE_EQ(result, 0.0);
    EXPECT_FALSE(std::isnan(result));
}

TEST_F(TestTimingStatistics, CoefficientOfVariationRejectsEmptyInput)
{
    EXPECT_THROW(stats::coefficientOfVariation(std::vector<double>{}), std::invalid_argument);
}

// --- robustMean -------------------------------------------------------------

TEST_F(TestTimingStatistics, RobustMeanMatchesPlainMeanWhenNoOutliers)
{
    const std::vector<double> clean{10.0, 10.5, 11.0, 10.2, 10.8};
    EXPECT_NEAR(stats::robustMean(clean), stats::mean(clean), 1e-9);
}

TEST_F(TestTimingStatistics, RobustMeanDiscardsSlowOutlier)
{
    // One contaminated sample must not drag the representative time with it. Samples carry
    // ordinary jitter so the deviation is non-zero and the outlier is actually scorable.
    const std::vector<double> contaminated{10.0, 10.2, 9.9, 10.1, 10.3, 9.8, 500.0};
    const auto robust = stats::robustMean(contaminated);
    EXPECT_NEAR(robust, 10.05, 0.2);
    EXPECT_LT(robust, stats::mean(contaminated));
}

TEST_F(TestTimingStatistics, RobustMeanKeepsFastSamples)
{
    // Only the high side is trimmed: interference cannot make a kernel faster, so an
    // unusually fast sample is signal and must still pull the result down.
    const std::vector<double> withFastSample{10.0, 10.2, 9.9, 10.1, 10.3, 9.8, 1.0};
    EXPECT_LT(stats::robustMean(withFastSample), 9.8);
}

TEST_F(TestTimingStatistics, RobustMeanPrefersConsistentCandidateOverVolatileOne)
{
    // The behaviour the statistic exists for. The volatile candidate has the better single
    // best sample (7.0 vs 9.8) but is usually far slower; ranking must not pick it.
    const std::vector<double> consistent{10.0, 10.1, 9.9, 10.0, 10.2, 9.8, 10.1, 10.0};
    const std::vector<double> volatileCandidate{7.0, 13.0, 14.0, 12.5, 15.0, 13.5, 12.0, 14.5};

    EXPECT_LT(*std::min_element(volatileCandidate.begin(), volatileCandidate.end()),
              *std::min_element(consistent.begin(), consistent.end()));
    EXPECT_LT(stats::robustMean(consistent), stats::robustMean(volatileCandidate));
}

TEST_F(TestTimingStatistics, RobustMeanHandlesZeroDeviation)
{
    // Every sample identical: no measurable spread, so nothing can be scored as an
    // outlier and the plain mean is returned rather than dividing by a zero deviation.
    // Reachable in practice when a short kernel quantises onto one timer tick.
    EXPECT_DOUBLE_EQ(stats::robustMean(std::vector<double>{4.0, 4.0, 4.0, 4.0}), 4.0);
}

TEST_F(TestTimingStatistics, RobustMeanWithZeroDeviationKeepsOutlier)
{
    // Documents the boundary of the zero-deviation case: with over half the samples
    // identical the deviation is still zero, so the outlier cannot be scored and is kept.
    const std::vector<double> values{7.0, 7.0, 7.0, 1000.0};
    EXPECT_DOUBLE_EQ(stats::robustMean(values), stats::mean(values));
}

TEST_F(TestTimingStatistics, RobustMeanLowerThresholdTrimsMore)
{
    const std::vector<double> values{10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 13.0};
    const auto aggressive = stats::robustMean(values, 1.0);
    const auto lenient = stats::robustMean(values, 10.0);
    EXPECT_LT(aggressive, lenient);
    EXPECT_DOUBLE_EQ(lenient, stats::mean(values));
}

TEST_F(TestTimingStatistics, RobustMeanNonPositiveThresholdKeepsEverything)
{
    // A non-positive threshold would score out the median itself and leave nothing to
    // average, so it is treated as "no trimming" instead of returning an empty mean.
    const std::vector<double> values{10.0, 10.0, 10.0, 500.0};
    EXPECT_DOUBLE_EQ(stats::robustMean(values, 0.0), stats::mean(values));
    EXPECT_DOUBLE_EQ(stats::robustMean(values, -1.0), stats::mean(values));
}

TEST_F(TestTimingStatistics, RobustMeanIsOrderIndependent)
{
    const std::vector<double> ascending{1.0, 2.0, 3.0, 4.0, 99.0};
    const std::vector<double> shuffled{99.0, 3.0, 1.0, 4.0, 2.0};
    EXPECT_DOUBLE_EQ(stats::robustMean(ascending), stats::robustMean(shuffled));
}

TEST_F(TestTimingStatistics, RobustMeanHandlesSingleSample)
{
    EXPECT_DOUBLE_EQ(stats::robustMean(std::vector<double>{42.0}), 42.0);
}

TEST_F(TestTimingStatistics, RobustMeanRejectsEmptyInput)
{
    EXPECT_THROW(stats::robustMean(std::vector<double>{}), std::invalid_argument);
}

TEST_F(TestTimingStatistics, RobustMeanWorksForFloatAndDouble)
{
    const std::vector<float> asFloat{10.0f, 10.2f, 9.9f, 10.1f, 10.3f, 9.8f, 500.0f};
    EXPECT_NEAR(stats::robustMean(asFloat), 10.05f, 0.2f);
}
