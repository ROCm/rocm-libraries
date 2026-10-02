// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_plugin_sdk/heuristics/uhd/ScoreTransform.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

/// @file TestScoreTransform.cpp
/// @brief Inverting the target transform a UHD was trained under. A wrong inverse keeps the
/// ranking but corrupts the score that RFC 0019 §11.3 compares across engines.
namespace hipdnn_plugin_sdk::uhd
{
namespace
{

TEST(TestIngestorScoreTransform, EachTransformInvertsToTheOriginalTarget)
{
    // Round-tripped: what must hold is that applyInverse undoes the trainer's transform.
    constexpr double TARGET = 137.25;

    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::log1p(TARGET), "log1p"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::log(TARGET), "log"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::sqrt(TARGET), "sqrt"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::exp(TARGET), "exp"), TARGET);
}

TEST(TestIngestorScoreTransform, AnUntransformedTargetPassesThroughUnchanged)
{
    // "" and "identity" are the same declaration.
    constexpr double RAW = 42.5;
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(RAW, ""), RAW);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(RAW, "identity"), RAW);
}

TEST(TestIngestorScoreTransform, AnUnrecognisedTransformDoesNotSilentlyPassThrough)
{
    // An unknown name would otherwise fall through to identity and report a log-scale number
    // as TFLOPS.
    EXPECT_FALSE(score_transform::isSupported("log10"));
    EXPECT_FALSE(score_transform::isSupported("Log1p")) << "the match must be exact";
    EXPECT_FALSE(score_transform::isSupported("boxcox"));

    for(const auto* known : score_transform::SUPPORTED_TRANSFORMS)
    {
        EXPECT_TRUE(score_transform::isSupported(known)) << "declared but not accepted: " << known;
    }
}

TEST(TestIngestorScoreTransform, TheDiagnosticListsEveryTransformThatIsAccepted)
{
    const auto list = score_transform::supportedTransformList();
    for(const auto* known : score_transform::SUPPORTED_TRANSFORMS)
    {
        const std::string name = (*known == '\0') ? "\"\"" : known;
        EXPECT_NE(list.find(name), std::string::npos) << "missing from the diagnostic: " << name;
    }
}

TEST(TestIngestorScoreTransform, TheInverseIsMonotoneSoRankingSurvivesIt)
{
    // Sampled across zero: a raw GBDT score is unbounded, and an inverse like raw*raw is
    // monotone only on the positives.
    for(const auto* transform : score_transform::SUPPORTED_TRANSFORMS)
    {
        for(const double lower : {-2.5, -0.5, 0.25, 1.5})
        {
            const double a = score_transform::applyInverse(lower, transform);
            const double b = score_transform::applyInverse(lower + 1.0, transform);

            // NaN means "no measurement"; only in-domain pairs must be ordered.
            if(std::isnan(a) || std::isnan(b))
            {
                continue;
            }
            EXPECT_LT(a, b) << "not order-preserving: " << transform << " at " << lower;
        }
    }
}

TEST(TestIngestorScoreTransform, AnOutOfDomainPredictionIsReportedAsNaNRatherThanAWrongNumber)
{
    // A raw GBDT score can leave an inverse's domain; NaN lets the caller apply RFC 0019 §5
    // step 7's "no measurement" rule instead of ranking a fabricated score.
    EXPECT_TRUE(std::isnan(score_transform::applyInverse(-0.5, "exp"))) // log of a negative
        << "exp's inverse produced a number outside its domain";
    EXPECT_TRUE(std::isnan(score_transform::applyInverse(-0.5, "sqrt"))) // squaring flips sign
        << "sqrt's inverse mapped a negative prediction to a positive score";

    EXPECT_DOUBLE_EQ(score_transform::applyInverse(4.0, "sqrt"), 16.0);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(0.0, "sqrt"), 0.0);
}

} // namespace
} // namespace hipdnn_plugin_sdk::uhd
