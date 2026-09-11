// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <hipdnn_plugin_sdk/ingestor/uhd/ScoreTransform.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <string>

/// @file TestScoreTransform.cpp
/// @brief The inverse of the target transform a UHD was trained under.
///
/// uhd_gen trains on `log1p(target)` because the TFLOPS distribution is long-tailed, and records
/// which transform it used. The runtime has to undo it before the score means TFLOPS again.
/// Getting this wrong does not break ranking -- every monotone transform preserves the order --
/// so the kernel selected stays correct and only the *number* is wrong. That number is what
/// RFC 0019 §11.3 compares across engines and what §15.2 hands back as a figure of merit, so
/// the error surfaces as an engine chosen over another for a reason no test asserts on.
namespace hipdnn_plugin_sdk::ingestor::uhd
{
namespace
{

TEST(TestIngestorScoreTransform, EachTransformInvertsToTheOriginalTarget)
{
    // Round-tripped rather than compared against a hardcoded expectation: what has to hold is
    // that applyInverse undoes what the trainer applied, and a table of constants would restate
    // the implementation instead of the relationship.
    constexpr double TARGET = 137.25;

    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::log1p(TARGET), "log1p"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::log(TARGET), "log"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::sqrt(TARGET), "sqrt"), TARGET);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(std::exp(TARGET), "exp"), TARGET);
}

TEST(TestIngestorScoreTransform, AnUntransformedTargetPassesThroughUnchanged)
{
    // "" and "identity" are the same declaration. A UHD omitting the field must not have its
    // score altered, which is the case that would otherwise go unnoticed: most models are
    // untransformed, so a wrong default would be wrong everywhere at once and look uniform.
    constexpr double RAW = 42.5;
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(RAW, ""), RAW);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(RAW, "identity"), RAW);
}

TEST(TestIngestorScoreTransform, AnUnrecognisedTransformDoesNotSilentlyPassThrough)
{
    // isSupported is what the loader checks. Without that check an unknown name would reach
    // applyInverse, fall through to the identity branch, and report a log-scale number as
    // TFLOPS -- a value ~100x too small, still positive, still ordered correctly.
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
    // The message an author sees after a rejection. It is generated from the same array
    // isSupported reads, so a transform can never be accepted without appearing in the list or
    // advertised without being accepted.
    const auto list = score_transform::supportedTransformList();
    for(const auto* known : score_transform::SUPPORTED_TRANSFORMS)
    {
        const std::string name = (*known == '\0') ? "\"\"" : known;
        EXPECT_NE(list.find(name), std::string::npos) << "missing from the diagnostic: " << name;
    }
}

TEST(TestIngestorScoreTransform, TheInverseIsMonotoneSoRankingSurvivesIt)
{
    // Why a wrong transform is invisible to selection, stated as a property: the inverse is
    // increasing, so applying the wrong one reorders nothing. Anything asserting only on the
    // chosen kernel cannot detect the defect this file exists to catch.
    //
    // Sampled across zero, not just on the positive side. The first version of this case used
    // 1.5 and 2.5 and passed while `sqrt` was inverting as raw*raw -- which is monotone on the
    // positives and inverts the order across zero, mapping -0.5 to 0.25 so that it outranks
    // +0.25. Two positive points cannot see that, and a GBDT raw score is unbounded.
    for(const auto* transform : score_transform::SUPPORTED_TRANSFORMS)
    {
        for(const double lower : {-2.5, -0.5, 0.25, 1.5})
        {
            const double a = score_transform::applyInverse(lower, transform);
            const double b = score_transform::applyInverse(lower + 1.0, transform);

            // NaN is the declared out-of-domain answer, and the ranking maps it to "no
            // measurement". Only the in-domain pairs carry an ordering obligation.
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
    // Each of these inverses has a domain the raw score can leave, because a GBDT prediction is
    // unbounded. Saying NaN lets the caller apply RFC 0019 §5 step 7's "no measurement" rule;
    // returning a plausible number instead puts a fabricated score into a ranking.
    EXPECT_TRUE(std::isnan(score_transform::applyInverse(-0.5, "exp")))   // log of a negative
        << "exp's inverse produced a number outside its domain";
    EXPECT_TRUE(std::isnan(score_transform::applyInverse(-0.5, "sqrt")))  // squaring flips sign
        << "sqrt's inverse mapped a negative prediction to a positive score";

    // In-domain values are untouched.
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(4.0, "sqrt"), 16.0);
    EXPECT_DOUBLE_EQ(score_transform::applyInverse(0.0, "sqrt"), 0.0);
}

} // namespace
} // namespace hipdnn_plugin_sdk::ingestor::uhd

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
