// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestNumericalValidation.cpp
 * @brief The correctness gate RFC 0019 §13.2 puts in front of a training label.
 *
 * Tested here rather than through the tool because the decision needs no device: it is a
 * function of the output images, and the images are the only thing a GPU contributes. What
 * a device run could not test any better, and what these cases exist for, is that the gate
 * fails in the right direction. Both directions are damaging and only one is visible:
 *
 *  - Too permissive, and the wrong-but-fast kernel keeps the best time in its group and
 *    becomes the label the ranker is trained to prefer -- §13.2's inverted oracle, which
 *    nothing downstream catches because §6.3 fingerprints the feature contract, not output.
 *  - Too strict, and every honest candidate is marked invalid, promotion is blocked on a
 *    defect that does not exist, and the next author to see it learns to ignore the gate.
 */

#include <hipdnn_bench/NumericalValidation.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{

constexpr int64_t kOutputUid = 7;

std::map<int64_t, hipdnn_bench::TensorDescription>
    tensors(hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::FLOAT)
{
    return {{kOutputUid, {"Y", dataType}}};
}

/// A candidate that executed and left @p values in the single output tensor.
hipdnn_bench::CandidateOutput ran(const std::vector<float>& values)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    std::vector<uint8_t> image(values.size() * sizeof(float));
    std::memcpy(image.data(), values.data(), image.size());
    candidate.images[kOutputUid] = std::move(image);
    return candidate;
}

/// The same, for a tensor of raw 16-bit codes (half, bfloat16) or an opaque dtype.
hipdnn_bench::CandidateOutput ranRaw(const std::vector<uint8_t>& bytes)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    candidate.images[kOutputUid] = bytes;
    return candidate;
}

std::vector<uint8_t> halfCodes(const std::vector<uint16_t>& codes)
{
    std::vector<uint8_t> bytes(codes.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), codes.data(), bytes.size());
    return bytes;
}

using hipdnn_bench::NumericalVerdict;

} // namespace

TEST(NumericalValidation, WrongKernelIsMarkedInvalidAndNamedInTheReason)
{
    // The case the whole gate exists for: two candidates compute the problem and a third
    // returns something else. Without this the third keeps whatever time it measured and,
    // because not computing the answer is fast, that time is the group's best.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 99.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
    // Named, because "a candidate is invalid" does not tell an author where to look and
    // §13.2's remedy is a matcher or kernel change they have to make by hand.
    EXPECT_NE(verdicts[2].reason.find("output_mismatch"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("tensor 'Y' element 2"), std::string::npos);
}

TEST(NumericalValidation, MajorityDecidesWhenTheCatalogsFirstCandidateIsTheBrokenOne)
{
    // Taking candidate 0 as the reference is the obvious implementation and it inverts the
    // verdicts exactly when the gate matters most: the broken kernel would be declared the
    // truth and every correct one marked invalid, blocking promotion on the wrong pack.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({99.0F, 99.0F}), ran({1.0F, 2.0F}), ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(NumericalValidation, RoundingDifferencesBetweenCorrectKernelsDoNotFailTheGate)
{
    // Two kernels that tile a reduction differently accumulate in a different order, so
    // their last bits differ by construction. A bitwise gate marks both invalid, which
    // blocks every promotion and is a worse failure than having no gate at all.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({1000.0F, 2000.0F}), ran({1000.0001F, 1999.9999F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
}

TEST(NumericalValidation, AnUncorroboratedCandidateIsUnknownRatherThanValid)
{
    // One candidate agreeing with itself is not evidence. Reporting it valid is the silent
    // "we did not check" that reads as "we checked", which §13.2 forbids by name.
    const auto verdicts = hipdnn_bench::crossCheckCandidates({ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_reference"), std::string::npos);
}

TEST(NumericalValidation, UnanimousUntouchedOutputIsNotEvidenceOfCorrectness)
{
    // The tool allocates zero-filled buffers and writes no inputs, so for many operations a
    // correct kernel and a kernel that writes nothing both leave zeros. Unanimity on zeros
    // would otherwise certify a catalog in which nothing ran.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({0.0F, 0.0F}), ran({0.0F, 0.0F}), ran({0.0F, 0.0F})}, tensors());

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN);
        EXPECT_NE(verdict.reason.find("degenerate_reference"), std::string::npos);
    }
}

TEST(NumericalValidation, AnEvenSplitLeavesNoCandidateTrusted)
{
    // Two candidates, two answers: one of them is wrong and nothing here can say which. The
    // timing of a candidate that is not known correct is not a label (§13.2), so both are
    // suppressed and emission blocks until the author resolves it in the pack.
    const auto verdicts
        = hipdnn_bench::crossCheckCandidates({ran({1.0F}), ran({5.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[0].reason.find("disputed_output"), std::string::npos);
}

TEST(NumericalValidation, NonFiniteOutputDisagreesWithAFiniteReference)
{
    // A NaN fails every magnitude comparison it takes part in, so a candidate that produced
    // one would slip through a gate written as `abs(a - b) > tolerance` alone.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({1.0F, 2.0F}), ran({1.0F, 2.0F}), ran({1.0F, std::nanf("")})}, tensors());

    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(NumericalValidation, HalfPrecisionIsDecodedRatherThanComparedAsBytes)
{
    // The binary16 decode is written out by hand here, and a wrong one is silent: it would
    // either wave a broken kernel through or condemn a good one. 0x3C00 is 1.0, 0x4000 is
    // 2.0 and 0x3C01 is one ulp above 1.0 -- inside tolerance -- while 0x4400 is 4.0.
    const auto agreeing = hipdnn_bench::crossCheckCandidates(
        {ranRaw(halfCodes({0x3C00, 0x4000})), ranRaw(halfCodes({0x3C01, 0x4000}))},
        tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(agreeing[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(agreeing[1].verdict, NumericalVerdict::AGREED);

    const auto split = hipdnn_bench::crossCheckCandidates(
        {ranRaw(halfCodes({0x3C00, 0x4000})),
         ranRaw(halfCodes({0x3C00, 0x4000})),
         ranRaw(halfCodes({0x3C00, 0x4400}))},
        tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(split[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(NumericalValidation, AnUndecodableDtypeIsUnknownRatherThanAssumedEqual)
{
    // FP8 and the packed types are deliberately outside the decoder (Open Question 19 leaves
    // the per-op reference open). Skipping such a tensor silently would make every candidate
    // of an FP8 problem compare equal on nothing at all and come back valid.
    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ranRaw({0x01, 0x02}), ranRaw({0x40, 0x50})},
        tensors(hipdnn_frontend::DataType::FP8_E4M3));

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_comparable_output"), std::string::npos);
}

TEST(NumericalValidation, ACandidateThatNeverRanNeitherJoinsNorSplitsACohort)
{
    // A candidate that could not be built is a coverage gap, not a corruption (§13.2): it is
    // recorded with its reason, and it must not count as a dissenting answer -- one build
    // failure would otherwise turn a unanimous catalog into an unresolvable even split.
    hipdnn_bench::CandidateOutput failed;
    failed.failure = "engine declined to build this configuration";

    const auto verdicts = hipdnn_bench::crossCheckCandidates(
        {ran({1.0F, 2.0F}), failed, ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[1].reason.find("engine declined"), std::string::npos);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(NumericalValidation, TheVerdictColumnCannotBeReadBackAsABoolean)
{
    // Three states in a CSV column that a reader will try to coerce. "Unknown" is spelled as
    // a word precisely so `astype(bool)` fails loudly instead of folding it into True.
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::AGREED), "True");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::DISAGREED), "False");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::UNKNOWN), "Unknown");
}
