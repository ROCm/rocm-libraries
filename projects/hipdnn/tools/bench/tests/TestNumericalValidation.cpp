// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestNumericalValidation.cpp
 * @brief The correctness gate RFC 0019 §13.2 puts in front of a training label.
 *
 * Device-free: the verdict is a function of the output images alone.
 */

#include <hipdnn_bench/NumericalValidation.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace
{

constexpr int64_t OUTPUT_UID = 7;

std::map<int64_t, hipdnn_bench::TensorDescription> tensors(hipdnn_frontend::DataType dataType
                                                           = hipdnn_frontend::DataType::FLOAT)
{
    return {{OUTPUT_UID, {"Y", dataType}}};
}

/// A candidate that executed and left @p values in the single output tensor.
hipdnn_bench::CandidateOutput ran(const std::vector<float>& values)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    std::vector<uint8_t> image(values.size() * sizeof(float));
    std::memcpy(image.data(), values.data(), image.size());
    candidate.images[OUTPUT_UID] = std::move(image);
    return candidate;
}

/// The same, for a tensor of raw 16-bit codes (half, bfloat16) or an opaque dtype.
hipdnn_bench::CandidateOutput ranRaw(const std::vector<uint8_t>& bytes)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    candidate.images[OUTPUT_UID] = bytes;
    return candidate;
}

std::vector<uint8_t> halfCodes(const std::vector<uint16_t>& codes)
{
    std::vector<uint8_t> bytes(codes.size() * sizeof(uint16_t));
    std::memcpy(bytes.data(), codes.data(), bytes.size());
    return bytes;
}

using hipdnn_bench::NumericalVerdict;

/// Every candidate of one problem, cross-checked in one go.
std::vector<hipdnn_bench::ValidationOutcome>
    crossCheck(std::vector<hipdnn_bench::CandidateOutput> candidates,
               const std::map<int64_t, hipdnn_bench::TensorDescription>& tensors)
{
    hipdnn_bench::CatalogCrossCheck check(tensors);
    for(auto& candidate : candidates)
    {
        check.add(std::move(candidate));
    }
    return check.verdicts();
}

/// A graph with one input X (uid 1, FLOAT) and one result Y (uid 2, @p outputType): Y = X * X.
hipdnn_bench::VariantPackPlan inputAndOutputPlan(hipdnn_frontend::DataType outputType)
{
    using hipdnn_frontend::graph::TensorAttributes;
    hipdnn_frontend::graph::Graph graph;
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(1).set_name("X").set_dim({2}).set_stride({1}).set_data_type(
        hipdnn_frontend::DataType::FLOAT);
    hipdnn_frontend::graph::PointwiseAttributes square;
    square.set_mode(hipdnn_frontend::PointwiseMode::MUL);
    auto y = graph.pointwise(x, x, square);
    y->set_uid(2).set_name("Y").set_dim({2}).set_stride({1}).set_data_type(outputType);
    y->set_output(true);
    return hipdnn_bench::planVariantPack(graph);
}

/// What the tool reads back after a candidate ran, selected by the tool's own
/// crossCheckedOutputs().
hipdnn_bench::CandidateOutput captured(const hipdnn_bench::VariantPackPlan& plan,
                                       const std::map<int64_t, std::vector<uint8_t>>& memory,
                                       std::map<int64_t, hipdnn_bench::TensorDescription>& tensors)
{
    hipdnn_bench::CandidateOutput candidate;
    candidate.executed = true;
    for(const auto& tensor : hipdnn_bench::crossCheckedOutputs(plan))
    {
        tensors[tensor.uid] = {tensor.name, tensor.dataType};
        candidate.images[tensor.uid] = memory.at(tensor.uid);
    }
    return candidate;
}

std::vector<uint8_t> floatBytes(const std::vector<float>& values)
{
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

} // namespace

TEST(TestNumericalValidation, IdenticalInputsAreNotAgreementAboutUndecodableOutputs)
{
    // The FLOAT input is identical across candidates and must not count as comparable output.
    const auto plan = inputAndOutputPlan(hipdnn_frontend::DataType::FP8_E4M3);
    ASSERT_TRUE(plan.error.empty()) << plan.error;
    const auto input = floatBytes({1.0F, 2.0F});

    std::map<int64_t, hipdnn_bench::TensorDescription> declared;
    hipdnn_bench::CatalogCrossCheck check(declared);
    check.add(captured(plan, {{1, input}, {2, {0x01, 0x02}}}, declared));
    check.add(captured(plan, {{1, input}, {2, {0x40, 0x50}}}, declared));
    const auto verdicts = check.verdicts();

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN) << verdict.reason;
        EXPECT_NE(verdict.reason.find("no_comparable_output"), std::string::npos);
    }
}

TEST(TestNumericalValidation, NonZeroInputsDoNotMakeAnUntouchedOutputEvidence)
{
    // Filled input X must not make the all-zero output Y look touched.
    const auto plan = inputAndOutputPlan(hipdnn_frontend::DataType::FLOAT);
    ASSERT_TRUE(plan.error.empty()) << plan.error;
    const std::map<int64_t, std::vector<uint8_t>> memory{{1, floatBytes({1.0F, 2.0F})},
                                                         {2, floatBytes({0.0F, 0.0F})}};

    std::map<int64_t, hipdnn_bench::TensorDescription> declared;
    hipdnn_bench::CatalogCrossCheck check(declared);
    check.add(captured(plan, memory, declared));
    check.add(captured(plan, memory, declared));
    check.add(captured(plan, memory, declared));

    for(const auto& verdict : check.verdicts())
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN) << verdict.reason;
        EXPECT_NE(verdict.reason.find("degenerate_reference"), std::string::npos);
    }
}

TEST(TestNumericalValidation, APluralityIsNotAgreement)
{
    // {1, 1, 2, 3}: the largest cohort holds only two of four.
    const auto verdicts
        = crossCheck({ran({1.0F}), ran({1.0F}), ran({2.0F}), ran({3.0F})}, tensors());

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::DISAGREED) << verdict.reason;
        EXPECT_NE(verdict.reason.find("disputed_output"), std::string::npos);
    }

    // Three of five is a strict majority, which decides.
    const auto decided
        = crossCheck({ran({1.0F}), ran({1.0F}), ran({2.0F}), ran({3.0F}), ran({1.0F})}, tensors());
    EXPECT_EQ(decided[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(decided[2].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(decided[2].reason.find("output_mismatch"), std::string::npos);
}

TEST(TestNumericalValidation, WrongKernelIsMarkedInvalidAndNamedInTheReason)
{
    const auto verdicts = crossCheck(
        {ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 3.0F}), ran({1.0F, 2.0F, 99.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[2].reason.find("output_mismatch"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("tensor 'Y' element 2"), std::string::npos);
}

TEST(TestNumericalValidation, MajorityDecidesWhenTheCatalogsFirstCandidateIsTheBrokenOne)
{
    const auto verdicts
        = crossCheck({ran({99.0F, 99.0F}), ran({1.0F, 2.0F}), ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, RoundingDifferencesBetweenCorrectKernelsDoNotFailTheGate)
{
    const auto verdicts
        = crossCheck({ran({1000.0F, 2000.0F}), ran({1000.0001F, 1999.9999F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, AnUncorroboratedCandidateIsUnknownRatherThanValid)
{
    const auto verdicts = crossCheck({ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_reference"), std::string::npos);
}

TEST(TestNumericalValidation, UnanimousUntouchedOutputIsNotEvidenceOfCorrectness)
{
    // Output buffers start zero-filled, so a kernel that writes nothing looks like this.
    const auto verdicts
        = crossCheck({ran({0.0F, 0.0F}), ran({0.0F, 0.0F}), ran({0.0F, 0.0F})}, tensors());

    for(const auto& verdict : verdicts)
    {
        EXPECT_EQ(verdict.verdict, NumericalVerdict::UNKNOWN);
        EXPECT_NE(verdict.reason.find("degenerate_reference"), std::string::npos);
    }
}

TEST(TestNumericalValidation, AnEvenSplitLeavesNoCandidateTrusted)
{
    // One of the two is wrong and nothing says which, so neither is a label (§13.2).
    const auto verdicts = crossCheck({ran({1.0F}), ran({5.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::DISAGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[0].reason.find("disputed_output"), std::string::npos);
}

TEST(TestNumericalValidation, NonFiniteOutputDisagreesWithAFiniteReference)
{
    // NaN fails every magnitude comparison, so `abs(a - b) > tolerance` alone would pass it.
    const auto verdicts
        = crossCheck({ran({1.0F, 2.0F}), ran({1.0F, 2.0F}), ran({1.0F, std::nanf("")})}, tensors());

    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(TestNumericalValidation, HalfPrecisionIsDecodedRatherThanComparedAsBytes)
{
    // 0x3C00 = 1.0, 0x4000 = 2.0, 0x3C01 = 1.0 + 1 ulp (within tolerance), 0x4400 = 4.0.
    const auto agreeing
        = crossCheck({ranRaw(halfCodes({0x3C00, 0x4000})), ranRaw(halfCodes({0x3C01, 0x4000}))},
                     tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(agreeing[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(agreeing[1].verdict, NumericalVerdict::AGREED);

    const auto split = crossCheck({ranRaw(halfCodes({0x3C00, 0x4000})),
                                   ranRaw(halfCodes({0x3C00, 0x4000})),
                                   ranRaw(halfCodes({0x3C00, 0x4400}))},
                                  tensors(hipdnn_frontend::DataType::HALF));
    EXPECT_EQ(split[2].verdict, NumericalVerdict::DISAGREED);
}

TEST(TestNumericalValidation, AnUndecodableDtypeIsUnknownRatherThanAssumedEqual)
{
    const auto verdicts = crossCheck({ranRaw({0x01, 0x02}), ranRaw({0x40, 0x50})},
                                     tensors(hipdnn_frontend::DataType::FP8_E4M3));

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[0].reason.find("no_comparable_output"), std::string::npos);
}

TEST(TestNumericalValidation, ACandidateThatNeverRanNeitherJoinsNorSplitsACohort)
{
    // Counting the failure as a dissent would turn a unanimous catalog into an even split.
    hipdnn_bench::CandidateOutput failed;
    failed.failure = "engine declined to build this configuration";

    const auto verdicts = crossCheck({ran({1.0F, 2.0F}), failed, ran({1.0F, 2.0F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::UNKNOWN);
    EXPECT_NE(verdicts[1].reason.find("engine declined"), std::string::npos);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
}

TEST(TestNumericalValidation, TheVerdictColumnCannotBeReadBackAsABoolean)
{
    // "Unknown" is a word so `astype(bool)` fails loudly instead of folding it into True.
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::AGREED), "True");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::DISAGREED), "False");
    EXPECT_STREQ(hipdnn_bench::verdictText(NumericalVerdict::UNKNOWN), "Unknown");
}

TEST(TestNumericalValidation, GarbageInASmallElementIsNotHiddenByTheTensorsLargest)
{
    // Element 1 is off by 2e-4: inside a tensor-wide bar of 1e-5 * 40 = 4e-4, but twice its
    // own value.
    const auto verdicts = crossCheck(
        {ran({40.0F, 1.0e-4F}), ran({40.0F, 1.0e-4F}), ran({40.0F, 3.0e-4F})}, tensors());

    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::DISAGREED);

    EXPECT_NE(verdicts[2].reason.find("element 1"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("3.000e-04"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("1.000e-04"), std::string::npos);
    EXPECT_NE(verdicts[2].reason.find("outside a tolerance of"), std::string::npos);
}

TEST(TestNumericalValidation, EveryCandidateOfAProblemReadsTheSameNonZeroInputs)
{
    using hipdnn_frontend::DataType;
    constexpr size_t ELEMENTS = 16;
    const uint64_t seed = hipdnn_bench::detail::graphFillSeed({0x01, 0x02, 0x03});
    const auto image = hipdnn_bench::detail::inputFillImage(
        DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID);
    ASSERT_EQ(image.size(), ELEMENTS * sizeof(float));

    // Not all zero.
    EXPECT_NE(std::count(image.begin(), image.end(), uint8_t{0}),
              static_cast<std::ptrdiff_t>(image.size()));

    // Deterministic for one problem, so every candidate reads the same bytes.
    EXPECT_EQ(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID));

    // Different per tensor (A == B would hide a transpose) and per graph.
    EXPECT_NE(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT, ELEMENTS * sizeof(float), seed, OUTPUT_UID + 1));
    EXPECT_NE(image,
              hipdnn_bench::detail::inputFillImage(
                  DataType::FLOAT,
                  ELEMENTS * sizeof(float),
                  hipdnn_bench::detail::graphFillSeed({0x01, 0x02, 0x04}),
                  OUTPUT_UID));

    // Every value is 1 or 2 in magnitude: exact in every encoded type.
    const auto halfImage = hipdnn_bench::detail::inputFillImage(
        DataType::HALF, ELEMENTS * sizeof(uint16_t), seed, OUTPUT_UID);
    ASSERT_EQ(halfImage.size(), ELEMENTS * sizeof(uint16_t));
    for(size_t index = 0; index < ELEMENTS; ++index)
    {
        const double single
            = std::abs(hipdnn_bench::detail::decodeElement(image, index, DataType::FLOAT));
        const double half
            = std::abs(hipdnn_bench::detail::decodeElement(halfImage, index, DataType::HALF));
        EXPECT_TRUE(single == 1.0 || single == 2.0) << "element " << index << " is " << single;
        EXPECT_TRUE(half == 1.0 || half == 2.0) << "element " << index << " is " << half;
    }

    // A type the encoder cannot write exactly keeps the zero fill rather than a guess.
    EXPECT_TRUE(hipdnn_bench::detail::inputFillImage(DataType::FP4_E2M1, ELEMENTS, seed, OUTPUT_UID)
                    .empty());
}

TEST(TestNumericalValidation, ACandidateThatJoinsACohortDoesNotKeepItsImage)
{
    const auto declared = tensors();
    hipdnn_bench::CatalogCrossCheck check(declared);

    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    check.add(ran({1.0F, 2.0F}));
    EXPECT_EQ(check.retainedImages(), 1U);
    // A new answer is kept: the minority verdict is written from it.
    check.add(ran({1.0F, 99.0F}));
    EXPECT_EQ(check.retainedImages(), 2U);

    // Releasing the joiners does not change the verdicts.
    const auto verdicts = check.verdicts();
    EXPECT_EQ(verdicts[0].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[1].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[2].verdict, NumericalVerdict::AGREED);
    EXPECT_EQ(verdicts[3].verdict, NumericalVerdict::DISAGREED);
    EXPECT_NE(verdicts[3].reason.find("tensor 'Y' element 1"), std::string::npos);
}
