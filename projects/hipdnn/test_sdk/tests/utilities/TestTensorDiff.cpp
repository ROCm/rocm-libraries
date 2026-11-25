// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Helpers.hpp"

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TensorDiff.hpp>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using hipdnn_data_sdk::helpers::createRaggedSdpaTensor;
using hipdnn_data_sdk::helpers::RAGGED_SDPA_DIMS;

// =================================================================================================
// computeTensorDiff
// =================================================================================================

TEST(TestComputeTensorDiff, IdenticalTensorsHaveNoMismatches)
{
    const std::vector<int64_t> dims = {4, 4};
    Tensor<float> ref(dims);
    ref.fillTensorWithRandomValues(-1.0f, 1.0f);

    Tensor<float> impl(dims);
    TensorView<float> refView(ref);
    const TensorView<float> implView(impl);
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        impl.setHostValue(refView.getHostValue(indices), indices);
    });

    auto summary = computeTensorDiff<float>(ref, impl, 1e-5f, 1e-5f);

    EXPECT_EQ(summary.totalElements, 16u);
    EXPECT_EQ(summary.mismatchCount, 0u);
    EXPECT_FLOAT_EQ(summary.maxAbsDiff, 0.0f);
    EXPECT_FLOAT_EQ(summary.meanAbsDiff, 0.0f);
    EXPECT_TRUE(summary.worstMismatches.empty());
}

TEST(TestComputeTensorDiff, DetectsSingleMismatch)
{
    const std::vector<int64_t> dims = {2, 3};
    Tensor<float> ref(dims);
    Tensor<float> impl(dims);

    // Fill both with zeros
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        ref.setHostValue(0.0f, indices);
        impl.setHostValue(0.0f, indices);
    });

    // Introduce a single difference
    ref.setHostValue(1.0f, std::vector<int64_t>{1, 2});
    impl.setHostValue(2.0f, std::vector<int64_t>{1, 2});

    auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 6u);
    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_FLOAT_EQ(summary.maxAbsDiff, 1.0f);
    EXPECT_FLOAT_EQ(summary.meanAbsDiff, 1.0f);
    ASSERT_EQ(summary.maxDiffIndices.size(), 2u);
    EXPECT_EQ(summary.maxDiffIndices[0], 1);
    EXPECT_EQ(summary.maxDiffIndices[1], 2);
    ASSERT_EQ(summary.worstMismatches.size(), 1u);
    EXPECT_FLOAT_EQ(summary.worstMismatches[0].refValue, 1.0f);
    EXPECT_FLOAT_EQ(summary.worstMismatches[0].implValue, 2.0f);
}

TEST(TestComputeTensorDiff, MismatchesWithinToleranceAreIgnored)
{
    const std::vector<int64_t> dims = {2, 2};
    Tensor<float> ref(dims);
    Tensor<float> impl(dims);

    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        ref.setHostValue(1.0f, indices);
        impl.setHostValue(1.0f, indices);
    });

    // Introduce a small difference (0.001) within tolerance
    impl.setHostValue(1.001f, std::vector<int64_t>{0, 0});

    auto summary = computeTensorDiff<float>(ref, impl, 0.01f, 0.0f);

    EXPECT_EQ(summary.mismatchCount, 0u);
}

TEST(TestComputeTensorDiff, ShapeMismatchReturnsEmptySummary)
{
    Tensor<float> ref({2, 3});
    Tensor<float> impl({3, 2});

    auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 0u);
    EXPECT_EQ(summary.mismatchCount, 0u);
    EXPECT_TRUE(summary.worstMismatches.empty());
}

TEST(TestComputeTensorDiff, WorstMismatchesCappedAtMaxMismatches)
{
    const std::vector<int64_t> dims = {10};
    Tensor<float> ref(dims);
    Tensor<float> impl(dims);

    // Every element differs
    for(int64_t i = 0; i < 10; ++i)
    {
        ref.setHostValue(0.0f, std::vector<int64_t>{i});
        impl.setHostValue(static_cast<float>(i + 1), std::vector<int64_t>{i});
    }

    const size_t maxMismatches = 3;
    auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, maxMismatches);

    EXPECT_EQ(summary.mismatchCount, 10u);
    ASSERT_EQ(summary.worstMismatches.size(), maxMismatches);

    // Must retain the largest 3 diffs (10, 9, 8), sorted descending
    EXPECT_FLOAT_EQ(summary.worstMismatches[0].absDiff, 10.0f);
    EXPECT_FLOAT_EQ(summary.worstMismatches[1].absDiff, 9.0f);
    EXPECT_FLOAT_EQ(summary.worstMismatches[2].absDiff, 8.0f);
}

TEST(TestComputeTensorDiff, ZeroMaxMismatchesSuppressesEntries)
{
    const std::vector<int64_t> dims = {4};
    Tensor<float> ref(dims);
    Tensor<float> impl(dims);

    for(int64_t i = 0; i < 4; ++i)
    {
        ref.setHostValue(0.0f, std::vector<int64_t>{i});
        impl.setHostValue(1.0f, std::vector<int64_t>{i});
    }

    auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 0);

    EXPECT_EQ(summary.mismatchCount, 4u);
    EXPECT_TRUE(summary.worstMismatches.empty());
    EXPECT_GT(summary.maxAbsDiff, 0.0f);
}

// =================================================================================================
// printTensorDiffSummary
// =================================================================================================

TEST(TestPrintTensorDiffSummary, OutputContainsExpectedFields)
{
    TensorDiffSummary summary{};
    summary.totalElements = 100;
    summary.mismatchCount = 5;
    summary.maxAbsDiff = 0.01f;
    summary.meanAbsDiff = 0.005f;
    summary.maxDiffIndices = {1, 2, 3};
    summary.worstMismatches = {{{1, 2, 3}, 1.0f, 1.01f, 0.01f}};

    std::ostringstream oss;
    printTensorDiffSummary(oss, "testTensor", summary);
    const std::string output = oss.str();

    EXPECT_NE(output.find("testTensor"), std::string::npos);
    EXPECT_NE(output.find("Total elements: 100"), std::string::npos);
    EXPECT_NE(output.find("Mismatched:     5"), std::string::npos);
    EXPECT_NE(output.find("Worst mismatches:"), std::string::npos);
}

TEST(TestPrintTensorDiffSummary, NoMismatchesOmitsWorstSection)
{
    TensorDiffSummary summary{};
    summary.totalElements = 50;
    summary.mismatchCount = 0;
    summary.maxAbsDiff = 0.0f;
    summary.meanAbsDiff = 0.0f;

    std::ostringstream oss;
    printTensorDiffSummary(oss, "clean", summary);
    const std::string output = oss.str();

    EXPECT_EQ(output.find("Worst mismatches:"), std::string::npos);
}

// =================================================================================================
// validateAndReport
// =================================================================================================

TEST(TestValidateAndReport, PassingValidationPrintsSuccessful)
{
    const std::vector<int64_t> dims = {3, 3};
    Tensor<float> ref(dims);
    ref.fillTensorWithRandomValues(-1.0f, 1.0f);

    Tensor<float> impl(dims);
    TensorView<float> refView(ref);
    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        impl.setHostValue(refView.getHostValue(indices), indices);
    });

    const CpuFpReferenceValidation<float> validator(1e-5f, 1e-5f);

    std::ostringstream oss;
    const bool result = validateAndReport<float>(oss, "y", validator, ref, impl, 1e-5f, 1e-5f);

    EXPECT_TRUE(result);
    EXPECT_NE(oss.str().find("successful"), std::string::npos);
}

TEST(TestValidateAndReport, FailingValidationPrintsDiff)
{
    const std::vector<int64_t> dims = {3, 3};
    Tensor<float> ref(dims);
    Tensor<float> impl(dims);

    iterateAlongDimensions(dims, [&](const std::vector<int64_t>& indices) {
        ref.setHostValue(0.0f, indices);
        impl.setHostValue(0.0f, indices);
    });

    ref.setHostValue(0.0f, std::vector<int64_t>{1, 1});
    impl.setHostValue(100.0f, std::vector<int64_t>{1, 1});

    const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);

    std::ostringstream oss;
    const bool result = validateAndReport<float>(oss, "output", validator, ref, impl, 0.0f, 0.0f);

    EXPECT_FALSE(result);
    const std::string output = oss.str();
    EXPECT_NE(output.find("failed"), std::string::npos);
    EXPECT_NE(output.find("Tensor diff"), std::string::npos);
}

TEST(TestValidateAndReport, ShapeMismatchPrintsShapeError)
{
    Tensor<float> ref({2, 3});
    Tensor<float> impl({3, 2});

    const CpuFpReferenceValidation<float> validator(0.0f, 0.0f);

    std::ostringstream oss;
    const bool result = validateAndReport<float>(oss, "x", validator, ref, impl, 0.0f, 0.0f);

    EXPECT_FALSE(result);
    EXPECT_NE(oss.str().find("shape mismatch"), std::string::npos);
}

// =================================================================================================
// Ragged tensors
// =================================================================================================

template <typename T>
class RaggedTensorDiff : public ::testing::Test
{
};

using RaggedTensorDiffTypes
    = ::testing::Types<float, hipdnn_data_sdk::types::half, hipdnn_data_sdk::types::bfloat16>;
TYPED_TEST_SUITE(RaggedTensorDiff, RaggedTensorDiffTypes, );

TYPED_TEST(RaggedTensorDiff, IdenticalTensorsCountInBlockElements)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f);

    auto summary = computeTensorDiff<TypeParam>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 20u);
    EXPECT_EQ(summary.mismatchCount, 0u);
}

TYPED_TEST(RaggedTensorDiff, InBlockNaNReferenceIsExcluded)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f);
    ref.setHostValue(std::numeric_limits<TypeParam>::quiet_NaN(), 1, 1, 2, 1);
    impl.setHostValue(TypeParam(2.0f), 1, 1, 2, 1);

    auto summary = computeTensorDiff<TypeParam>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 19u);
    EXPECT_EQ(summary.mismatchCount, 0u);
}

TYPED_TEST(RaggedTensorDiff, ReportsInBlockDifference)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f);
    impl.setHostValue(TypeParam(2.0f), 1, 1, 2, 1);

    auto summary = computeTensorDiff<TypeParam>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 20u);
    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_FLOAT_EQ(summary.maxAbsDiff, 1.5f);
    EXPECT_EQ(summary.maxDiffIndices, (std::vector<int64_t>{1, 1, 2, 1}));
}

TYPED_TEST(RaggedTensorDiff, MismatchedLayoutsReturnEmptySummary)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f, {0, 3, 5});

    auto summary = computeTensorDiff<TypeParam>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.totalElements, 0u);
    EXPECT_EQ(summary.mismatchCount, 0u);
}

TYPED_TEST(RaggedTensorDiff, ValidateAndReportPrintsLayoutMismatch)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f, {0, 3, 5});
    const CpuFpReferenceValidation<TypeParam> validator;

    std::ostringstream oss;
    const bool result = validateAndReport<TypeParam>(oss, "o", validator, ref, impl, 0.0f, 0.0f);

    EXPECT_FALSE(result);
    EXPECT_NE(oss.str().find("ragged layout mismatch"), std::string::npos) << oss.str();
}

// Same lengths at a different origin is a layout mismatch, and the report names origins and
// lengths.
TYPED_TEST(RaggedTensorDiff, ValidateAndReportPrintsOriginsAndLengths)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    auto impl = createRaggedSdpaTensor<TypeParam>(0.5f, {1, 3, 6});
    ASSERT_EQ(ref.raggedIterationInfo()->lengths, impl.raggedIterationInfo()->lengths);
    const CpuFpReferenceValidation<TypeParam> validator;

    std::ostringstream oss;
    const bool result = validateAndReport<TypeParam>(oss, "o", validator, ref, impl, 0.0f, 0.0f);

    EXPECT_FALSE(result);
    const std::string output = oss.str();
    EXPECT_NE(output.find("ragged layout mismatch"), std::string::npos) << output;
    EXPECT_NE(output.find("origins"), std::string::npos) << output;
    EXPECT_NE(output.find("lengths"), std::string::npos) << output;
}

TYPED_TEST(RaggedTensorDiff, ValidateAndReportPrintsLayoutMismatchAgainstDense)
{
    auto ref = createRaggedSdpaTensor<TypeParam>(0.5f);
    Tensor<TypeParam> impl(RAGGED_SDPA_DIMS);
    impl.fillTensorWithValue(0.5f);
    const CpuFpReferenceValidation<TypeParam> validator;

    std::ostringstream oss;
    const bool result = validateAndReport<TypeParam>(oss, "o", validator, ref, impl, 0.0f, 0.0f);

    EXPECT_FALSE(result);
    const std::string output = oss.str();
    EXPECT_NE(output.find("ragged layout mismatch"), std::string::npos) << output;
    EXPECT_NE(output.find("implementation dense"), std::string::npos) << output;
}

// =================================================================================================
// Non-finite elements
// =================================================================================================

namespace
{

constexpr float NAN_VALUE = std::numeric_limits<float>::quiet_NaN();
constexpr float INF_VALUE = std::numeric_limits<float>::infinity();

template <typename T = float>
Tensor<T> makeVector(const std::vector<float>& values)
{
    Tensor<T> tensor({static_cast<int64_t>(values.size())});
    for(size_t i = 0; i < values.size(); ++i)
    {
        tensor.setHostValue(static_cast<T>(values[i]),
                            std::vector<int64_t>{static_cast<int64_t>(i)});
    }
    return tensor;
}

} // namespace

// The validators fail a NaN, but NaN never compares greater than a threshold, so the diff
// used to report `Mismatched: 0` for a tensor that failed because of it.
TEST(TestComputeTensorDiffNonFinite, NaNImplementationAgainstFiniteReferenceIsCounted)
{
    auto ref = makeVector({1.0f, 2.0f, 3.0f});
    auto impl = makeVector({1.0f, NAN_VALUE, 3.0f});

    const auto summary = computeTensorDiff<float>(ref, impl, 1.0f, 1.0f);

    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 1u);
    ASSERT_EQ(summary.worstMismatches.size(), 1u);
    EXPECT_EQ(summary.worstMismatches[0].indices, (std::vector<int64_t>{1}));
    EXPECT_EQ(summary.worstMismatches[0].verdict, ElementVerdict::NON_FINITE_REJECT);
}

TEST(TestComputeTensorDiffNonFinite, FiniteReferenceAgainstImplementationInfIsCounted)
{
    auto ref = makeVector({1.0f, 2.0f});
    auto impl = makeVector({1.0f, INF_VALUE});

    const auto summary = computeTensorDiff<float>(ref, impl, 1.0f, 1.0f);

    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 1u);
}

TEST(TestComputeTensorDiffNonFinite, ReferenceInfAgainstImplementationNaNIsCountedWithMaskOn)
{
    auto ref = makeVector({1.0f, INF_VALUE});
    auto impl = makeVector({1.0f, NAN_VALUE});

    const auto summary = computeTensorDiff<float>(
        ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching(), MatchingInfinities::REJECTED);

    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 1u);
}

TEST(TestComputeTensorDiffNonFinite, OppositeSignInfinitiesAreCountedWithMaskOn)
{
    auto ref = makeVector({INF_VALUE});
    auto impl = makeVector({-INF_VALUE});

    const auto summary
        = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching());

    EXPECT_EQ(summary.mismatchCount, 1u);
}

TEST(TestComputeTensorDiffNonFinite, SameSignInfinitiesMatchWithMaskOn)
{
    auto ref = makeVector({INF_VALUE, -INF_VALUE, 1.0f});
    auto impl = makeVector({INF_VALUE, -INF_VALUE, 1.0f});

    const auto summary
        = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching());

    EXPECT_EQ(summary.mismatchCount, 0u);
    EXPECT_TRUE(summary.worstMismatches.empty());
}

TEST(TestComputeTensorDiffNonFinite, SameSignInfinitiesAreRejectedWithMaskOff)
{
    auto ref = makeVector({INF_VALUE, 1.0f});
    auto impl = makeVector({INF_VALUE, 1.0f});

    const auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 1u);
}

TEST(TestComputeTensorDiffNonFinite, SameSignInfinitiesMatchWithMatchingInfinitiesAccepted)
{
    auto ref = makeVector({INF_VALUE, -INF_VALUE, 1.0f});
    auto impl = makeVector({INF_VALUE, -INF_VALUE, 1.0f});

    const auto summary = computeTensorDiff<float>(
        ref, impl, 0.0f, 0.0f, 5, NonFiniteMask{}, MatchingInfinities::ACCEPTED);

    EXPECT_EQ(summary.mismatchCount, 0u);
}

TEST(TestComputeTensorDiffNonFinite, MatchingInfinitiesStillRejectsNaN)
{
    auto ref = makeVector({NAN_VALUE, INF_VALUE});
    auto impl = makeVector({NAN_VALUE, INF_VALUE});

    const auto summary = computeTensorDiff<float>(
        ref, impl, 0.0f, 0.0f, 5, NonFiniteMask{}, MatchingInfinities::ACCEPTED);

    EXPECT_EQ(summary.mismatchCount, 1u);
    ASSERT_EQ(summary.worstMismatches.size(), 1u);
    EXPECT_EQ(summary.worstMismatches[0].indices, (std::vector<int64_t>{0}));
}

TEST(TestComputeTensorDiffNonFinite, MatchedNaNPositionsAreNotCountedWithMaskOn)
{
    auto ref = makeVector({NAN_VALUE, 1.0f});
    auto impl = makeVector({NAN_VALUE, 1.0f});

    const auto summary
        = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching());

    EXPECT_EQ(summary.mismatchCount, 0u);
}

TEST(TestComputeTensorDiffNonFinite, UnwrittenSentinelElementIsCountedAndLabelled)
{
    using hipdnn_data_sdk::types::half;
    const auto sentinel = payloadNanSentinelBits<half>();
    ASSERT_TRUE(sentinel.has_value());

    // The reference holds NaN there, so without the sentinel rule NaN would match NaN.
    auto ref = makeVector<half>({1.0f, NAN_VALUE, 2.0f});
    auto impl = makeVector<half>({1.0f, 0.0f, 2.0f});
    impl.setHostValue(payloadNanSentinelValue<half>(), std::vector<int64_t>{1});

    const auto summary
        = computeTensorDiff<half>(ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching(sentinel));

    EXPECT_EQ(summary.mismatchCount, 1u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 1u);
    ASSERT_EQ(summary.worstMismatches.size(), 1u);
    EXPECT_EQ(summary.worstMismatches[0].verdict, ElementVerdict::UNWRITTEN_REJECT);

    std::ostringstream oss;
    printTensorDiffSummary(oss, "o", summary);
    EXPECT_NE(oss.str().find("[1]: ref=nan, impl=nan (unwritten-output sentinel)"),
              std::string::npos)
        << oss.str();
}

TEST(TestComputeTensorDiffNonFinite, ComputedNaNIsNotTheUnwrittenSentinel)
{
    using hipdnn_data_sdk::types::half;
    const auto sentinel = payloadNanSentinelBits<half>();
    ASSERT_TRUE(sentinel.has_value());

    auto ref = makeVector<half>({NAN_VALUE});
    auto impl = makeVector<half>({NAN_VALUE});

    const auto summary
        = computeTensorDiff<half>(ref, impl, 0.0f, 0.0f, 5, NonFiniteMask::matching(sentinel));

    EXPECT_EQ(summary.mismatchCount, 0u);
}

// A finite mismatch with a huge difference used to sort ahead of everything; a NaN entry must
// come first and survive the cap, however many finite mismatches arrive before it.
TEST(TestComputeTensorDiffNonFinite, NaNEntriesAreListedAndSortedFirst)
{
    const std::vector<float> refValues(8, 0.0f);
    const std::vector<float> implValues
        = {1.0f, 100.0f, 2.0f, 3.0f, 4.0f, NAN_VALUE, 5.0f, INF_VALUE};
    auto ref = makeVector(refValues);
    auto impl = makeVector(implValues);

    const auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 4);

    EXPECT_EQ(summary.mismatchCount, 8u);
    EXPECT_EQ(summary.nonFiniteMismatchCount, 2u);
    ASSERT_EQ(summary.worstMismatches.size(), 4u);
    EXPECT_EQ(summary.worstMismatches[0].indices, (std::vector<int64_t>{5}));
    EXPECT_EQ(summary.worstMismatches[0].verdict, ElementVerdict::NON_FINITE_REJECT);
    EXPECT_EQ(summary.worstMismatches[1].indices, (std::vector<int64_t>{7}));
    EXPECT_EQ(summary.worstMismatches[1].verdict, ElementVerdict::NON_FINITE_REJECT);
    EXPECT_FLOAT_EQ(summary.worstMismatches[2].absDiff, 100.0f);
    EXPECT_FLOAT_EQ(summary.worstMismatches[3].absDiff, 5.0f);
    EXPECT_EQ(summary.worstMismatches[2].verdict, ElementVerdict::GRADED);
}

TEST(TestComputeTensorDiffNonFinite, RejectsBeyondTheCapKeepTheLowestIndices)
{
    auto ref = makeVector(std::vector<float>(6, 0.0f));
    auto impl = makeVector({NAN_VALUE, NAN_VALUE, NAN_VALUE, NAN_VALUE, NAN_VALUE, NAN_VALUE});

    const auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f, 3);

    EXPECT_EQ(summary.mismatchCount, 6u);
    ASSERT_EQ(summary.worstMismatches.size(), 3u);
    EXPECT_EQ(summary.worstMismatches[0].indices, (std::vector<int64_t>{0}));
    EXPECT_EQ(summary.worstMismatches[1].indices, (std::vector<int64_t>{1}));
    EXPECT_EQ(summary.worstMismatches[2].indices, (std::vector<int64_t>{2}));
}

TEST(TestComputeTensorDiffNonFinite, DiffStatisticsCoverFiniteDifferencesOnly)
{
    auto ref = makeVector({0.0f, 0.0f, 0.0f, 0.0f});
    auto impl = makeVector({1.0f, NAN_VALUE, INF_VALUE, 3.0f});

    const auto summary = computeTensorDiff<float>(ref, impl, 0.0f, 0.0f);

    EXPECT_EQ(summary.mismatchCount, 4u);
    EXPECT_FLOAT_EQ(summary.maxAbsDiff, 3.0f);
    EXPECT_EQ(summary.maxDiffIndices, (std::vector<int64_t>{3}));
    EXPECT_FLOAT_EQ(summary.meanAbsDiff, 2.0f);
}

TEST(TestComputeTensorDiffNonFinite, PrintsNonFiniteValuesAsNanAndInf)
{
    auto ref = makeVector({INF_VALUE, -INF_VALUE, 1.0f});
    auto impl = makeVector({NAN_VALUE, INF_VALUE, -INF_VALUE});

    std::ostringstream oss;
    printTensorDiff<float>(oss, "t", ref, impl, 0.0f, 0.0f);
    const std::string output = oss.str();

    EXPECT_NE(output.find("Mismatched:     3 (100.00%)"), std::string::npos) << output;
    EXPECT_NE(output.find("[0]: ref=inf, impl=nan\n"), std::string::npos) << output;
    EXPECT_NE(output.find("[1]: ref=-inf, impl=inf\n"), std::string::npos) << output;
    EXPECT_NE(output.find("[2]: ref=1.000000e+00, impl=-inf\n"), std::string::npos) << output;
}

TEST(TestComputeTensorDiffNonFinite, ValidateAndReportListsTheNaNTheValidatorRejected)
{
    auto ref = makeVector({1.0f, 2.0f});
    auto impl = makeVector({1.0f, NAN_VALUE});
    const CpuFpReferenceValidation<float> validator(1.0f, 1.0f);

    std::ostringstream oss;
    const bool result = validateAndReport<float>(oss, "y", validator, ref, impl, 1.0f, 1.0f);

    EXPECT_FALSE(result);
    EXPECT_NE(oss.str().find("Mismatched:     1 (50.00%)"), std::string::npos) << oss.str();
    EXPECT_NE(oss.str().find("[1]: ref=2.000000e+00, impl=nan"), std::string::npos) << oss.str();
}

// The all-finite rendering is the report operators already read; the new rule must not
// change it.
TEST(TestComputeTensorDiffNonFinite, AllFiniteDiffOutputIsUnchanged)
{
    auto ref = makeVector({1.0f, 1.0f, 1.0f, 1.0f});
    auto impl = makeVector({1.0f, 1.5f, 1.0f, 1.25f});

    std::ostringstream oss;
    printTensorDiff<float>(oss, "t", ref, impl, 0.0f, 0.0f);

    const std::string expected
        = "  Tensor diff for \"t\":\n"
          "    Total elements: 4\n"
          "    Mismatched:     2 (50.00%)\n"
          "    Max abs diff:   5.000000e-01 at [1]\n"
          "    Mean abs diff:  3.750000e-01\n"
          "    Worst mismatches:\n"
          "      [1]: ref=1.000000e+00, impl=1.500000e+00, diff=5.000000e-01\n"
          "      [3]: ref=1.000000e+00, impl=1.250000e+00, diff=2.500000e-01\n";
    EXPECT_EQ(oss.str(), expected);
}
