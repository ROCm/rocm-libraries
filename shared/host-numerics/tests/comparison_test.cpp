// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testComparisonProgram() {
    using namespace roc::host_numerics;

    const std::array<float, 0> emptyStorage{};
    const Layout emptyLayout(Shape{0, 3}, {1, 1});
    ComparisonOptions emptyOptions;
    emptyOptions.computeElementwiseStatistics = false;
    emptyOptions.computeFrobenius = false;
    emptyOptions.maxReportedMismatches = 0;
    emptyOptions.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    const auto compareEmpty = [&](const ComparisonOptions& options) {
        return compare(Tensor::copyNativeStorage(emptyLayout, std::span<const float>(emptyStorage)),
                       Tensor::copyNativeStorage(emptyLayout, std::span<const float>(emptyStorage)),
                       options);
    };
    const ComparisonReport emptyResult = compareEmpty(emptyOptions);
    require(emptyResult.compared == 0 && emptyResult.allCloseEvaluated &&
                emptyResult.allClosePassed && !emptyResult.frobeniusEvaluated &&
                !emptyResult.ulpEvaluated,
            "Runtime fast comparison rejected an empty tensor.");

    const auto requireInvalidEmptyOptions = [&](const ComparisonOptions& options,
                                                const char* message) {
        bool rejected = false;
        try {
            (void)compareEmpty(options);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        require(rejected, message);
    };
    ComparisonOptions missingUlpType;
    missingUlpType.computeUlp = true;
    requireInvalidEmptyOptions(missingUlpType,
                               "Empty comparison accepted ULP evidence without a scalar type.");
    ComparisonOptions invalidUlpType = missingUlpType;
    invalidUlpType.ulpType = ScalarType::Count;
    requireInvalidEmptyOptions(invalidUlpType,
                               "Empty comparison accepted ScalarType::Count as a ULP type.");
    ComparisonOptions missingUlpEvidence;
    missingUlpEvidence.maximumUlpTolerance = 0.0;
    requireInvalidEmptyOptions(missingUlpEvidence,
                               "Empty comparison accepted a ULP criterion without ULP evidence.");
    ComparisonOptions missingFrobeniusEvidence;
    missingFrobeniusEvidence.computeFrobenius = false;
    missingFrobeniusEvidence.relativeFrobeniusTolerance = 0.0;
    requireInvalidEmptyOptions(
        missingFrobeniusEvidence,
        "Empty comparison accepted a Frobenius criterion without Frobenius evidence.");
    bool rejectedZeroSelectionStride = false;
    try {
        (void)OutputSelection::strided(0, 0);
    } catch (const std::invalid_argument&) {
        rejectedZeroSelectionStride = true;
    }
    require(rejectedZeroSelectionStride, "Output selection accepted a zero stride.");

    const std::array<double, 1> evidenceObserved{2.0};
    const std::array<double, 1> evidenceExpected{1.0};
    ComparisonOptions evidenceOnly;
    evidenceOnly.allClose = false;
    evidenceOnly.computeUlp = true;
    evidenceOnly.ulpType = ScalarType::Float64;
    const ComparisonReport evidenceResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(evidenceObserved)),
                Tensor::copyNativeStorage(std::span<const double>(evidenceExpected)), evidenceOnly);
    require(!evidenceResult.allCloseEvaluated && !evidenceResult.frobeniusEvaluated &&
                !evidenceResult.ulpEvaluated && evidenceResult.passed() &&
                evidenceResult.frobeniusDifference == 1.0 && evidenceResult.ulpCompared == 1,
            "Evidence-only comparison reported an evaluated criterion.");

    const std::array<double, 3> reversedStorage{1.0, 2.0, 3.0};
    const Layout reversedLayout(Shape{3}, {-1}, 2);
    ComparisonOptions reversedMetrics;
    reversedMetrics.allClose = false;
    reversedMetrics.computeElementwiseStatistics = false;
    reversedMetrics.computeFrobenius = true;
    reversedMetrics.maxReportedMismatches = 0;
    const auto reversedRuntimeResult =
        compare(Tensor::copyNativeStorage(reversedLayout, std::span<const double>(reversedStorage)),
                Tensor::copyNativeStorage(reversedLayout, std::span<const double>(reversedStorage)),
                reversedMetrics);
    require(reversedRuntimeResult.compared == 3 && reversedRuntimeResult.frobeniusDifference == 0.0,
            "Negative-stride comparison produced incorrect evidence.");

    const std::array<float, 8> expectedStorage{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f, 5.0f, 6.0f};
    auto observedStorage = expectedStorage;
    observedStorage[4] += 0.5f;
    observedStorage[6] += 1.0f;
    const Layout layout(Shape{2, 3}, {1, 3});

    ComparisonOptions selected;
    selected.selection = OutputSelection::strided(0, 2, std::numeric_limits<size_t>::max(),
                                                  IndexOrder::FirstDimensionFastest);
    selected.computeFrobenius = false;
    const auto selectedResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), selected);
    require(selectedResult.compared == 3 && selectedResult.mismatches == 1,
            "Selected comparison visited the wrong logical elements.");
    require(selectedResult.reportedMismatches[0].index == 4 &&
                selectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}) &&
                selectedResult.reportedMismatches[0].observedOffset == 6,
            "Selected comparison reported the wrong logical location.");
    selected.selection = OutputSelection::explicitIndices({4}, IndexOrder::FirstDimensionFastest);
    const auto explicitSelectedResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), selected);
    require(
        explicitSelectedResult.compared == 1 && explicitSelectedResult.mismatches == 1 &&
            explicitSelectedResult.reportedMismatches[0].coordinates == std::vector<size_t>({0, 2}),
        "Explicit comparison selection visited the wrong logical element.");
    ComparisonOptions paddedMetrics;
    paddedMetrics.allClose = false;
    paddedMetrics.computeElementwiseStatistics = false;
    paddedMetrics.computeFrobenius = true;
    paddedMetrics.maxReportedMismatches = 0;
    paddedMetrics.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    const auto paddedMetricResult = compare(
        Tensor::copyNativeStorage(layout, std::span<const float>(observedStorage)),
        Tensor::copyNativeStorage(layout, std::span<const float>(expectedStorage)), paddedMetrics);
    require(paddedMetricResult.compared == 6 && paddedMetricResult.frobeniusDifference > 0.0,
            "Regular strided comparison produced incorrect evidence.");

    const std::array<uint64_t, 1> wideIntegerObserved{uint64_t{1} << 53};
    const std::array<uint64_t, 1> wideIntegerExpected{(uint64_t{1} << 53) + 1};
    ComparisonOptions fastIntegerOptions;
    fastIntegerOptions.computeElementwiseStatistics = false;
    fastIntegerOptions.computeFrobenius = false;
    fastIntegerOptions.maxReportedMismatches = 0;
    const auto fastIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                fastIntegerOptions);
    require(!fastIntegerComparison.passed() && fastIntegerComparison.mismatches == 1,
            "Fast comparison rounded distinct wide integers together.");

    fastIntegerOptions.maxReportedMismatches = 1;
    const auto reportedIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                fastIntegerOptions);
    require(!reportedIntegerComparison.passed() && reportedIntegerComparison.mismatches == 1 &&
                reportedIntegerComparison.reportedMismatches.size() == 1 &&
                !reportedIntegerComparison.reportedMismatches[0].matched,
            "Detailed comparison changed the wide-integer elementwise decision.");

    const auto runtimeIntegerComparison =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)));
    require(!runtimeIntegerComparison.passed() && runtimeIntegerComparison.mismatches == 1,
            "Runtime detailed comparison rounded distinct wide integers together.");

    ComparisonOptions subUnitIntegerTolerance;
    subUnitIntegerTolerance.absoluteTolerance = 0.5;
    subUnitIntegerTolerance.computeElementwiseStatistics = false;
    subUnitIntegerTolerance.computeFrobenius = false;
    subUnitIntegerTolerance.maxReportedMismatches = 0;
    require(!compare(Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerObserved)),
                     Tensor::copyNativeStorage(std::span<const uint64_t>(wideIntegerExpected)),
                     subUnitIntegerTolerance)
                 .passed(),
            "Runtime comparison lost a sub-unit tolerance at the uint64 precision boundary.");

    const std::array<uint64_t, 2> exactUnsignedObserved{
        uint64_t{1} << 53,
        std::numeric_limits<uint64_t>::max() - 1,
    };
    const std::array<uint64_t, 2> exactUnsignedExpected{
        (uint64_t{1} << 53) + 1,
        std::numeric_limits<uint64_t>::max(),
    };
    ComparisonOptions exactUnsignedOptions;
    exactUnsignedOptions.computeUlp = true;
    exactUnsignedOptions.ulpType = ScalarType::UInt64;
    exactUnsignedOptions.maximumUlpTolerance = 0.0;
    exactUnsignedOptions.maxReportedMismatches = 1;
    const ComparisonReport exactUnsignedResult =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedExpected)),
                exactUnsignedOptions);
    require(exactUnsignedResult.mismatches == 2 &&
                exactUnsignedResult.maxAbsoluteDifference == 1.0 &&
                exactUnsignedResult.frobeniusDifference == std::sqrt(2.0) &&
                exactUnsignedResult.maximumUlp == 1.0 && exactUnsignedResult.sumUlp == 2.0 &&
                exactUnsignedResult.averageUlp == 1.0 && !exactUnsignedResult.ulpPassed &&
                exactUnsignedResult.reportedMismatches[0].absoluteDifference == 1.0,
            "UInt64 comparison evidence lost adjacent differences above 2^53.");

    const std::array<int64_t, 3> exactSignedObserved{
        int64_t{1} << 53,
        std::numeric_limits<int64_t>::max() - 1,
        std::numeric_limits<int64_t>::lowest(),
    };
    const std::array<int64_t, 3> exactSignedExpected{
        (int64_t{1} << 53) + 1,
        std::numeric_limits<int64_t>::max(),
        std::numeric_limits<int64_t>::lowest() + 1,
    };
    ComparisonOptions exactSignedOptions = exactUnsignedOptions;
    exactSignedOptions.ulpType = ScalarType::Int64;
    const ComparisonReport exactSignedResult =
        compare(Tensor::copyNativeStorage(std::span<const int64_t>(exactSignedObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(exactSignedExpected)),
                exactSignedOptions);
    require(exactSignedResult.mismatches == 3 && exactSignedResult.maxAbsoluteDifference == 1.0 &&
                exactSignedResult.frobeniusDifference == std::sqrt(3.0) &&
                exactSignedResult.maximumUlp == 1.0 && exactSignedResult.sumUlp == 3.0 &&
                exactSignedResult.averageUlp == 1.0 && !exactSignedResult.ulpPassed,
            "Int64 comparison evidence lost adjacent differences at its precision boundaries.");

    const std::array<uint64_t, 1> mixedUnsignedObserved{uint64_t{1} << 53};
    const std::array<int64_t, 1> mixedSignedExpected{(int64_t{1} << 53) + 1};
    const ComparisonReport mixedIntegerResult =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(mixedUnsignedObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(mixedSignedExpected)),
                exactUnsignedOptions);
    require(mixedIntegerResult.mismatches == 1 && mixedIntegerResult.maxAbsoluteDifference == 1.0 &&
                mixedIntegerResult.maximumUlp == 1.0 && !mixedIntegerResult.ulpPassed,
            "Mixed UInt64/Int64 comparison lost an adjacent difference above 2^53.");

    const std::array<uint64_t, 1> unsignedMaximum{std::numeric_limits<uint64_t>::max()};
    const std::array<uint64_t, 1> unsignedZero{};
    const ComparisonReport unsignedExtremeResult = compare(
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedMaximum)),
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedZero)), exactUnsignedOptions);
    require(unsignedExtremeResult.maximumUlp ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()) &&
                unsignedExtremeResult.maxAbsoluteDifference ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()),
            "UInt64 comparison evidence overflowed at the full-width difference.");

    const std::array<int64_t, 1> negativeOne{-1};
    const ComparisonReport mixedSignExtremeResult = compare(
        Tensor::copyNativeStorage(std::span<const uint64_t>(unsignedMaximum)),
        Tensor::copyNativeStorage(std::span<const int64_t>(negativeOne)), exactUnsignedOptions);
    require(mixedSignExtremeResult.mismatches == 1 &&
                mixedSignExtremeResult.maximumUlp == std::ldexp(1.0, 64) &&
                mixedSignExtremeResult.maxAbsoluteDifference == std::ldexp(1.0, 64),
            "Mixed signed/unsigned comparison confused -1 with UInt64 maximum.");

    const std::array<int64_t, 1> signedIntegerObserved{std::numeric_limits<int64_t>::lowest()};
    const std::array<int64_t, 1> signedIntegerExpected{std::numeric_limits<int64_t>::max()};
    const ComparisonReport signedExtremeResult =
        compare(Tensor::copyNativeStorage(std::span<const int64_t>(signedIntegerObserved)),
                Tensor::copyNativeStorage(std::span<const int64_t>(signedIntegerExpected)),
                exactSignedOptions);
    require(!signedExtremeResult.passed() &&
                signedExtremeResult.maximumUlp ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()) &&
                signedExtremeResult.maxAbsoluteDifference ==
                    static_cast<double>(std::numeric_limits<uint64_t>::max()),
            "Runtime comparison overflowed the signed-integer decision.");
}
}  // namespace host_numerics_test
