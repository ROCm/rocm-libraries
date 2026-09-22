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

    ComparisonOptions encodedEquality;
    encodedEquality.equalNaNs = true;
    encodedEquality.computeElementwiseStatistics = false;
    encodedEquality.computeFrobenius = false;
    encodedEquality.maxReportedMismatches = 0;
    encodedEquality.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);

    const float quietNaN = std::numeric_limits<float>::quiet_NaN();
    const std::array<float, 4> identicalValues{quietNaN, -0.0f, 1.0f, -2.0f};
    const ComparisonReport identicalResult = compare(
        Tensor::copyNativeStorage(std::span<const float>(identicalValues)),
        Tensor::copyNativeStorage(std::span<const float>(identicalValues)), encodedEquality);
    require(identicalResult.passed() && identicalResult.compared == identicalValues.size() &&
                identicalResult.matchedNaNs == 0 && identicalResult.matchedInfinities == 0,
            "Encoded-equality comparison changed all-close-only report semantics.");

    ComparisonOptions identicalUlp = encodedEquality;
    identicalUlp.allClose = false;
    identicalUlp.computeUlp = true;
    identicalUlp.ulpType = ScalarType::Float32;
    identicalUlp.maximumUlpTolerance = 0.0;
    const ComparisonReport identicalUlpResult =
        compare(Tensor::copyNativeStorage(std::span<const float>(identicalValues)),
                Tensor::copyNativeStorage(std::span<const float>(identicalValues)), identicalUlp);
    require(identicalUlpResult.passed() && identicalUlpResult.compared == identicalValues.size() &&
                !identicalUlpResult.allCloseEvaluated && identicalUlpResult.ulpEvaluated &&
                identicalUlpResult.ulpCompared == identicalValues.size() &&
                identicalUlpResult.maximumUlp == 0.0 && identicalUlpResult.sumUlp == 0.0,
            "Encoded-equality comparison produced incorrect zero-distance ULP evidence.");

    const std::array<std::complex<float>, 2> identicalComplex{{{1.0f, 2.0f}, {-3.0f, 4.0f}}};
    identicalUlp.ulpType = ScalarType::ComplexFloat32;
    const ComparisonReport identicalComplexUlpResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<float>>(identicalComplex)),
                Tensor::copyNativeStorage(std::span<const std::complex<float>>(identicalComplex)),
                identicalUlp);
    require(identicalComplexUlpResult.passed() &&
                identicalComplexUlpResult.ulpCompared == 2 * identicalComplex.size(),
            "Encoded-equality comparison counted complex ULP components incorrectly.");

    const std::array<float, 3> ulpOnlyObserved{
        std::bit_cast<float>(uint32_t{0x7fc00001}),
        std::numeric_limits<float>::infinity(),
        std::nextafter(1.0f, 2.0f),
    };
    const std::array<float, 3> ulpOnlyExpected{
        std::bit_cast<float>(uint32_t{0x7fc00002}),
        std::numeric_limits<float>::infinity(),
        1.0f,
    };
    ComparisonOptions ulpOnly = identicalUlp;
    ulpOnly.maximumUlpTolerance = 1.0;
    const ComparisonReport ulpOnlyResult =
        compare(Tensor::copyNativeStorage(std::span<const float>(ulpOnlyObserved)),
                Tensor::copyNativeStorage(std::span<const float>(ulpOnlyExpected)), ulpOnly);
    require(ulpOnlyResult.passed() && ulpOnlyResult.ulpCompared == ulpOnlyExpected.size() &&
                ulpOnlyResult.maximumUlp == 1.0 && ulpOnlyResult.sumUlp == 1.0,
            "ULP-only comparison changed matching non-finite or finite-distance evidence.");
    ulpOnly.equalNaNs = false;
    require(!compare(Tensor::copyNativeStorage(std::span<const float>(ulpOnlyObserved)),
                     Tensor::copyNativeStorage(std::span<const float>(ulpOnlyExpected)), ulpOnly)
                 .passed(),
            "ULP-only comparison matched NaNs when equalNaNs was disabled.");

    for (size_t typeIndex = 0; typeIndex < scalarTypeCount; ++typeIndex) {
        const ScalarType type = static_cast<ScalarType>(typeIndex);
        Tensor expected(type, Shape{16});
        const Tensor observed = expected.deepCopy();
        const ComparisonReport result = compare(observed, expected, encodedEquality);
        require(result.passed() && result.compared == expected.elementCount(),
                "Encoded-equality comparison failed for a supported scalar type.");
    }

    ComparisonOptions unequalNaNs = encodedEquality;
    unequalNaNs.equalNaNs = false;
    const std::array<float, 1> identicalNaN{quietNaN};
    require(!compare(Tensor::copyNativeStorage(std::span<const float>(identicalNaN)),
                     Tensor::copyNativeStorage(std::span<const float>(identicalNaN)), unequalNaNs)
                 .passed(),
            "Encoded-equality comparison accepted NaNs when equalNaNs was disabled.");

    const std::array<float, 1> positiveZero{0.0f};
    const std::array<float, 1> negativeZero{-0.0f};
    require(
        compare(Tensor::copyNativeStorage(std::span<const float>(positiveZero)),
                Tensor::copyNativeStorage(std::span<const float>(negativeZero)), encodedEquality)
            .passed(),
        "Encoded-equality fallback distinguished signed zeros.");

    const std::array<std::byte, 2> packedExpected{std::byte{0xa5}, std::byte{0xbc}};
    const std::array<std::byte, 2> packedObserved{std::byte{0xa6}, std::byte{0xbc}};
    const Layout packedOffsetLayout(Shape{3}, {1}, 1);
    require(compare(Tensor::copyEncodedBackingStorage(ScalarType::Float4E2M1, packedOffsetLayout,
                                                      packedObserved),
                    Tensor::copyEncodedBackingStorage(ScalarType::Float4E2M1, packedOffsetLayout,
                                                      packedExpected),
                    encodedEquality)
                .passed(),
            "Encoded-equality optimization compared packed bits outside the logical tensor.");
    const std::array<std::byte, 2> packedMismatch{std::byte{0xa5}, std::byte{0xac}};
    require(!compare(Tensor::copyEncodedBackingStorage(ScalarType::Float4E2M1, packedOffsetLayout,
                                                       packedMismatch),
                     Tensor::copyEncodedBackingStorage(ScalarType::Float4E2M1, packedOffsetLayout,
                                                       packedExpected),
                     encodedEquality)
                 .passed(),
            "Encoded-equality optimization ignored a packed logical mismatch.");

    const std::array<float, 5> paddedExpected{1.0f, 2.0f, 99.0f, 3.0f, 4.0f};
    const std::array<float, 5> paddedObserved{1.0f, 2.0f, -99.0f, 3.0f, 4.0f};
    const Layout paddedLayout(Shape{2, 2}, {1, 3});
    require(compare(Tensor::copyNativeStorage(paddedLayout, std::span<const float>(paddedObserved)),
                    Tensor::copyNativeStorage(paddedLayout, std::span<const float>(paddedExpected)),
                    encodedEquality)
                .passed(),
            "Encoded-equality optimization compared padding outside the logical tensor.");

    const std::array<float, 1> closeObserved{1.0001f};
    const std::array<float, 1> closeExpected{1.0f};
    ComparisonOptions tolerantEquality = allCloseComparisonOptions(0.001, 0.0, true);
    tolerantEquality.computeElementwiseStatistics = false;
    tolerantEquality.computeFrobenius = false;
    tolerantEquality.maxReportedMismatches = 0;
    require(
        compare(Tensor::copyNativeStorage(std::span<const float>(closeObserved)),
                Tensor::copyNativeStorage(std::span<const float>(closeExpected)), tolerantEquality)
            .passed(),
        "Encoded-equality optimization bypassed the numerical tolerance fallback.");

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
    ComparisonOptions exactUnsignedUlpOnly = exactUnsignedOptions;
    exactUnsignedUlpOnly.allClose = false;
    exactUnsignedUlpOnly.computeElementwiseStatistics = false;
    exactUnsignedUlpOnly.computeFrobenius = false;
    exactUnsignedUlpOnly.maxReportedMismatches = 0;
    const ComparisonReport exactUnsignedUlpOnlyResult =
        compare(Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedObserved)),
                Tensor::copyNativeStorage(std::span<const uint64_t>(exactUnsignedExpected)),
                exactUnsignedUlpOnly);
    require(
        exactUnsignedUlpOnlyResult.maximumUlp == 1.0 && exactUnsignedUlpOnlyResult.sumUlp == 2.0 &&
            exactUnsignedUlpOnlyResult.ulpCompared == 2 && !exactUnsignedUlpOnlyResult.ulpPassed,
        "ULP-only UInt64 comparison lost exact differences above 2^53.");

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

    constexpr size_t largeComparisonElements = 600'003;
    std::vector<float> largeExpected(largeComparisonElements);
    for (size_t index = 0; index < largeExpected.size(); ++index)
        largeExpected[index] = static_cast<float>(static_cast<int>(index % 101) - 50);
    std::vector<float> largeObserved = largeExpected;
    for (const size_t index : {size_t{3}, largeComparisonElements / 2, largeComparisonElements - 2})
        largeObserved[index] += 1.0f;

    const Layout largeLayout(Shape{3, largeComparisonElements / 3}, {1, 3});
    const Tensor largeObservedTensor =
        Tensor::copyNativeStorage(largeLayout, std::span<const float>(largeObserved));
    const Tensor largeExpectedTensor =
        Tensor::copyNativeStorage(largeLayout, std::span<const float>(largeExpected));
    ComparisonOptions parallelOptions;
    parallelOptions.computeFrobenius = false;
    parallelOptions.computeUlp = true;
    parallelOptions.ulpType = ScalarType::Float32;
    parallelOptions.maxReportedMismatches = 3;
    parallelOptions.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    const ComparisonReport parallelResult =
        compare(largeObservedTensor, largeExpectedTensor, parallelOptions);

    ComparisonOptions serialOptions = parallelOptions;
    serialOptions.selection = OutputSelection::strided(0, 1, std::numeric_limits<size_t>::max(),
                                                       IndexOrder::FirstDimensionFastest);
    const ComparisonReport serialResult =
        compare(largeObservedTensor, largeExpectedTensor, serialOptions);
    require(parallelResult.compared == serialResult.compared &&
                parallelResult.mismatches == serialResult.mismatches &&
                parallelResult.matchedNaNs == serialResult.matchedNaNs &&
                parallelResult.matchedInfinities == serialResult.matchedInfinities &&
                parallelResult.nonFiniteMismatches == serialResult.nonFiniteMismatches &&
                parallelResult.maxAbsoluteDifference == serialResult.maxAbsoluteDifference &&
                parallelResult.maxRelativeDifference == serialResult.maxRelativeDifference &&
                parallelResult.maximumUlp == serialResult.maximumUlp &&
                parallelResult.sumUlp == serialResult.sumUlp &&
                parallelResult.reportedMismatches.size() == 3,
            "Parallel comparison reduction changed aggregate evidence.");
    for (size_t index = 0; index < parallelResult.reportedMismatches.size(); ++index)
        require(parallelResult.reportedMismatches[index].index ==
                    serialResult.reportedMismatches[index].index,
                "Parallel comparison reduction changed mismatch report order.");

    ComparisonOptions allCloseOnly = allCloseComparisonOptions(1.0, 0.0, true);
    allCloseOnly.maxReportedMismatches = 0;
    const ComparisonReport tolerantParallelResult =
        compare(largeObservedTensor, largeExpectedTensor, allCloseOnly);
    require(tolerantParallelResult.passed() &&
                tolerantParallelResult.compared == largeComparisonElements,
            "Parallel all-close comparison rejected values within tolerance.");
    allCloseOnly.absoluteTolerance = 0.0;
    const ComparisonReport exactParallelResult =
        compare(largeObservedTensor, largeExpectedTensor, allCloseOnly);
    require(!exactParallelResult.passed() && exactParallelResult.mismatches == 3,
            "Parallel all-close comparison lost mismatches.");

    const Layout largeLastDimensionLayout =
        Layout::contiguousLastDimensionFastest(largeLayout.shape());
    const Tensor largeLastDimensionObserved =
        Tensor::copyNativeStorage(largeLastDimensionLayout, std::span<const float>(largeObserved));
    const Tensor largeLastDimensionExpected =
        Tensor::copyNativeStorage(largeLastDimensionLayout, std::span<const float>(largeExpected));
    allCloseOnly.selection = OutputSelection::all(IndexOrder::LastDimensionFastest);
    const ComparisonReport lastDimensionParallelResult =
        compare(largeLastDimensionObserved, largeLastDimensionExpected, allCloseOnly);
    require(!lastDimensionParallelResult.passed() &&
                lastDimensionParallelResult.compared == largeComparisonElements &&
                lastDimensionParallelResult.mismatches == 3,
            "Last-dimension-fast parallel all-close comparison lost mismatches.");

    const std::array<float, 4> explicitlySelectedObserved{9.0f, 1.0f, 2.0f, 3.0f};
    const std::array<float, 4> explicitlySelectedExpected{0.0f, 1.0f, 2.0f, 3.0f};
    allCloseOnly.selection =
        OutputSelection::explicitIndices({1, 2}, IndexOrder::FirstDimensionFastest);
    const ComparisonReport explicitSelectionResult =
        compare(Tensor::copyNativeStorage(std::span<const float>(explicitlySelectedObserved)),
                Tensor::copyNativeStorage(std::span<const float>(explicitlySelectedExpected)),
                allCloseOnly);
    require(explicitSelectionResult.passed() && explicitSelectionResult.compared == 2,
            "All-close comparison treated explicit indices as a contiguous prefix.");
}
}  // namespace host_numerics_test
