// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "host_numerics_test_support.hpp"

namespace host_numerics_test {
void testComparisonMetrics() {
    using namespace roc::host_numerics;

    const std::array<double, 3> expected{3.0, 4.0, 0.0};
    const std::array<double, 3> observed{0.0, 4.0, 3.0};
    ComparisonOptions metrics;
    metrics.allClose = false;
    metrics.relativeFrobeniusTolerance = 0.9;
    metrics.computeUlp = true;
    metrics.ulpType = ScalarType::Float64;
    const auto metricResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(observed)),
                Tensor::copyNativeStorage(std::span<const double>(expected)), metrics);
    require(std::abs(metricResult.frobeniusExpected - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusObserved - 5.0) < 1e-12 &&
                std::abs(metricResult.frobeniusDifference - std::sqrt(18.0)) < 1e-12 &&
                std::abs(metricResult.relativeFrobeniusError - std::sqrt(18.0) / 5.0) < 1e-12 &&
                std::abs(metricResult.relativeMaximumError - 0.75) < 1e-12 &&
                !metricResult.allCloseEvaluated && metricResult.frobeniusEvaluated &&
                !metricResult.ulpEvaluated && metricResult.frobeniusPassed,
            "Comparison Frobenius evidence is incorrect.");

    const std::array<double, 1> unitExpected{1.0};
    const std::array<double, 1> unitDifference{2.0};
    ComparisonOptions normOptions;
    normOptions.allClose = false;
    normOptions.computeElementwiseStatistics = true;
    normOptions.computeFrobenius = true;
    normOptions.relativeFrobeniusTolerance = 1.0;
    require(compare(Tensor::copyNativeStorage(std::span<const double>(unitDifference)),
                    Tensor::copyNativeStorage(std::span<const double>(unitExpected)), normOptions)
                .passed(),
            "Relative Frobenius comparison rejected its inclusive tolerance boundary.");

    const std::array<double, 1> zeroNorm{};
    normOptions.zeroExpectedNormIsNaN = true;
    const ComparisonReport zeroNormResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(zeroNorm)),
                Tensor::copyNativeStorage(std::span<const double>(zeroNorm)), normOptions);
    require(std::isnan(zeroNormResult.relativeFrobeniusError) &&
                std::isnan(zeroNormResult.relativeMaximumError) && !zeroNormResult.passed(),
            "IEEE zero-norm comparison policy did not preserve 0/0 as NaN.");

    const std::array<double, 1> infiniteNormValue{std::numeric_limits<double>::infinity()};
    normOptions.nonFiniteValuesInvalidateRelativeNorms = true;
    const ComparisonReport nonFiniteNormResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(infiniteNormValue)),
                Tensor::copyNativeStorage(std::span<const double>(infiniteNormValue)), normOptions);
    require(std::isnan(nonFiniteNormResult.relativeFrobeniusError) &&
                std::isnan(nonFiniteNormResult.relativeMaximumError) &&
                !nonFiniteNormResult.passed(),
            "Non-finite norm invalidation did not produce a failing NaN ratio.");
    ComparisonOptions sampledMetrics = metrics;
    sampledMetrics.selection = OutputSelection::strided(1, 2);
    const auto sampledMetricResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(observed)),
                Tensor::copyNativeStorage(std::span<const double>(expected)), sampledMetrics);
    require(sampledMetricResult.compared == 1,
            "Irregular comparison selection visited the wrong element count.");

    const double oneUlp = std::ldexp(1.0, -52);
    const std::array<double, 1> ulpObserved{1.0 + oneUlp};
    const std::array<double, 1> ulpExpected{1.0};
    ComparisonOptions ulp;
    ulp.computeUlp = true;
    ulp.ulpType = ScalarType::Float64;
    ulp.maximumUlpTolerance = 1.0;
    const auto ulpResult =
        compare(Tensor::copyNativeStorage(std::span<const double>(ulpObserved)),
                Tensor::copyNativeStorage(std::span<const double>(ulpExpected)), ulp);
    require(ulpResult.maximumUlp == 1.0 && ulpResult.averageUlp == 1.0 &&
                ulpResult.allCloseEvaluated && !ulpResult.frobeniusEvaluated &&
                ulpResult.ulpEvaluated && ulpResult.ulpPassed,
            "Comparison ULP evidence is incorrect.");
    require(encodedUlpDistance(0.0, static_cast<double>(std::numeric_limits<float>::denorm_min()),
                               ScalarType::Float32) == 1.0,
            "Encoded ULP distance mishandled the F32 zero/subnormal boundary.");
    require(encodedUlpDistance(1.0, 1.5, ScalarType::Float4E2M1) == 1.0,
            "Encoded ULP distance mishandled a packed scalar sign width.");
    require(encodedUlpDistance(1.0, 2.0, ScalarType::E5M3) == 8.0,
            "Encoded ULP distance treated unsigned E5M3 as a signed encoding.");

    const std::array<std::complex<double>, 2> complexExpected{
        std::complex<double>(std::numeric_limits<double>::infinity(), 2.0),
        std::complex<double>(std::numeric_limits<double>::quiet_NaN(), 4.0)};
    const auto complexObserved = complexExpected;
    ComparisonOptions nonFinite;
    nonFinite.equalNaNs = true;
    const auto nonFiniteResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexObserved)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexExpected)),
                nonFinite);
    require(nonFiniteResult.passed() && nonFiniteResult.matchedInfinities == 1 &&
                nonFiniteResult.matchedNaNs == 1,
            "Complex non-finite comparison policy is incorrect.");
    const std::array<double, 1> doubleInfinity{std::numeric_limits<double>::infinity()};
    const std::array<float, 1> floatInfinity{std::numeric_limits<float>::infinity()};
    ComparisonOptions noElementwiseStatistics;
    noElementwiseStatistics.computeElementwiseStatistics = false;
    noElementwiseStatistics.computeFrobenius = false;
    const ComparisonReport noElementwiseStatisticsResult = compare(
        Tensor::copyNativeStorage(std::span<const double>(doubleInfinity)),
        Tensor::copyNativeStorage(std::span<const float>(floatInfinity)), noElementwiseStatistics);
    require(noElementwiseStatisticsResult.passed() &&
                noElementwiseStatisticsResult.matchedInfinities == 0,
            "Disabled elementwise statistics collected matched infinities.");

    const ComparisonOptions numpyDefaults = allCloseComparisonOptions();
    require(numpyDefaults.absoluteTolerance == 1e-8 && numpyDefaults.relativeTolerance == 1e-5 &&
                !numpyDefaults.equalNaNs &&
                numpyDefaults.complexComparisonMode == ComplexComparisonMode::Magnitude,
            "Allclose options do not expose NumPy's finite-value defaults.");
    const std::array<double, 1> lowerValue{8.0};
    const std::array<double, 1> referenceValue{10.0};
    ComparisonOptions numpyBoundary = allCloseComparisonOptions(0.0, 0.2);
    numpyBoundary.computeFrobenius = false;
    require(
        compare(Tensor::copyNativeStorage(std::span<const double>(lowerValue)),
                Tensor::copyNativeStorage(std::span<const double>(referenceValue)), numpyBoundary)
            .passed(),
        "Allclose rejected the inclusive NumPy boundary.");
    require(!compare(Tensor::copyNativeStorage(std::span<const double>(referenceValue)),
                     Tensor::copyNativeStorage(std::span<const double>(lowerValue)), numpyBoundary)
                 .passed(),
            "Allclose lost NumPy's expected-reference asymmetry.");

    const std::array<std::complex<double>, 1> componentwiseObserved{std::complex<double>(1.0, 1.0)};
    const std::array<std::complex<double>, 1> componentwiseExpected{std::complex<double>(0.0, 0.0)};
    ComparisonOptions componentwiseComplex = allCloseComparisonOptions(1.0, 0.0);
    componentwiseComplex.computeFrobenius = false;
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseExpected)),
                     componentwiseComplex)
                 .passed(),
            "Complex allclose did not apply magnitude tolerance.");
    componentwiseComplex.complexComparisonMode = ComplexComparisonMode::Componentwise;
    require(
        compare(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(componentwiseObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(componentwiseExpected)),
            componentwiseComplex)
            .passed(),
        "Explicit componentwise complex comparison rejected passing components.");

    ComparisonOptions magnitudeAllCloseOnly = allCloseComparisonOptions(1.0, 0.0);
    magnitudeAllCloseOnly.computeElementwiseStatistics = false;
    magnitudeAllCloseOnly.computeFrobenius = false;
    magnitudeAllCloseOnly.selection = OutputSelection::all(IndexOrder::FirstDimensionFastest);
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(componentwiseExpected)),
                     magnitudeAllCloseOnly)
                 .passed(),
            "The optimized all-close-only path ignored complex magnitude mode.");

    const std::array<std::complex<double>, 1> magnitudeBoundaryObserved{
        std::complex<double>(0.0, 0.0)};
    const std::array<std::complex<double>, 1> magnitudeBoundaryExpected{
        std::complex<double>(3.0, 4.0)};
    ComparisonOptions magnitudeBoundary = allCloseComparisonOptions(0.0, 1.0);
    magnitudeBoundary.computeFrobenius = false;
    require(compare(Tensor::copyNativeStorage(
                        std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
                    Tensor::copyNativeStorage(
                        std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
                    magnitudeBoundary)
                .passed(),
            "Complex allclose rejected its inclusive magnitude boundary.");
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(magnitudeBoundaryExpected)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(magnitudeBoundaryObserved)),
                     magnitudeBoundary)
                 .passed(),
            "Complex allclose lost expected-magnitude asymmetry.");

    const double quietNaN = std::numeric_limits<double>::quiet_NaN();
    const std::array<std::complex<double>, 1> crossComponentNaNObserved{
        std::complex<double>(quietNaN, 1.0)};
    const std::array<std::complex<double>, 1> crossComponentNaNExpected{
        std::complex<double>(1.0, quietNaN)};
    ComparisonOptions magnitudeEqualNaNs = allCloseComparisonOptions(0.0, 0.0, true);
    magnitudeEqualNaNs.computeFrobenius = false;
    const ComparisonReport crossComponentNaNResult = compare(
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(crossComponentNaNObserved)),
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(crossComponentNaNExpected)),
        magnitudeEqualNaNs);
    require(crossComponentNaNResult.passed() && crossComponentNaNResult.matchedNaNs == 1,
            "Complex magnitude comparison did not match logical NaN values.");

    const double infinity = std::numeric_limits<double>::infinity();
    const std::array<std::complex<double>, 1> complexInfinity{
        std::complex<double>(infinity, infinity)};
    const ComparisonReport complexInfinityResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                magnitudeEqualNaNs);
    require(complexInfinityResult.passed() && complexInfinityResult.matchedInfinities == 1,
            "Complex magnitude comparison did not count a matched infinity as one logical value.");
    ComparisonOptions magnitudeInfinityWithFrobenius = allCloseComparisonOptions(0.0, 0.0, true);
    const ComparisonReport complexInfinityWithFrobeniusResult =
        compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexInfinity)),
                magnitudeInfinityWithFrobenius);
    require(complexInfinityWithFrobeniusResult.passed() &&
                complexInfinityWithFrobeniusResult.matchedInfinities == 1,
            "Complex magnitude infinity statistics depended on Frobenius evidence.");

    const std::array<std::complex<double>, 1> mismatchedComplexInfinityObserved{
        std::complex<double>(infinity, 1.0)};
    const std::array<std::complex<double>, 1> mismatchedComplexInfinityExpected{
        std::complex<double>(infinity, 2.0)};
    ComparisonOptions permissiveComplexInfinity = allCloseComparisonOptions(infinity, infinity);
    permissiveComplexInfinity.computeFrobenius = false;
    require(!compare(Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityObserved)),
                     Tensor::copyNativeStorage(
                         std::span<const std::complex<double>>(mismatchedComplexInfinityExpected)),
                     permissiveComplexInfinity)
                 .passed(),
            "Complex magnitude comparison applied finite tolerances to unequal infinities.");

    const std::array<double, 1> mixedReal{3.0};
    const std::array<std::complex<double>, 1> mixedComplex{std::complex<double>(3.0, 4.0)};
    ComparisonOptions mixedMagnitude = allCloseComparisonOptions(0.0, 1.0);
    mixedMagnitude.computeFrobenius = false;
    require(compare(Tensor::copyNativeStorage(std::span<const double>(mixedReal)),
                    Tensor::copyNativeStorage(std::span<const std::complex<double>>(mixedComplex)),
                    mixedMagnitude)
                .passed(),
            "Mixed real/complex comparison did not scale tolerance by the complex reference.");
    require(!compare(Tensor::copyNativeStorage(std::span<const std::complex<double>>(mixedComplex)),
                     Tensor::copyNativeStorage(std::span<const double>(mixedReal)), mixedMagnitude)
                 .passed(),
            "Mixed real/complex comparison lost expected-magnitude asymmetry.");

    const std::array<std::complex<double>, 1> signedZeroNaNObserved{
        std::complex<double>(quietNaN, 0.0)};
    const std::array<std::complex<double>, 1> signedZeroNaNExpected{
        std::complex<double>(quietNaN, -0.0)};
    ComparisonOptions signedZeroNaN = allCloseComparisonOptions(0.0, 0.0, true);
    signedZeroNaN.computeFrobenius = false;
    const ComparisonReport signedZeroNaNResult = compare(
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(signedZeroNaNObserved)),
        Tensor::copyNativeStorage(std::span<const std::complex<double>>(signedZeroNaNExpected)),
        signedZeroNaN);
    require(signedZeroNaNResult.passed() && signedZeroNaNResult.matchedNaNs == 1,
            "Logical complex NaN matching did not preserve NumPy signed-zero behavior.");

    const std::array<double, 4> absoluteCandidates{1e-6, 1e-5, 1e-4, 1e-3};
    const std::array<double, 1> relativeCandidates{0.0};
    const std::array<double, 1> closeObserved{1.00009};
    const std::array<double, 1> closeExpected{1.0};
    const auto tolerance = findAllCloseTolerance(
        Tensor::copyNativeStorage(std::span<const double>(closeObserved)),
        Tensor::copyNativeStorage(std::span<const double>(closeExpected)),
        std::span<const double>(absoluteCandidates), std::span<const double>(relativeCandidates));
    require(tolerance && tolerance->absolute == 1e-4 && tolerance->relative == 0.0,
            "Allclose tolerance search selected the wrong candidate.");

    const std::array<std::complex<double>, 1> complexSearchObserved{
        std::complex<double>(0.09, 0.09)};
    const std::array<std::complex<double>, 1> complexSearchExpected{std::complex<double>(0.0, 0.0)};
    const std::array<double, 1> complexSearchAbsoluteCandidates{0.1};
    const std::array<double, 1> complexSearchRelativeCandidates{0.0};
    require(
        !findAllCloseTolerance(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchExpected)),
            std::span<const double>(complexSearchAbsoluteCandidates),
            std::span<const double>(complexSearchRelativeCandidates)),
        "Allclose tolerance search defaulted complex values to componentwise comparison.");
    ComparisonOptions componentwiseSearch = allCloseComparisonOptions();
    componentwiseSearch.complexComparisonMode = ComplexComparisonMode::Componentwise;
    require(
        findAllCloseTolerance(
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchObserved)),
            Tensor::copyNativeStorage(std::span<const std::complex<double>>(complexSearchExpected)),
            std::span<const double>(complexSearchAbsoluteCandidates),
            std::span<const double>(complexSearchRelativeCandidates), componentwiseSearch)
            .has_value(),
        "Allclose tolerance search did not preserve explicit componentwise comparison.");

    std::array<float, 5> rangedSentinel{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), 0.0f, std::numeric_limits<float>::infinity()};
    const SentinelReport rangedSentinelResult = checkUnwrittenSentinel(
        ScalarType::Float32, std::as_bytes(std::span<const float>(rangedSentinel)), 2, 2,
        SentinelRegion::Before);
    require(rangedSentinelResult.checked == 2 && rangedSentinelResult.mismatches == 1 &&
                rangedSentinelResult.reportedMismatches[0].region == SentinelRegion::Before &&
                rangedSentinelResult.reportedMismatches[0].index == 3,
            "Sentinel range did not report an absolute storage element index.");

    std::array<float, 5> guarded{
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    Tensor guardedView = Tensor::copyNativeStorage<float>(Layout(Shape{2, 2}, {1, 3}),
                                                          std::span<const float>(guarded));
    require(checkUnusedTensorStorage(guardedView, guarded.size()).passed(),
            "Unwritten tensor padding sentinel was rejected.");
    const float writtenPadding = 0.0f;
    std::memcpy(guardedView.rawEncodedBackingStorage().data() + 2 * sizeof(float), &writtenPadding,
                sizeof(writtenPadding));
    const auto sentinel = checkUnusedTensorStorage(guardedView, guarded.size());
    require(
        !sentinel.passed() && sentinel.mismatches == 1 && sentinel.reportedMismatches[0].index == 2,
        "Written tensor padding was not detected.");
    bool rejectedOversizedSentinelAllocation = false;
    try {
        (void)checkUnusedTensorStorage(guardedView, guarded.size() + 1);
    } catch (const std::invalid_argument&) {
        rejectedOversizedSentinelAllocation = true;
    }
    require(rejectedOversizedSentinelAllocation,
            "Sentinel check accepted an allocation larger than its storage.");
}
}  // namespace host_numerics_test
