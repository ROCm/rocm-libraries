// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_numerics/operation_types.hpp>
#include <roc/host_numerics/tensor.hpp>
#include <span>
#include <vector>

namespace roc::host_numerics {
enum class UlpComparisonMode {
    /// `|expected - observed|` divided by the spacing at `expected`.
    RelativeSpacing,
    /// Distance between ordered encodings after conversion to `ulpType`; integer types use exact
    /// absolute difference before the result is represented as double, and types without an
    /// encoded-distance implementation use `RelativeSpacing`.
    EncodedDistance,
};

enum class ComplexComparisonMode {
    /// Compare real and imaginary components independently. Both component comparisons must pass.
    Componentwise,

    /// Compare the complex-difference magnitude against a tolerance scaled by the expected
    /// complex-value magnitude.
    Magnitude,
};

/// Configures three independent comparison criteria plus optional evidence and reporting.
///
/// For each finite real component, let
///
///   difference = |observed - expected|
///   tolerance = absoluteTolerance + relativeTolerance * |expected|
///               + symmetricRelativeTolerance * (|observed| + |expected| + 1)
///
/// The elementwise component passes when the values are exactly equal or `difference <= tolerance`.
/// `strictTolerance` changes the elementwise and relative-Frobenius tolerance tests to strict
/// inequality; exact elementwise equality still passes. Opposite signed zeros fail when
/// `equalSignedZero` is false. NaNs pass only when both values are NaN and `equalNaNs` is true.
/// Infinities pass only when they are equal.
///
/// With `symmetricRelativeTolerance == 0` and `strictTolerance == false`, this is NumPy's
/// asymmetric real-valued isclose rule: expected is the reference value. The symmetric term and
/// strict inequality are legacy host-numerics behavior, not NumPy behavior.
///
/// In `ComplexComparisonMode::Magnitude`, the finite-value rule instead uses
///
///   difference = |observed - expected|
///   tolerance = absoluteTolerance + relativeTolerance * |expected|
///               + symmetricRelativeTolerance * (|observed| + |expected| + 1)
///
/// where each absolute value is a complex magnitude. This is NumPy's finite complex `isclose`
/// formula when the symmetric term is zero and the comparison is non-strict. Magnitude mode
/// decodes both complex components to double before evaluating the formula, so storage-type
/// rounding at a tolerance boundary can differ from NumPy. A complex value containing any NaN is
/// one logical NaN; when `equalNaNs` is true, two such values match before signed-zero and infinity
/// classification. Non-NaN infinities match only when both complex components are exactly equal.
///
/// Pass/fail criteria are:
///
/// - `allClose`: every selected logical element must pass the component rule above.
/// - `relativeFrobeniusTolerance`: `||observed - expected||F / ||expected||F` must satisfy the
///   configured inclusive or strict tolerance. By default, a zero expected norm produces 0 for a
///   zero difference and infinity otherwise. This criterion requires `computeFrobenius`.
/// - `maximumUlpTolerance`: the maximum component ULP distance must be less than or equal to the
///   configured tolerance. This criterion requires `computeUlp` and a concrete `ulpType`.
struct ComparisonOptions {
    // Criteria. `allClose` directly enables the elementwise criterion. The Frobenius and ULP
    // criteria are enabled by their optional tolerance fields below.
    bool allClose = true;
    double absoluteTolerance = 0.0;
    double relativeTolerance = 0.0;
    double symmetricRelativeTolerance = 0.0;
    bool strictTolerance = false;
    bool equalNaNs = false;
    bool equalSignedZero = true;
    /// Preserve IEEE 0/0 = NaN instead of defining an exact zero norm ratio as zero.
    bool zeroExpectedNormIsNaN = false;
    /// Make relative norm evidence NaN when either tensor contains any non-finite value.
    bool nonFiniteValuesInvalidateRelativeNorms = false;
    ComplexComparisonMode complexComparisonMode = ComplexComparisonMode::Componentwise;

    // Evidence. These fields do not affect pass/fail by themselves.
    /// Collect non-finite/signed-zero counters and maximum elementwise differences.
    bool computeElementwiseStatistics = true;
    /// Collect observed, expected, and difference Frobenius norms and maximum magnitudes.
    bool computeFrobenius = true;
    /// Collect maximum, sum, average, and component count in the configured ULP representation.
    bool computeUlp = false;
    /// `std::nullopt` means no ULP representation is configured. `ScalarType::Count` is invalid.
    std::optional<ScalarType> ulpType;
    UlpComparisonMode ulpMode = UlpComparisonMode::EncodedDistance;

    // Criteria over collected evidence. A configured tolerance requires its compute flag.
    std::optional<double> relativeFrobeniusTolerance;
    std::optional<double> maximumUlpTolerance;

    // Reporting. `reportMatchingElements` records visited comparisons, including mismatches, in
    // `reportedComparisons`. `maxReportedMismatches` caps each reported vector independently.
    bool reportMatchingElements = false;
    size_t maxReportedMismatches = 10;
    OutputSelection selection;
};

struct ComparisonValue {
    double real = 0.0;
    double imaginary = 0.0;
    bool complex = false;
};

/// One reported logical tensor element. `index` is the selected element's linear logical index in
/// `ComparisonOptions::selection.indexOrder()`; `coordinates` are tensor coordinates; offsets are
/// element offsets from the start of each tensor's storage, not byte offsets. Reported values are
/// normalized to double and can round wide integers, while integer `absoluteDifference` is
/// computed exactly before conversion to double. For a complex value, `absoluteDifference` is the
/// complex-difference magnitude. `tolerance` is the larger component tolerance in componentwise
/// mode and the complex-value tolerance in magnitude mode.
struct Mismatch {
    size_t index = 0;
    std::vector<size_t> coordinates;
    ptrdiff_t observedOffset = 0;
    ptrdiff_t expectedOffset = 0;
    double observed = 0.0;
    double expected = 0.0;
    double observedImaginary = 0.0;
    double expectedImaginary = 0.0;
    double absoluteDifference = 0.0;
    double tolerance = 0.0;
    bool matched = false;
};

/// Comparison outcome, collected evidence, and bounded reports.
///
/// `allCloseEvaluated`, `frobeniusEvaluated`, and `ulpEvaluated` state whether the corresponding
/// criterion contributed to `passed()`. An unevaluated criterion retains a `...Passed` value of
/// true for source compatibility. Evidence can be present while its criterion is unevaluated.
struct ComparisonReport {
    /// Number of selected logical tensor elements. A complex element counts once.
    size_t compared = 0;

    /// Number of logical elements that failed the elementwise criterion; zero if it was disabled.
    size_t mismatches = 0;

    // Elementwise evidence. matchedNaNs and matchedInfinities count real components in
    // componentwise mode and logical values in magnitude mode. nonFiniteMismatches and
    // signedZeroMismatches always count logical values. For complex values, maxAbsoluteDifference
    // uses the complex-difference magnitude. The relative maxima use the maximum component ratio in
    // componentwise mode and the complex-magnitude ratio in magnitude mode. Relative evidence uses
    // expected as the reference; symmetric-relative evidence divides by
    // |observed| + |expected| + 1 using the corresponding component or complex magnitudes.
    size_t matchedNaNs = 0;
    size_t matchedInfinities = 0;
    size_t nonFiniteMismatches = 0;
    size_t signedZeroMismatches = 0;
    double maxAbsoluteDifference = 0.0;
    double maxRelativeDifference = 0.0;
    double maxSymmetricRelativeDifference = 0.0;

    // Norm evidence over selected finite values. Matched non-finite values are excluded.
    // A non-finite mismatch makes the difference and relative errors infinite unless
    // nonFiniteValuesInvalidateRelativeNorms requests NaN relative errors.
    double maximumObservedMagnitude = 0.0;
    double maximumExpectedMagnitude = 0.0;
    double frobeniusDifference = 0.0;
    double frobeniusObserved = 0.0;
    double frobeniusExpected = 0.0;
    double relativeFrobeniusError = 0.0;
    double relativeMaximumError = 0.0;

    // ULP evidence counts real components; each complex element contributes two components.
    // Matched NaNs and infinities count as compared components with zero distance. Integer input
    // differences for an integer ulpType are computed exactly before conversion to double.
    double maximumUlp = 0.0;
    double sumUlp = 0.0;
    double averageUlp = 0.0;
    size_t ulpCompared = 0;

    bool allCloseEvaluated = false;
    bool frobeniusEvaluated = false;
    bool ulpEvaluated = false;
    bool allClosePassed = true;
    bool frobeniusPassed = true;
    bool ulpPassed = true;
    std::vector<Mismatch> reportedMismatches;
    std::vector<Mismatch> reportedComparisons;

    /// Conjunction of the three `...Passed` fields. `compare` leaves an unevaluated criterion true,
    /// so this is the conjunction of enabled criteria and is true when none were enabled.
    bool passed() const {
        return allClosePassed && frobeniusPassed && ulpPassed;
    }
};

struct ComparisonTolerance {
    double absolute = 0.0;
    double relative = 0.0;
};

/// Caller-supplied classification for a checked sentinel element. The sentinel routines copy this
/// label into reports; they do not infer or validate where the checked range lies.
enum class SentinelRegion {
    /// No relationship to a logical allocation was specified.
    Unspecified,
    /// Guard storage preceding a logical allocation.
    Before,
    /// Allocated storage not addressed by the logical tensor layout, such as padding or holes.
    Inside,
    /// Guard storage following a logical allocation.
    After,
};

/// One overwritten sentinel. `index` is always an element offset from the start of the storage
/// span passed to the sentinel routine, independent of `region` and the checked range's start.
struct SentinelMismatch {
    SentinelRegion region = SentinelRegion::Unspecified;
    size_t index = 0;
    ComparisonValue observed;
};

struct SentinelReport {
    size_t checked = 0;
    size_t mismatches = 0;
    std::vector<SentinelMismatch> reportedMismatches;

    bool passed() const {
        return mismatches == 0;
    }

    void append(const SentinelReport& other, size_t maxReportedMismatches) {
        checked += other.checked;
        mismatches += other.mismatches;
        for (const auto& mismatch : other.reportedMismatches) {
            if (reportedMismatches.size() >= maxReportedMismatches) break;
            reportedMismatches.push_back(mismatch);
        }
    }
};

/// Legacy symmetric-relative tolerance used by `defaultComparisonOptions`.
inline constexpr double defaultSymmetricRelativeTolerance(ScalarType type) {
    switch (type) {
        case ScalarType::Float16:
            return 0.01;
        case ScalarType::BFloat16:
            return 0.1;
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E4M3Fnuz:
            return 0.125;
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E5M2Fnuz:
            return 0.25;
        case ScalarType::Float32:
        case ScalarType::ComplexFloat32:
            return 0.0002;
        case ScalarType::Float64:
        case ScalarType::ComplexFloat64:
            return 1e-12;
        default:
            return 0.0;
    }
}

/// Legacy host-numerics policy: type-specific symmetric-relative tolerance, strict inequality
/// when that tolerance is nonzero, unequal NaNs, elementwise statistics, and Frobenius evidence.
ComparisonOptions defaultComparisonOptions(
    ScalarType type, std::optional<double> symmetricRelativeTolerance = std::nullopt);

/// Absolute-only elementwise comparison with equal NaNs. This is not NumPy's default policy.
ComparisonOptions nearComparisonOptions(double absoluteTolerance);

/// NumPy-formula finite-value allclose options. Arguments remain absolute-then-relative for API
/// compatibility; the defaults are NumPy's `atol=1e-8` and `rtol=1e-5`. Complex values use
/// double-precision magnitude comparison as documented on `ComparisonOptions`.
ComparisonOptions allCloseComparisonOptions(double absoluteTolerance = 1e-8,
                                            double relativeTolerance = 1e-5,
                                            bool equalNaNs = false);

int ulpMantissaBits(ScalarType type);

double ulpDistance(double exact, double approximation, int mantissaBits);

double encodedUlpDistance(double exact, double approximation, ScalarType type);

ComparisonReport compare(const Tensor& observed, const Tensor& expected,
                         const ComparisonOptions& options = {});

/// Returns the first candidate pair, in absolute-major then relative-minor input order, for which
/// all enabled criteria pass. The search sets the asymmetric absolute and relative tolerances and
/// clears `symmetricRelativeTolerance`; all other supplied options remain active. The default uses
/// `allCloseComparisonOptions()`, including magnitude comparison for complex values.
std::optional<ComparisonTolerance> findAllCloseTolerance(
    const Tensor& observed, const Tensor& expected, std::span<const double> absoluteCandidates,
    std::span<const double> relativeCandidates,
    ComparisonOptions options = allCloseComparisonOptions());

/// Checks `[firstElement, firstElement + elementCount)` for the type's unwritten sentinel and
/// reports absolute storage element indices. Signed integers use the lowest value; unsigned
/// integers use the all-ones value; floating and scale types use infinity when supported and NaN
/// otherwise; complex types require both components to be infinite. Boolean has no unwritten
/// sentinel, so every checked Boolean element is a mismatch.
///
/// Throws `std::invalid_argument` when the scalar type has no storage, the element range
/// overflows, or the storage span does not contain the complete range.
SentinelReport checkUnwrittenSentinel(ScalarType type, std::span<const std::byte> storage,
                                      size_t firstElement, size_t elementCount,
                                      SentinelRegion region = SentinelRegion::Unspecified,
                                      size_t maxReportedMismatches = 10);

/// Checks every element in `[0, allocatedElements)` that is not addressed by `logicalTensor`.
/// Reported indices are storage element offsets in that same range. Throws `std::invalid_argument`
/// if the allocated range exceeds storage or any logical tensor coordinate addresses outside it.
SentinelReport checkUnusedTensorStorage(const Tensor& logicalTensor, size_t allocatedElements,
                                        SentinelRegion region = SentinelRegion::Inside,
                                        size_t maxReportedMismatches = 10);
}  // namespace roc::host_numerics
