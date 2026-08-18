// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <roc/host_validation/tensor.hpp>
#include <span>
#include <vector>

namespace roc::host_validation {
enum class ComparisonIndexOrder {
    FirstDimensionFastest,
    LastDimensionFastest,
};

enum class UlpComparisonMode {
    RelativeSpacing,
    EncodedDistance,
};

struct ComparisonSelection {
    size_t first = 0;
    size_t stride = 1;
    size_t maxElements = std::numeric_limits<size_t>::max();
    ComparisonIndexOrder indexOrder = ComparisonIndexOrder::LastDimensionFastest;
};

struct ComparisonOptions {
    bool pointwise = true;
    double absoluteTolerance = 0.0;
    double relativeTolerance = 0.0;
    double symmetricRelativeTolerance = 0.0;
    bool strictTolerance = false;
    bool equalNaNs = false;
    bool equalSignedZero = true;
    bool computePointwiseStatistics = true;
    bool computeFrobenius = true;
    bool computeUlp = false;
    ScalarType ulpType = ScalarType::Count;
    UlpComparisonMode ulpMode = UlpComparisonMode::EncodedDistance;
    std::optional<double> relativeFrobeniusTolerance;
    std::optional<double> maximumUlpTolerance;
    bool reportMatchingElements = false;
    size_t maxReportedMismatches = 10;
    ComparisonSelection selection;
};

struct ComparisonValue {
    double real = 0.0;
    double imaginary = 0.0;
    bool complex = false;
};

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

struct ComparisonResult {
    size_t compared = 0;
    size_t mismatches = 0;
    size_t matchedNaNs = 0;
    size_t matchedInfinities = 0;
    size_t nonFiniteMismatches = 0;
    size_t signedZeroMismatches = 0;
    double maxAbsoluteDifference = 0.0;
    double maxRelativeDifference = 0.0;
    double maxSymmetricRelativeDifference = 0.0;
    double maximumObservedMagnitude = 0.0;
    double maximumExpectedMagnitude = 0.0;
    double frobeniusDifference = 0.0;
    double frobeniusObserved = 0.0;
    double frobeniusExpected = 0.0;
    double relativeFrobeniusError = 0.0;
    double maximumUlp = 0.0;
    double sumUlp = 0.0;
    double averageUlp = 0.0;
    size_t ulpCompared = 0;
    bool pointwisePassed = true;
    bool frobeniusPassed = true;
    bool ulpPassed = true;
    std::vector<Mismatch> reportedMismatches;
    std::vector<Mismatch> reportedComparisons;

    bool passed() const {
        return pointwisePassed && frobeniusPassed && ulpPassed;
    }
};

using ComparisonPlan = ComparisonOptions;
using ComparisonReport = ComparisonResult;

struct ComparisonTolerance {
    double absolute = 0.0;
    double relative = 0.0;
};

enum class SentinelRegion {
    Unspecified,
    Before,
    Inside,
    After,
};

struct SentinelMismatch {
    SentinelRegion region = SentinelRegion::Unspecified;
    size_t index = 0;
    ComparisonValue observed;
};

struct SentinelResult {
    size_t checked = 0;
    size_t mismatches = 0;
    std::vector<SentinelMismatch> reportedMismatches;

    bool passed() const {
        return mismatches == 0;
    }

    void append(const SentinelResult& other, size_t maxReportedMismatches) {
        checked += other.checked;
        mismatches += other.mismatches;
        for (const auto& mismatch : other.reportedMismatches) {
            if (reportedMismatches.size() >= maxReportedMismatches) break;
            reportedMismatches.push_back(mismatch);
        }
    }
};

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

ComparisonOptions defaultComparisonOptions(
    ScalarType type, std::optional<double> symmetricRelativeTolerance = std::nullopt);

ComparisonOptions nearComparisonOptions(double absoluteTolerance);

ComparisonOptions allCloseComparisonOptions(double absoluteTolerance, double relativeTolerance,
                                            bool equalNaNs = false);

int ulpMantissaBits(ScalarType type);

double ulpDistance(double exact, double approximation, int mantissaBits);

double encodedUlpDistance(double exact, double approximation, ScalarType type);

ComparisonResult compare(const Tensor& observed, const Tensor& expected,
                         const ComparisonOptions& options = {});

std::optional<ComparisonTolerance> findAllCloseTolerance(const Tensor& observed,
                                                         const Tensor& expected,
                                                         std::span<const double> absoluteCandidates,
                                                         std::span<const double> relativeCandidates,
                                                         ComparisonOptions options = {});

SentinelResult checkUnwrittenSentinel(ScalarType type, std::span<const std::byte> storage,
                                      size_t firstElement, size_t elementCount,
                                      SentinelRegion region = SentinelRegion::Unspecified,
                                      size_t maxReportedMismatches = 10);

SentinelResult checkUnusedTensorStorage(const Tensor& logicalTensor, size_t allocatedElements,
                                        SentinelRegion region = SentinelRegion::Inside,
                                        size_t maxReportedMismatches = 10);
}  // namespace roc::host_validation
