// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/comparison_accumulator.hpp"
#include "detail/comparison_common.hpp"
#include "detail/comparison_fast_path.hpp"
#include "detail/comparison_iteration.hpp"

namespace roc::host_numerics {
ComparisonOptions nearComparisonOptions(double absoluteTolerance) {
    ComparisonOptions options;
    options.absoluteTolerance = absoluteTolerance;
    options.equalNaNs = true;
    return options;
}

ComparisonOptions allCloseComparisonOptions(double absoluteTolerance, double relativeTolerance,
                                            bool equalNaNs) {
    ComparisonOptions options;
    options.absoluteTolerance = absoluteTolerance;
    options.relativeTolerance = relativeTolerance;
    options.equalNaNs = equalNaNs;
    options.complexComparisonMode = ComplexComparisonMode::Magnitude;
    return options;
}

int ulpMantissaBits(ScalarType type) {
    const auto& info = scalarTypeInfo(type);
    if (info.category == ScalarCategory::SignedInteger ||
        info.category == ScalarCategory::UnsignedInteger ||
        info.category == ScalarCategory::Boolean)
        return 0;
    if (info.category == ScalarCategory::Scale) return info.mantissaBits;
    return info.mantissaBits;
}

double ulpDistance(double exact, double approximation, int mantissaBits) {
    if (exact == approximation) return 0.0;
    if (!std::isfinite(exact) || !std::isfinite(approximation))
        return std::numeric_limits<double>::infinity();

    int exponent = 0;
    const double mantissa = std::frexp(exact, &exponent);
    if (std::abs(mantissa) == 0.5) --exponent;

    const double ulpSize = std::ldexp(1.0, exponent - mantissaBits);
    if (ulpSize == 0.0) return std::numeric_limits<double>::infinity();
    return std::abs(exact - approximation) / ulpSize;
}

double encodedUlpDistance(double exact, double approximation, ScalarType type) {
    return detail::encodedUlpDistance(exact, approximation, type);
}

ComparisonReport compare(const Tensor& observed, const Tensor& expected,
                         const ComparisonOptions& options) {
    detail::validateComparisonOptions(options);
    if (observed.shape() != expected.shape())
        throw std::invalid_argument("Host numerics tensor comparison shape mismatch.");

    if (observed.type() == expected.type() && detail::allCloseOnlyComparison(options)) {
        ComparisonReport result = visitScalarType(observed.type(), [&]<typename Tag>() {
            return detail::compareAllCloseOnlyKnown<Tag>(observed, expected, options);
        });
        const bool needsSamples = options.maxReportedMismatches != 0 &&
                                  (options.reportMatchingElements || result.mismatches != 0);
        if (!needsSamples) return result;

        ComparisonOptions detailed = options;
        detailed.computeElementwiseStatistics = true;
        return compare(observed, expected, detailed);
    }

    detail::ComparisonAccumulator accumulator(options, observed.shape());
    const auto integerCategory = [](ScalarCategory category) {
        return category == ScalarCategory::Boolean || category == ScalarCategory::SignedInteger ||
               category == ScalarCategory::UnsignedInteger;
    };
    if (integerCategory(scalarTypeInfo(observed.type()).category) &&
        integerCategory(scalarTypeInfo(expected.type()).category)) {
        visitScalarType(observed.type(), [&]<typename ObservedTag>() {
            visitScalarType(expected.type(), [&]<typename ExpectedTag>() {
                constexpr ScalarCategory observedCategory =
                    scalarTypeInfo(ObservedTag::type).category;
                constexpr ScalarCategory expectedCategory =
                    scalarTypeInfo(ExpectedTag::type).category;
                if constexpr ((observedCategory == ScalarCategory::Boolean ||
                               observedCategory == ScalarCategory::SignedInteger ||
                               observedCategory == ScalarCategory::UnsignedInteger) &&
                              (expectedCategory == ScalarCategory::Boolean ||
                               expectedCategory == ScalarCategory::SignedInteger ||
                               expectedCategory == ScalarCategory::UnsignedInteger)) {
                    detail::forEachSelectedOffsetPair(
                        observed.layout(), expected.layout(), options.selection,
                        [&](size_t logicalIndex, ptrdiff_t observedOffset,
                            ptrdiff_t expectedOffset) {
                            accumulator.observeIntegral(
                                logicalIndex, observedOffset, expectedOffset,
                                detail::loadFastComparisonReal<ObservedTag>(
                                    observed.rawEncodedBackingStorage(), observedOffset),
                                detail::loadFastComparisonReal<ExpectedTag>(
                                    expected.rawEncodedBackingStorage(), expectedOffset));
                        });
                }
            });
        });
    } else if (observed.type() == expected.type()) {
        visitScalarType(observed.type(), [&]<typename Tag>() {
            detail::forEachSelectedOffsetPair(
                observed.layout(), expected.layout(), options.selection,
                [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                    std::optional<bool> decision;
                    if (options.allClose)
                        decision.emplace(detail::knownAllCloseDecision<Tag>(
                            observed, observedOffset, expected, expectedOffset, options));
                    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                        accumulator.observe(
                            logicalIndex, observedOffset, expectedOffset,
                            detail::loadComparisonValueKnown<Tag>(
                                observed.rawEncodedBackingStorage(), observedOffset),
                            detail::loadComparisonValueKnown<Tag>(
                                expected.rawEncodedBackingStorage(), expectedOffset),
                            decision);
                    } else {
                        accumulator.observeReal(
                            logicalIndex, observedOffset, expectedOffset,
                            detail::loadComparisonValueKnown<Tag>(
                                observed.rawEncodedBackingStorage(), observedOffset)
                                .real,
                            detail::loadComparisonValueKnown<Tag>(
                                expected.rawEncodedBackingStorage(), expectedOffset)
                                .real,
                            decision);
                    }
                });
        });
    } else {
        detail::forEachSelectedOffsetPair(
            observed.layout(), expected.layout(), options.selection,
            [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                accumulator.observe(logicalIndex, observedOffset, expectedOffset,
                                    detail::loadComparisonValue(observed, observedOffset),
                                    detail::loadComparisonValue(expected, expectedOffset));
            });
    }
    return accumulator.finish();
}

std::string formatComparisonReport(const ComparisonReport& report) {
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10) << "comparison "
           << (report.passed() ? "passed" : "failed") << ": " << report.mismatches << " of "
           << report.compared << " compared elements mismatched";
    for (const Mismatch& mismatch : report.reportedMismatches) {
        output << "\n  index " << mismatch.index << " [";
        for (size_t dimension = 0; dimension < mismatch.coordinates.size(); ++dimension) {
            if (dimension != 0) output << ", ";
            output << mismatch.coordinates[dimension];
        }
        output << "]: observed ";
        const bool complex = mismatch.observedImaginary != 0.0 ||
                             mismatch.expectedImaginary != 0.0 ||
                             !std::isfinite(mismatch.observedImaginary) ||
                             !std::isfinite(mismatch.expectedImaginary);
        if (complex)
            output << '(' << mismatch.observed << ", " << mismatch.observedImaginary << ')';
        else
            output << mismatch.observed;
        output << ", expected ";
        if (complex)
            output << '(' << mismatch.expected << ", " << mismatch.expectedImaginary << ')';
        else
            output << mismatch.expected;
        output << ", absolute difference " << mismatch.absoluteDifference << ", tolerance "
               << mismatch.tolerance;
    }
    if (report.mismatches > report.reportedMismatches.size())
        output << "\n  " << report.mismatches - report.reportedMismatches.size()
               << " additional mismatches not shown";
    return output.str();
}

std::optional<ComparisonTolerance> findAllCloseTolerance(const Tensor& observed,
                                                         const Tensor& expected,
                                                         std::span<const double> absoluteCandidates,
                                                         std::span<const double> relativeCandidates,
                                                         ComparisonOptions options) {
    for (const double absolute : absoluteCandidates) {
        for (const double relative : relativeCandidates) {
            options.absoluteTolerance = absolute;
            options.relativeTolerance = relative;
            if (compare(observed, expected, options).passed())
                return ComparisonTolerance{absolute, relative};
        }
    }
    return std::nullopt;
}

SentinelReport checkUnwrittenSentinel(ScalarType type, std::span<const std::byte> storage,
                                      size_t firstElement, size_t elementCount,
                                      SentinelRegion region, size_t maxReportedMismatches) {
    detail::validateSentinelRange(type, storage, firstElement, elementCount);

    SentinelReport result;
    result.checked = elementCount;
    result.reportedMismatches.reserve(std::min(maxReportedMismatches, elementCount));
    for (size_t index = 0; index < elementCount; ++index) {
        const ptrdiff_t offset = static_cast<ptrdiff_t>(firstElement + index);
        ComparisonValue value;
        if (scalarTypeInfo(type).category == ScalarCategory::Complex)
            value = detail::typedComparisonValue(
                detail::decodeScalar<std::complex<double>>(type, storage, offset));
        else
            value =
                detail::typedComparisonValue(detail::decodeScalar<double>(type, storage, offset));
        if (!detail::isUnwrittenSentinelValue(type, value)) {
            ++result.mismatches;
            if (result.reportedMismatches.size() < maxReportedMismatches)
                result.reportedMismatches.push_back({region, static_cast<size_t>(offset), value});
        }
    }
    return result;
}

SentinelReport checkUnusedTensorStorage(const Tensor& logicalTensor, size_t allocatedElements,
                                        SentinelRegion region, size_t maxReportedMismatches) {
    const auto& layout = logicalTensor.layout();
    detail::validateSentinelRange(logicalTensor.type(), logicalTensor.rawEncodedBackingStorage(), 0,
                                  allocatedElements);
    std::vector<bool> used(allocatedElements, false);
    detail::forEachSelectedIndex(
        layout.shape(), {}, [&](size_t, std::span<const size_t> coordinates) {
            const ptrdiff_t offset = layout.elementOffset(coordinates);
            if (offset < 0 || static_cast<size_t>(offset) >= allocatedElements)
                throw std::invalid_argument("Logical tensor addresses outside allocated storage.");
            used[static_cast<size_t>(offset)] = true;
        });

    SentinelReport result;
    result.reportedMismatches.reserve(maxReportedMismatches);
    for (size_t index = 0; index < allocatedElements; ++index) {
        if (used[index]) continue;
        const SentinelReport element = checkUnwrittenSentinel(
            logicalTensor.type(), logicalTensor.rawEncodedBackingStorage(), index, 1, region,
            maxReportedMismatches - result.reportedMismatches.size());
        result.append(element, maxReportedMismatches);
    }
    return result;
}
}  // namespace roc::host_numerics
