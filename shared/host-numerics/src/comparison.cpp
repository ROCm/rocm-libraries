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
#include "detail/threading.hpp"

namespace roc::host_numerics {
namespace {
template <typename Observe>
ComparisonReport accumulateComparison(const Tensor& observed, const Tensor& expected,
                                      const ComparisonOptions& options, Observe&& observe) {
    const size_t selectedCount = options.selection.selectedCount(observed.elementCount());
    constexpr size_t minimumElementsPerThread = 500'000;
    // Keep norm accumulation serial so its rounding order is independent of the OpenMP runtime.
    // Complete selections can be partitioned into ordered logical ranges without materializing
    // indices; sparse selections retain their existing traversal.
    const int threadCount =
        options.selection.selectsAll() && !options.computeFrobenius
            ? detail::operationThreadCount(selectedCount, minimumElementsPerThread)
            : 1;

    if (threadCount <= 1) {
        detail::ComparisonAccumulator accumulator(options, observed.shape());
        detail::forEachSelectedOffsetPair(
            observed.layout(), expected.layout(), options.selection,
            [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                observe(accumulator, logicalIndex, observedOffset, expectedOffset);
            });
        return accumulator.finish();
    }

    const size_t partitionCount = static_cast<size_t>(threadCount);
    std::vector<detail::ComparisonAccumulator> partials;
    partials.reserve(partitionCount);
    for (int thread = 0; thread < threadCount; ++thread)
        partials.emplace_back(options, observed.shape());

    detail::forEachParallelIndex(
        partitionCount, selectedCount, true, minimumElementsPerThread, [&](size_t partition) {
            const auto [firstSelected, pastLastSelected] =
                detail::evenlyPartitionedRange(selectedCount, partitionCount, partition);
            detail::forEachLayoutOffsetPairRange(
                observed.layout(), expected.layout(), options.selection.indexOrder(), firstSelected,
                pastLastSelected,
                [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                    observe(partials[partition], logicalIndex, observedOffset, expectedOffset);
                });
        });

    detail::ComparisonAccumulator accumulator(options, observed.shape());
    for (auto& partial : partials) accumulator.merge(std::move(partial));
    return accumulator.finish();
}
}  // namespace

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

    const auto observedStorage = observed.rawEncodedBackingStorage();
    const auto expectedStorage = expected.rawEncodedBackingStorage();

    if (observed.type() == expected.type()) {
        if (const auto encoded = detail::exactEncodedComparisonReport(observed, expected, options))
            return *encoded;
    }

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

    const auto integerCategory = [](ScalarCategory category) {
        return category == ScalarCategory::Boolean || category == ScalarCategory::SignedInteger ||
               category == ScalarCategory::UnsignedInteger;
    };
    if (integerCategory(scalarTypeInfo(observed.type()).category) &&
        integerCategory(scalarTypeInfo(expected.type()).category)) {
        return visitScalarType(observed.type(), [&]<typename ObservedTag>() -> ComparisonReport {
            return visitScalarType(
                expected.type(), [&]<typename ExpectedTag>() -> ComparisonReport {
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
                        return accumulateComparison(
                            observed, expected, options,
                            [&](detail::ComparisonAccumulator& accumulator, size_t logicalIndex,
                                ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                                accumulator.observeIntegral(
                                    logicalIndex, observedOffset, expectedOffset,
                                    detail::loadFastComparisonReal<ObservedTag>(observedStorage,
                                                                                observedOffset),
                                    detail::loadFastComparisonReal<ExpectedTag>(expectedStorage,
                                                                                expectedOffset));
                            });
                    }
                    throw std::logic_error("Unreachable non-integral comparison dispatch.");
                });
        });
    } else if (observed.type() == expected.type()) {
        return visitScalarType(observed.type(), [&]<typename Tag>() {
            return accumulateComparison(
                observed, expected, options,
                [&](detail::ComparisonAccumulator& accumulator, size_t logicalIndex,
                    ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                        const ComparisonValue observedValue =
                            detail::loadComparisonValueKnown<Tag>(observedStorage, observedOffset);
                        const ComparisonValue expectedValue =
                            detail::loadComparisonValueKnown<Tag>(expectedStorage, expectedOffset);
                        std::optional<bool> decision;
                        if (options.allClose) {
                            if (options.complexComparisonMode == ComplexComparisonMode::Magnitude)
                                decision.emplace(detail::compareComplexMagnitude(
                                                     observedValue, expectedValue, options)
                                                     .close);
                            else
                                decision.emplace(detail::valuesClose(observedValue.real,
                                                                     expectedValue.real, options) &&
                                                 detail::valuesClose(observedValue.imaginary,
                                                                     expectedValue.imaginary,
                                                                     options));
                        }
                        accumulator.observe(logicalIndex, observedOffset, expectedOffset,
                                            observedValue, expectedValue, decision);
                    } else {
                        const auto observedValue =
                            detail::loadFastComparisonReal<Tag>(observedStorage, observedOffset);
                        const auto expectedValue =
                            detail::loadFastComparisonReal<Tag>(expectedStorage, expectedOffset);
                        const std::optional<bool> decision =
                            options.allClose ? std::optional<bool>(detail::valuesCloseFast(
                                                   observedValue, expectedValue, options))
                                             : std::nullopt;
                        accumulator.observeReal(logicalIndex, observedOffset, expectedOffset,
                                                static_cast<double>(observedValue),
                                                static_cast<double>(expectedValue), decision);
                    }
                });
        });
    } else {
        return accumulateComparison(
            observed, expected, options,
            [&](detail::ComparisonAccumulator& accumulator, size_t logicalIndex,
                ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                accumulator.observe(logicalIndex, observedOffset, expectedOffset,
                                    detail::loadComparisonValue(observed, observedOffset),
                                    detail::loadComparisonValue(expected, expectedOffset));
            });
    }
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
