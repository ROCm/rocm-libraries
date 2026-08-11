// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace roc::host_validation {
namespace detail {
inline bool pointwiseOnlyComparison(const ComparisonOptions& options) {
    return options.pointwise && !options.computePointwiseStatistics && !options.computeFrobenius &&
           !options.computeUlp && !options.relativeFrobeniusTolerance &&
           !options.maximumUlpTolerance;
}

inline bool oppositeZeroSigns(double observed, double expected) {
    return observed == 0.0 && expected == 0.0 && std::signbit(observed) != std::signbit(expected);
}

template <typename T>
ComparisonValue typedComparisonValue(const T& value) {
    using Value = std::remove_cvref_t<T>;
    if constexpr (RuntimeIsComplexV<Value>) {
        return {
            static_cast<double>(value.real()),
            static_cast<double>(value.imag()),
            true,
        };
    } else {
        return {static_cast<double>(value), 0.0, false};
    }
}

template <typename Value>
auto fastTypedComparisonValue(const Value& value) {
    using Type = std::remove_cvref_t<Value>;
    if constexpr (std::is_same_v<Type, double> || std::is_integral_v<Type>)
        return value;
    else if constexpr (std::is_same_v<Type, float>)
        return value;
    else
        return static_cast<float>(value);
}

template <typename T>
long double integralMagnitude(T value) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_same_v<T, bool>) {
        return value ? 1.0L : 0.0L;
    } else if constexpr (std::is_unsigned_v<T>) {
        return static_cast<long double>(value);
    } else {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned magnitude =
            value < 0 ? static_cast<Unsigned>(-(value + 1)) + 1 : static_cast<Unsigned>(value);
        return static_cast<long double>(magnitude);
    }
}

template <typename Observed, typename Expected>
long double integralDifference(Observed observed, Expected expected) {
    static_assert(std::is_integral_v<Observed> && std::is_integral_v<Expected>);
    if constexpr (std::is_same_v<Observed, Expected>) {
        using Type = Observed;
        if constexpr (std::is_unsigned_v<Type>) {
            const Type difference =
                observed >= expected ? observed - expected : expected - observed;
            return static_cast<long double>(difference);
        } else {
            const bool observedNegative = observed < 0;
            const bool expectedNegative = expected < 0;
            const long double observedMagnitude = integralMagnitude(observed);
            const long double expectedMagnitude = integralMagnitude(expected);
            if (observedNegative != expectedNegative) return observedMagnitude + expectedMagnitude;
            return std::abs(observedMagnitude - expectedMagnitude);
        }
    } else {
        return std::abs(static_cast<long double>(observed) - static_cast<long double>(expected));
    }
}

template <typename Observed, typename Expected>
bool valuesCloseFast(Observed observed, Expected expected, const ComparisonOptions& options) {
    if (observed == expected)
        return options.equalSignedZero || !oppositeZeroSigns(observed, expected);
    if constexpr (std::is_integral_v<Observed> && std::is_integral_v<Expected>) {
        const long double observedMagnitude = integralMagnitude(observed);
        const long double expectedMagnitude = integralMagnitude(expected);
        const long double difference = integralDifference(observed, expected);
        const long double tolerance =
            options.absoluteTolerance + options.relativeTolerance * expectedMagnitude +
            options.symmetricRelativeTolerance * (observedMagnitude + expectedMagnitude + 1.0L);
        return options.strictTolerance ? difference < tolerance : difference <= tolerance;
    } else {
        if (std::isnan(observed) || std::isnan(expected))
            return options.equalNaNs && std::isnan(observed) && std::isnan(expected);
        if (std::isinf(observed) || std::isinf(expected)) return false;

        const double difference = std::abs(observed - expected);
        const double tolerance =
            options.absoluteTolerance + options.relativeTolerance * std::abs(expected) +
            options.symmetricRelativeTolerance * (std::abs(observed) + std::abs(expected) + 1.0);
        return options.strictTolerance ? difference < tolerance : difference <= tolerance;
    }
}

struct ComponentResult {
    bool close = false;
    bool matchedNaN = false;
    bool matchedInfinity = false;
    bool nonFiniteMismatch = false;
    bool signedZeroMismatch = false;
    double difference = 0.0;
    double tolerance = 0.0;
    double relativeDifference = 0.0;
    double symmetricRelativeDifference = 0.0;
};

inline ComponentResult compareComponent(double observed, double expected,
                                        const ComparisonOptions& options) {
    ComponentResult result;

    if (std::isnan(observed) || std::isnan(expected)) {
        result.matchedNaN = options.equalNaNs && std::isnan(observed) && std::isnan(expected);
        result.close = result.matchedNaN;
        result.nonFiniteMismatch = !result.close;
        result.difference = result.close ? 0.0 : std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        result.symmetricRelativeDifference = result.difference;
        return result;
    }

    if (std::isinf(observed) || std::isinf(expected)) {
        result.matchedInfinity = observed == expected;
        result.close = result.matchedInfinity;
        result.nonFiniteMismatch = !result.close;
        result.difference = result.close ? 0.0 : std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        result.symmetricRelativeDifference = result.difference;
        return result;
    }

    if (!options.equalSignedZero && oppositeZeroSigns(observed, expected)) {
        result.signedZeroMismatch = true;
        result.close = false;
        return result;
    }

    result.difference = std::abs(observed - expected);
    result.tolerance =
        options.absoluteTolerance + options.relativeTolerance * std::abs(expected) +
        options.symmetricRelativeTolerance * (std::abs(observed) + std::abs(expected) + 1.0);
    result.relativeDifference =
        expected == 0.0 ? (result.difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                        : result.difference / std::abs(expected);
    result.symmetricRelativeDifference =
        result.difference / (std::abs(observed) + std::abs(expected) + 1.0);
    result.close =
        observed == expected || (options.strictTolerance ? result.difference < result.tolerance
                                                         : result.difference <= result.tolerance);
    return result;
}

inline bool pointwiseStatisticsOnlyComparison(const ComparisonOptions& options) {
    return options.computePointwiseStatistics && !options.computeFrobenius && !options.computeUlp &&
           !options.relativeFrobeniusTolerance && !options.maximumUlpTolerance;
}

inline void coordinatesForLinearIndex(size_t logicalIndex, const Shape& shape,
                                      ComparisonIndexOrder order, std::span<size_t> coordinates) {
    if (coordinates.size() != shape.rank())
        throw std::invalid_argument("Comparison coordinate rank mismatch.");

    if (order == ComparisonIndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            coordinates[dimension] = logicalIndex % shape[dimension];
            logicalIndex /= shape[dimension];
        }
    } else {
        for (size_t dimension = shape.rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            coordinates[index] = logicalIndex % shape[index];
            logicalIndex /= shape[index];
        }
    }
}

inline bool advanceCoordinates(std::span<size_t> coordinates, const Shape& shape,
                               ComparisonIndexOrder order,
                               std::span<const ptrdiff_t> observedStrides,
                               std::span<const ptrdiff_t> expectedStrides,
                               ptrdiff_t& observedOffset, ptrdiff_t& expectedOffset) {
    const auto advanceDimension = [&](size_t dimension) {
        ++coordinates[dimension];
        observedOffset += observedStrides[dimension];
        expectedOffset += expectedStrides[dimension];
        if (coordinates[dimension] < shape[dimension]) return true;
        coordinates[dimension] = 0;
        observedOffset -= static_cast<ptrdiff_t>(shape[dimension]) * observedStrides[dimension];
        expectedOffset -= static_cast<ptrdiff_t>(shape[dimension]) * expectedStrides[dimension];
        return false;
    };

    if (order == ComparisonIndexOrder::FirstDimensionFastest) {
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension)
            if (advanceDimension(dimension)) return true;
    } else {
        for (size_t dimension = shape.rank(); dimension > 0; --dimension)
            if (advanceDimension(dimension - 1)) return true;
    }
    return false;
}

template <typename Function>
void forEachSelectedOffsetPair(const Layout& observedLayout, const Layout& expectedLayout,
                               const ComparisonSelection& selection, Function&& function) {
    if (observedLayout.shape() != expectedLayout.shape())
        throw std::invalid_argument("Comparison offset traversal shape mismatch.");
    if (selection.stride == 0)
        throw std::invalid_argument("Comparison selection stride must be non-zero.");

    const Shape& shape = observedLayout.shape();
    const size_t total = shape.elementCount();
    if (selection.first >= total || selection.maxElements == 0) return;

    if (shape.rank() == 0) {
        function(0, observedLayout.offset(), expectedLayout.offset());
        return;
    }

    if (selection.first == 0 && selection.stride == 1) {
        const bool firstDimensionFastest =
            selection.indexOrder == ComparisonIndexOrder::FirstDimensionFastest;
        const size_t innerDimension = firstDimensionFastest ? 0 : shape.rank() - 1;
        const size_t innerSize = shape[innerDimension];
        const size_t selectedTotal = std::min(total, selection.maxElements);
        const size_t outerCount = (selectedTotal + innerSize - 1) / innerSize;
        std::vector<size_t> coordinates(shape.rank(), 0);

        for (size_t outerIndex = 0; outerIndex < outerCount; ++outerIndex) {
            size_t remaining = outerIndex;
            ptrdiff_t observedBase = observedLayout.offset();
            ptrdiff_t expectedBase = expectedLayout.offset();

            if (firstDimensionFastest) {
                for (size_t dimension = 1; dimension < shape.rank(); ++dimension) {
                    coordinates[dimension] = remaining % shape[dimension];
                    remaining /= shape[dimension];
                    observedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    observedLayout.strides()[dimension];
                    expectedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    expectedLayout.strides()[dimension];
                }
            } else {
                for (size_t dimension = shape.rank() - 1; dimension > 0; --dimension) {
                    const size_t index = dimension - 1;
                    coordinates[index] = remaining % shape[index];
                    remaining /= shape[index];
                    observedBase += static_cast<ptrdiff_t>(coordinates[index]) *
                                    observedLayout.strides()[index];
                    expectedBase += static_cast<ptrdiff_t>(coordinates[index]) *
                                    expectedLayout.strides()[index];
                }
            }

            const size_t logicalBase = outerIndex * innerSize;
            const size_t count = std::min(innerSize, selectedTotal - logicalBase);
            for (size_t innerIndex = 0; innerIndex < count; ++innerIndex) {
                function(logicalBase + innerIndex,
                         observedBase + static_cast<ptrdiff_t>(innerIndex) *
                                            observedLayout.strides()[innerDimension],
                         expectedBase + static_cast<ptrdiff_t>(innerIndex) *
                                            expectedLayout.strides()[innerDimension]);
            }
        }
        return;
    }

    std::vector<size_t> coordinates(shape.rank(), 0);
    coordinatesForLinearIndex(selection.first, shape, selection.indexOrder, coordinates);
    ptrdiff_t observedOffset = observedLayout.elementOffset(coordinates);
    ptrdiff_t expectedOffset = expectedLayout.elementOffset(coordinates);

    if (selection.stride == 1) {
        size_t logicalIndex = selection.first;
        size_t selected = 0;
        while (logicalIndex < total && selected < selection.maxElements) {
            function(logicalIndex, observedOffset, expectedOffset);
            ++selected;
            ++logicalIndex;
            if (logicalIndex >= total || selected >= selection.maxElements) break;
            if (!advanceCoordinates(coordinates, shape, selection.indexOrder,
                                    observedLayout.strides(), expectedLayout.strides(),
                                    observedOffset, expectedOffset))
                break;
        }
        return;
    }

    size_t selected = 0;
    for (size_t logicalIndex = selection.first;
         logicalIndex < total && selected < selection.maxElements; ++selected) {
        coordinatesForLinearIndex(logicalIndex, shape, selection.indexOrder, coordinates);
        function(logicalIndex, observedLayout.elementOffset(coordinates),
                 expectedLayout.elementOffset(coordinates));
        if (selection.stride > std::numeric_limits<size_t>::max() - logicalIndex) break;
        logicalIndex += selection.stride;
    }
}

template <typename Observed, typename Expected>
ComparisonResult comparePointwiseOnlyTyped(const TypedTensorView<Observed>& observed,
                                           const TypedTensorView<Expected>& expected,
                                           const ComparisonOptions& options) {
    const Layout& observedLayout = observed.layout();
    const Layout& expectedLayout = expected.layout();
    const auto observedStorage = observed.storage();
    const auto expectedStorage = expected.storage();
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonResult result;
        if (options.selection.first == 0 && options.selection.stride == 1 &&
            options.selection.indexOrder == ComparisonIndexOrder::FirstDimensionFastest &&
            observedLayout.shape().rank() != 0) {
            const Shape& shape = observedLayout.shape();
            const size_t innerSize = shape[0];
            const size_t selectedTotal =
                std::min(shape.elementCount(), options.selection.maxElements);
            if (selectedTotal == 0) return result;
            const size_t outerCount = (selectedTotal + innerSize - 1) / innerSize;
            std::vector<size_t> coordinates(shape.rank(), 0);

            for (size_t outerIndex = 0; outerIndex < outerCount; ++outerIndex) {
                size_t remaining = outerIndex;
                ptrdiff_t observedBase = observedLayout.offset();
                ptrdiff_t expectedBase = expectedLayout.offset();
                for (size_t dimension = 1; dimension < shape.rank(); ++dimension) {
                    coordinates[dimension] = remaining % shape[dimension];
                    remaining /= shape[dimension];
                    observedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    observedLayout.strides()[dimension];
                    expectedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    expectedLayout.strides()[dimension];
                }

                const size_t logicalBase = outerIndex * innerSize;
                const size_t count = std::min(innerSize, selectedTotal - logicalBase);
                for (size_t innerIndex = 0; innerIndex < count; ++innerIndex) {
                    const ptrdiff_t observedOffset =
                        observedBase +
                        static_cast<ptrdiff_t>(innerIndex) * observedLayout.strides()[0];
                    const ptrdiff_t expectedOffset =
                        expectedBase +
                        static_cast<ptrdiff_t>(innerIndex) * expectedLayout.strides()[0];
                    bool close = false;
                    if constexpr (RuntimeIsComplexV<std::remove_cvref_t<Observed>> ||
                                  RuntimeIsComplexV<std::remove_cvref_t<Expected>>) {
                        const ComparisonValue observedValue =
                            typedComparisonValue(observedStorage[observedOffset]);
                        const ComparisonValue expectedValue =
                            typedComparisonValue(expectedStorage[expectedOffset]);
                        close = predicate(observedValue.real, expectedValue.real);
                        close =
                            close && predicate(observedValue.imaginary, expectedValue.imaginary);
                    } else {
                        close =
                            predicate(fastTypedComparisonValue(observedStorage[observedOffset]),
                                      fastTypedComparisonValue(expectedStorage[expectedOffset]));
                    }
                    result.mismatches += static_cast<size_t>(!close);
                }
            }
            result.compared = selectedTotal;
            result.pointwisePassed = result.mismatches == 0;
            return result;
        }

        forEachSelectedOffsetPair(
            observedLayout, expectedLayout, options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                ++result.compared;
                bool close = false;
                if constexpr (RuntimeIsComplexV<std::remove_cvref_t<Observed>> ||
                              RuntimeIsComplexV<std::remove_cvref_t<Expected>>) {
                    const ComparisonValue observedValue =
                        typedComparisonValue(observedStorage[observedOffset]);
                    const ComparisonValue expectedValue =
                        typedComparisonValue(expectedStorage[expectedOffset]);
                    close = predicate(observedValue.real, expectedValue.real);
                    close = close && predicate(observedValue.imaginary, expectedValue.imaginary);
                } else {
                    close = predicate(fastTypedComparisonValue(observedStorage[observedOffset]),
                                      fastTypedComparisonValue(expectedStorage[expectedOffset]));
                }
                result.mismatches += static_cast<size_t>(!close);
            });
        result.pointwisePassed = result.mismatches == 0;
        return result;
    };

    if (options.equalSignedZero && !options.equalNaNs && options.strictTolerance &&
        options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0) {
        const double tolerance = options.symmetricRelativeTolerance;
        return run([tolerance](auto observedValue, auto expectedValue) {
            if constexpr (std::is_integral_v<decltype(observedValue)> &&
                          std::is_integral_v<decltype(expectedValue)>) {
                return observedValue == expectedValue ||
                       integralDifference(observedValue, expectedValue) <
                           static_cast<long double>(tolerance) *
                               (integralMagnitude(observedValue) +
                                integralMagnitude(expectedValue) + 1.0L);
            } else {
                using Real = decltype(observedValue - expectedValue);
                const Real typedTolerance = static_cast<Real>(tolerance);
                return observedValue == expectedValue ||
                       std::abs(observedValue - expectedValue) <
                           typedTolerance * (std::abs(observedValue) + std::abs(expectedValue) +
                                             static_cast<Real>(1));
            }
        });
    }

    if (options.equalSignedZero && !options.equalNaNs && !options.strictTolerance &&
        options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0 &&
        options.symmetricRelativeTolerance == 0.0)
        return run(
            [](auto observedValue, auto expectedValue) { return observedValue == expectedValue; });

    return run([&options](auto observedValue, auto expectedValue) {
        return valuesCloseFast(observedValue, expectedValue, options);
    });
}

struct ComparisonPair {
    // Evidence is normalized for statistics and reporting. pointwiseClose is
    // evaluated in the caller's semantic precision so normalization cannot
    // change pass/fail (for example, adjacent uint64_t values above 2^53).
    ComparisonValue observed;
    ComparisonValue expected;
    bool pointwiseClose = true;
};

using ComparisonPairChunkLoader = void (*)(const void*, ptrdiff_t, ptrdiff_t, const void*,
                                           ptrdiff_t, ptrdiff_t, size_t, const ComparisonOptions&,
                                           ComparisonPair*);

// Synchronous ABI bridge used by the public templates below. The compiled
// engine does not retain storage pointers or the loader after the call returns.
ComparisonResult compareWithPairLoader(const Layout& observedLayout, const void* observedStorage,
                                       const Layout& expectedLayout, const void* expectedStorage,
                                       ComparisonPairChunkLoader pairLoader,
                                       const ComparisonOptions& options);

template <typename Observed, typename Expected>
bool pointwiseValuesClose(Observed observed, Expected expected, const ComparisonOptions& options) {
    if (options.equalSignedZero && !options.equalNaNs && options.strictTolerance &&
        options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0) {
        if constexpr (std::is_integral_v<Observed> && std::is_integral_v<Expected>) {
            return observed == expected ||
                   integralDifference(observed, expected) <
                       static_cast<long double>(options.symmetricRelativeTolerance) *
                           (integralMagnitude(observed) + integralMagnitude(expected) + 1.0L);
        } else {
            using Real = decltype(observed - expected);
            const Real typedTolerance = static_cast<Real>(options.symmetricRelativeTolerance);
            return observed == expected ||
                   std::abs(observed - expected) <
                       typedTolerance *
                           (std::abs(observed) + std::abs(expected) + static_cast<Real>(1));
        }
    }

    if (options.equalSignedZero && !options.equalNaNs && !options.strictTolerance &&
        options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0 &&
        options.symmetricRelativeTolerance == 0.0)
        return observed == expected;

    return valuesCloseFast(observed, expected, options);
}

template <typename Observed, typename Expected>
void loadTypedComparisonChunk(const void* observedStorage, ptrdiff_t observedOffset,
                              ptrdiff_t observedStride, const void* expectedStorage,
                              ptrdiff_t expectedOffset, ptrdiff_t expectedStride, size_t count,
                              const ComparisonOptions& options, ComparisonPair* pairs) {
    const auto* observedValues = static_cast<const Observed*>(observedStorage);
    const auto* expectedValues = static_cast<const Expected*>(expectedStorage);
    for (size_t index = 0; index < count; ++index) {
        const auto& observed =
            observedValues[observedOffset + static_cast<ptrdiff_t>(index) * observedStride];
        const auto& expected =
            expectedValues[expectedOffset + static_cast<ptrdiff_t>(index) * expectedStride];
        if constexpr (RuntimeIsComplexV<std::remove_cvref_t<Observed>> ||
                      RuntimeIsComplexV<std::remove_cvref_t<Expected>>) {
            ComparisonPair& pair = pairs[index];
            pair.observed = typedComparisonValue(observed);
            pair.expected = typedComparisonValue(expected);
            pair.pointwiseClose =
                !options.pointwise ||
                (pointwiseValuesClose(pair.observed.real, pair.expected.real, options) &&
                 pointwiseValuesClose(pair.observed.imaginary, pair.expected.imaginary, options));
        } else {
            const auto observedValue = fastTypedComparisonValue(observed);
            const auto expectedValue = fastTypedComparisonValue(expected);
            pairs[index] = {
                typedComparisonValue(observedValue),
                typedComparisonValue(expectedValue),
                !options.pointwise || pointwiseValuesClose(observedValue, expectedValue, options),
            };
        }
    }
}

template <typename Observed, typename Expected>
ComparisonResult comparePointwiseStatisticsTyped(const TypedTensorView<Observed>& observed,
                                                 const TypedTensorView<Expected>& expected,
                                                 const ComparisonOptions& options) {
    ComparisonResult result;
    result.reportedMismatches.reserve(options.maxReportedMismatches);
    if (options.reportMatchingElements)
        result.reportedComparisons.reserve(options.maxReportedMismatches);

    const auto observedStorage = observed.storage();
    const auto expectedStorage = expected.storage();
    forEachSelectedOffsetPair(
        observed.layout(), expected.layout(), options.selection,
        [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
            const auto& observedElement = observedStorage[observedOffset];
            const auto& expectedElement = expectedStorage[expectedOffset];
            ComparisonPair pair;
            if constexpr (RuntimeIsComplexV<std::remove_cvref_t<Observed>> ||
                          RuntimeIsComplexV<std::remove_cvref_t<Expected>>) {
                pair.observed = typedComparisonValue(observedElement);
                pair.expected = typedComparisonValue(expectedElement);
                if (options.pointwise)
                    pair.pointwiseClose =
                        pointwiseValuesClose(pair.observed.real, pair.expected.real, options) &&
                        pointwiseValuesClose(pair.observed.imaginary, pair.expected.imaginary,
                                             options);
            } else {
                const auto observedValue = fastTypedComparisonValue(observedElement);
                const auto expectedValue = fastTypedComparisonValue(expectedElement);
                pair.observed = typedComparisonValue(observedValue);
                pair.expected = typedComparisonValue(expectedValue);
                if (options.pointwise)
                    pair.pointwiseClose =
                        pointwiseValuesClose(observedValue, expectedValue, options);
            }
            ++result.compared;

            const bool complexValue = pair.observed.complex || pair.expected.complex;
            const bool exactReal = pair.observed.real == pair.expected.real;
            const bool exactImaginary =
                !complexValue || pair.observed.imaginary == pair.expected.imaginary;
            const bool signedZeroMatches =
                options.equalSignedZero ||
                (!oppositeZeroSigns(pair.observed.real, pair.expected.real) &&
                 (!complexValue ||
                  !oppositeZeroSigns(pair.observed.imaginary, pair.expected.imaginary)));
            if (pair.pointwiseClose && exactReal && exactImaginary && signedZeroMatches &&
                !options.reportMatchingElements) {
                result.matchedInfinities += static_cast<size_t>(std::isinf(pair.observed.real));
                if (complexValue)
                    result.matchedInfinities +=
                        static_cast<size_t>(std::isinf(pair.observed.imaginary));
                return;
            }

            const ComponentResult real =
                compareComponent(pair.observed.real, pair.expected.real, options);
            const ComponentResult imaginary =
                complexValue
                    ? compareComponent(pair.observed.imaginary, pair.expected.imaginary, options)
                    : ComponentResult{.close = true};
            const bool close =
                options.pointwise ? pair.pointwiseClose : real.close && imaginary.close;
            const bool nonFiniteMismatch = real.nonFiniteMismatch || imaginary.nonFiniteMismatch;

            result.matchedNaNs +=
                static_cast<size_t>(real.matchedNaN) + static_cast<size_t>(imaginary.matchedNaN);
            result.matchedInfinities += static_cast<size_t>(real.matchedInfinity) +
                                        static_cast<size_t>(imaginary.matchedInfinity);
            result.nonFiniteMismatches += nonFiniteMismatch;
            result.signedZeroMismatches += real.signedZeroMismatch || imaginary.signedZeroMismatch;

            double difference =
                complexValue ? std::hypot(real.difference, imaginary.difference) : real.difference;
            if (nonFiniteMismatch) difference = std::numeric_limits<double>::infinity();
            result.maxAbsoluteDifference = std::max(result.maxAbsoluteDifference, difference);
            result.maxRelativeDifference =
                std::max({result.maxRelativeDifference, real.relativeDifference,
                          imaginary.relativeDifference});
            result.maxSymmetricRelativeDifference =
                std::max({result.maxSymmetricRelativeDifference, real.symmetricRelativeDifference,
                          imaginary.symmetricRelativeDifference});

            const bool reportComparison =
                options.reportMatchingElements &&
                result.reportedComparisons.size() < options.maxReportedMismatches;
            const bool reportMismatch =
                options.pointwise && !close &&
                result.reportedMismatches.size() < options.maxReportedMismatches;
            if (reportComparison || reportMismatch) {
                std::vector<size_t> coordinates(observed.shape().rank(), 0);
                coordinatesForLinearIndex(logicalIndex, observed.shape(),
                                          options.selection.indexOrder, coordinates);
                const Mismatch sample{
                    logicalIndex,
                    std::move(coordinates),
                    observedOffset,
                    expectedOffset,
                    pair.observed.real,
                    pair.expected.real,
                    pair.observed.imaginary,
                    pair.expected.imaginary,
                    difference,
                    std::max(real.tolerance, imaginary.tolerance),
                    close,
                };
                if (reportComparison) result.reportedComparisons.push_back(sample);
                if (reportMismatch) result.reportedMismatches.push_back(sample);
            }

            if (options.pointwise && !close) ++result.mismatches;
        });
    result.pointwisePassed = !options.pointwise || result.mismatches == 0;
    return result;
}
}  // namespace detail

template <typename Observed, typename Expected>
ComparisonResult compare(const TypedTensorView<Observed>& observed,
                         const TypedTensorView<Expected>& expected,
                         const ComparisonOptions& options = {}) {
    if (observed.shape() != expected.shape())
        throw std::invalid_argument("Host validation comparison shape mismatch.");

    if (detail::pointwiseOnlyComparison(options)) {
        ComparisonResult result = detail::comparePointwiseOnlyTyped(observed, expected, options);
        const bool needsSamples = options.maxReportedMismatches != 0 &&
                                  (options.reportMatchingElements || result.mismatches != 0);
        if (!needsSamples) return result;

        ComparisonOptions detailed = options;
        detailed.computePointwiseStatistics = true;
        return detail::comparePointwiseStatisticsTyped(observed, expected, detailed);
    }

    if (detail::pointwiseStatisticsOnlyComparison(options))
        return detail::comparePointwiseStatisticsTyped(observed, expected, options);

    return detail::compareWithPairLoader(
        observed.layout(), observed.storage().data(), expected.layout(), expected.storage().data(),
        &detail::loadTypedComparisonChunk<Observed, Expected>, options);
}
}  // namespace roc::host_validation
