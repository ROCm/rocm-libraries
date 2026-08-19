// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Internal comparison implementation shared by the compiled runtime engine.

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
inline void validateComparisonOptions(const ComparisonOptions& options) {
    if (options.selection.stride == 0)
        throw std::invalid_argument("Comparison selection stride must be non-zero.");
    if (options.relativeFrobeniusTolerance && !options.computeFrobenius)
        throw std::invalid_argument("A relative Frobenius tolerance requires Frobenius evidence.");
    if (options.maximumUlpTolerance && !options.computeUlp)
        throw std::invalid_argument("A maximum ULP tolerance requires ULP evidence.");
    if (options.computeUlp && !options.ulpType)
        throw std::invalid_argument("ULP evidence requires an explicit scalar type.");
    if (options.ulpType && !isConcreteScalarType(*options.ulpType))
        throw std::invalid_argument("ULP scalar type must be a concrete scalar type.");
}

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

inline ComponentResult compareComplexMagnitude(const ComparisonValue& observed,
                                               const ComparisonValue& expected,
                                               const ComparisonOptions& options) {
    ComponentResult result;
    const bool observedNaN = std::isnan(observed.real) || std::isnan(observed.imaginary);
    const bool expectedNaN = std::isnan(expected.real) || std::isnan(expected.imaginary);
    if (observedNaN || expectedNaN) {
        result.matchedNaN = options.equalNaNs && observedNaN && expectedNaN;
        result.close = result.matchedNaN;
        result.nonFiniteMismatch = !result.close;
        result.difference = result.close ? 0.0 : std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        result.symmetricRelativeDifference = result.difference;
        return result;
    }

    const bool signedZeroMismatch =
        !options.equalSignedZero && (oppositeZeroSigns(observed.real, expected.real) ||
                                     oppositeZeroSigns(observed.imaginary, expected.imaginary));
    if (signedZeroMismatch) {
        result.signedZeroMismatch = true;
        return result;
    }

    const bool exactlyEqual =
        observed.real == expected.real && observed.imaginary == expected.imaginary;
    if (exactlyEqual) {
        result.close = true;
        result.matchedInfinity = std::isinf(observed.real) || std::isinf(observed.imaginary);
        return result;
    }

    const bool nonFinite = std::isinf(observed.real) || std::isinf(observed.imaginary) ||
                           std::isinf(expected.real) || std::isinf(expected.imaginary);
    if (nonFinite) {
        result.nonFiniteMismatch = true;
        result.difference = std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        result.symmetricRelativeDifference = result.difference;
        return result;
    }

    const double observedMagnitude = std::hypot(observed.real, observed.imaginary);
    const double expectedMagnitude = std::hypot(expected.real, expected.imaginary);
    result.difference =
        std::hypot(observed.real - expected.real, observed.imaginary - expected.imaginary);
    result.tolerance =
        options.absoluteTolerance + options.relativeTolerance * expectedMagnitude +
        options.symmetricRelativeTolerance * (observedMagnitude + expectedMagnitude + 1.0);
    result.relativeDifference =
        expectedMagnitude == 0.0
            ? (result.difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
            : result.difference / expectedMagnitude;
    result.symmetricRelativeDifference =
        result.difference / (observedMagnitude + expectedMagnitude + 1.0);
    result.close = options.strictTolerance ? result.difference < result.tolerance
                                           : result.difference <= result.tolerance;
    return result;
}

inline bool pointwiseStatisticsOnlyComparison(const ComparisonOptions& options) {
    return options.computePointwiseStatistics && !options.computeFrobenius && !options.computeUlp &&
           !options.relativeFrobeniusTolerance && !options.maximumUlpTolerance;
}

inline void coordinatesForLinearIndex(size_t logicalIndex, const Shape& shape, IndexOrder order,
                                      std::span<size_t> coordinates) {
    shape.coordinates(logicalIndex, order, coordinates);
}

inline bool advanceCoordinates(std::span<size_t> coordinates, const Shape& shape, IndexOrder order,
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

    if (order == IndexOrder::FirstDimensionFastest) {
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
            selection.indexOrder == IndexOrder::FirstDimensionFastest;
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

}  // namespace detail
}  // namespace roc::host_validation
