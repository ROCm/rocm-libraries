// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/comparison_common.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace roc::host_numerics::detail {
void validateComparisonOptions(const ComparisonOptions& options) {
    if (options.relativeFrobeniusTolerance && !options.computeFrobenius)
        throw std::invalid_argument("A relative Frobenius tolerance requires Frobenius evidence.");
    if (options.maximumUlpTolerance && !options.computeUlp)
        throw std::invalid_argument("A maximum ULP tolerance requires ULP evidence.");
    if (options.computeUlp && !options.ulpType)
        throw std::invalid_argument("ULP evidence requires an explicit scalar type.");
    if (options.ulpType && !isConcreteScalarType(*options.ulpType))
        throw std::invalid_argument("ULP scalar type must be a concrete scalar type.");
}

bool allCloseOnlyComparison(const ComparisonOptions& options) {
    return options.allClose && !options.computeElementwiseStatistics && !options.computeFrobenius &&
           !options.computeUlp && !options.relativeFrobeniusTolerance &&
           !options.maximumUlpTolerance;
}

ComponentResult compareComponent(double observed, double expected,
                                 const ComparisonOptions& options) {
    ComponentResult result;

    if (std::isnan(observed) || std::isnan(expected)) {
        result.matchedNaN = options.equalNaNs && std::isnan(observed) && std::isnan(expected);
        result.close = result.matchedNaN;
        result.nonFiniteMismatch = !result.close;
        result.difference = result.close ? 0.0 : std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        return result;
    }

    if (std::isinf(observed) || std::isinf(expected)) {
        result.matchedInfinity = observed == expected;
        result.close = result.matchedInfinity;
        result.nonFiniteMismatch = !result.close;
        result.difference = result.close ? 0.0 : std::numeric_limits<double>::infinity();
        result.relativeDifference = result.difference;
        return result;
    }

    result.difference = std::abs(observed - expected);
    result.tolerance = options.absoluteTolerance + options.relativeTolerance * std::abs(expected);
    result.relativeDifference =
        expected == 0.0 ? (result.difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
                        : result.difference / std::abs(expected);
    result.close = result.difference <= result.tolerance;
    return result;
}

ComponentResult compareComplexMagnitude(const ComparisonValue& observed,
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
        return result;
    }

    const double expectedMagnitude = std::hypot(expected.real, expected.imaginary);
    result.difference =
        std::hypot(observed.real - expected.real, observed.imaginary - expected.imaginary);
    result.tolerance = options.absoluteTolerance + options.relativeTolerance * expectedMagnitude;
    result.relativeDifference =
        expectedMagnitude == 0.0
            ? (result.difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity())
            : result.difference / expectedMagnitude;
    result.close = result.difference <= result.tolerance;
    return result;
}

bool elementwiseStatisticsOnlyComparison(const ComparisonOptions& options) {
    return options.computeElementwiseStatistics && !options.computeFrobenius &&
           !options.computeUlp && !options.relativeFrobeniusTolerance &&
           !options.maximumUlpTolerance;
}

void coordinatesForLinearIndex(size_t logicalIndex, const Shape& shape, IndexOrder order,
                               std::span<size_t> coordinates) {
    shape.coordinates(logicalIndex, order, coordinates);
}

bool advanceCoordinates(std::span<size_t> coordinates, const Shape& shape, IndexOrder order,
                        std::span<const ptrdiff_t> observedStrides,
                        std::span<const ptrdiff_t> expectedStrides, ptrdiff_t& observedOffset,
                        ptrdiff_t& expectedOffset) {
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
}  // namespace roc::host_numerics::detail
