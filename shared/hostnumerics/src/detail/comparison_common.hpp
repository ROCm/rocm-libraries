// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Internal comparison implementation shared by the compiled runtime engine.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <roc/hostnumerics/comparison.hpp>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "layout_iteration.hpp"

namespace roc::hostnumerics {
namespace detail {
struct ComparisonInvocation : ComparisonOptions {
    template <typename Indices>
    ComparisonInvocation(const ComparisonOptions& options, const Shape& shape,
                         const Indices& indices, std::span<const size_t> order)
        : ComparisonOptions(options) {
        bindIndices(*this, shape, indices, order);
    }
    size_t indicesCount = 0;
    std::vector<size_t> indicesOrder;
    std::function<size_t(size_t)> indexAt;
    std::optional<size_t> firstContiguousIndex;
};

void validateComparisonOptions(const ComparisonOptions& options);
bool allCloseOnlyComparison(const ComparisonOptions& options);

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
uintmax_t unsignedIntegralMagnitude(T value) {
    static_assert(std::is_integral_v<T>);
    if constexpr (std::is_same_v<T, bool>) {
        return value ? 1 : 0;
    } else if constexpr (std::is_unsigned_v<T>) {
        return static_cast<uintmax_t>(value);
    } else {
        using Unsigned = std::make_unsigned_t<T>;
        const Unsigned magnitude =
            value < 0 ? static_cast<Unsigned>(-(value + 1)) + 1 : static_cast<Unsigned>(value);
        return static_cast<uintmax_t>(magnitude);
    }
}

template <typename T>
long double integralMagnitude(T value) {
    return static_cast<long double>(unsignedIntegralMagnitude(value));
}

template <typename Observed, typename Expected>
long double integralDifference(Observed observed, Expected expected) {
    static_assert(std::is_integral_v<Observed> && std::is_integral_v<Expected>);
    const bool observedNegative = [&] {
        if constexpr (std::is_signed_v<Observed>)
            return observed < 0;
        else
            return false;
    }();
    const bool expectedNegative = [&] {
        if constexpr (std::is_signed_v<Expected>)
            return expected < 0;
        else
            return false;
    }();
    const uintmax_t observedMagnitude = unsignedIntegralMagnitude(observed);
    const uintmax_t expectedMagnitude = unsignedIntegralMagnitude(expected);

    if (observedNegative == expectedNegative) {
        const uintmax_t difference = observedMagnitude >= expectedMagnitude
                                         ? observedMagnitude - expectedMagnitude
                                         : expectedMagnitude - observedMagnitude;
        return static_cast<long double>(difference);
    }

    if (expectedMagnitude <= std::numeric_limits<uintmax_t>::max() - observedMagnitude)
        return static_cast<long double>(observedMagnitude + expectedMagnitude);
    return static_cast<long double>(observedMagnitude) +
           static_cast<long double>(expectedMagnitude);
}

template <typename Observed, typename Expected>
bool valuesCloseFast(Observed observed, Expected expected, const ComparisonOptions& options) {
    if (observed == expected) return true;
    if constexpr (std::is_integral_v<Observed> && std::is_integral_v<Expected>) {
        const long double expectedMagnitude = integralMagnitude(expected);
        const long double difference = integralDifference(observed, expected);
        const long double tolerance =
            options.absoluteTolerance + options.relativeTolerance * expectedMagnitude;
        return difference <= tolerance;
    } else {
        if (std::isnan(observed) || std::isnan(expected))
            return options.equalNaNs && std::isnan(observed) && std::isnan(expected);
        if (std::isinf(observed) || std::isinf(expected)) return false;

        const double difference =
            std::abs(static_cast<double>(observed) - static_cast<double>(expected));
        const double tolerance =
            options.absoluteTolerance + options.relativeTolerance * std::abs(expected);
        return difference <= tolerance;
    }
}

struct ComponentResult {
    bool close = false;
    bool matchedNaN = false;
    bool matchedInfinity = false;
    bool nonFiniteMismatch = false;
    double difference = 0.0;
    double tolerance = 0.0;
    double relativeDifference = 0.0;
};

ComponentResult compareComponent(double observed, double expected,
                                 const ComparisonOptions& options);
ComponentResult compareComplexMagnitude(const ComparisonValue& observed,
                                        const ComparisonValue& expected,
                                        const ComparisonOptions& options);
bool elementwiseStatisticsOnlyComparison(const ComparisonOptions& options);

template <typename Function>
void forEachSelectedOffsetPair(const Layout& observedLayout, const Layout& expectedLayout,
                               const ComparisonInvocation& invocation, Function&& function) {
    if (observedLayout.shape() != expectedLayout.shape())
        throw std::runtime_error("Comparison offset traversal shape mismatch.");
    const Shape& shape = observedLayout.shape();
    const size_t count = invocation.indicesCount;
    if (count == 0) return;
    if (invocation.firstContiguousIndex) {
        const size_t first = *invocation.firstContiguousIndex;
        forEachLayoutOffsetPairRange(observedLayout, expectedLayout, invocation.indicesOrder, first,
                                     first + count, std::forward<Function>(function));
        return;
    }
    forEachBoundCoordinate(shape, invocation,
                           [&](size_t index, std::span<const size_t> coordinates) {
                               function(index, preparedElementOffset(observedLayout, coordinates),
                                        preparedElementOffset(expectedLayout, coordinates));
                           });
}

template <typename Observed, typename Expected>
bool valuesClose(Observed observed, Expected expected, const ComparisonOptions& options) {
    if (!options.equalNaNs && options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0)
        return observed == expected;

    return valuesCloseFast(observed, expected, options);
}

}  // namespace detail
}  // namespace roc::hostnumerics
