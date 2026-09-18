// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>
#include <optional>
#include <type_traits>

#include "comparison_common.hpp"
#include "comparison_metrics.hpp"

namespace roc::host_numerics::detail {
class ComparisonAccumulator {
   public:
    ComparisonAccumulator(const ComparisonOptions& options, const Shape& shape);

    void observeReal(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                     double observed, double expected,
                     std::optional<bool> allCloseDecision = std::nullopt,
                     const ExactRealEvidence* exactEvidence = nullptr);

    template <typename Observed, typename Expected>
    void observeIntegral(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                         Observed observed, Expected expected) {
        static_assert(std::is_integral_v<Observed> && std::is_integral_v<Expected>);
        const long double observedMagnitude = integralMagnitude(observed);
        const long double expectedMagnitude = integralMagnitude(expected);
        const long double difference = integralDifference(observed, expected);
        const long double tolerance =
            static_cast<long double>(m_options.absoluteTolerance) +
            static_cast<long double>(m_options.relativeTolerance) * expectedMagnitude;

        ComponentResult component;
        component.difference = static_cast<double>(difference);
        component.tolerance = static_cast<double>(tolerance);
        component.relativeDifference =
            expectedMagnitude == 0.0L
                ? (difference == 0.0L ? 0.0 : std::numeric_limits<double>::infinity())
                : static_cast<double>(difference / expectedMagnitude);
        component.close = integralValuesEqual(observed, expected) || difference <= tolerance;
        const ExactRealEvidence exactEvidence{
            component,
            observedMagnitude,
            expectedMagnitude,
            difference,
        };
        observeReal(logicalIndex, observedOffset, expectedOffset, static_cast<double>(observed),
                    static_cast<double>(expected),
                    m_options.allClose ? std::optional<bool>(component.close) : std::nullopt,
                    &exactEvidence);
    }

    void observe(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                 const ComparisonValue& observed, const ComparisonValue& expected,
                 std::optional<bool> allCloseDecision = std::nullopt,
                 const ExactRealEvidence* exactEvidence = nullptr);

    void merge(ComparisonAccumulator&& other);

    ComparisonReport finish();

   private:
    void accumulateUlpDistance(double distance);
    void accumulateUlp(double exact, double approximation);

    ComparisonOptions m_options;
    const Shape* m_shape = nullptr;
    ComparisonReport m_result;
    long double m_differenceSquares = 0.0;
    long double m_observedSquares = 0.0;
    long double m_expectedSquares = 0.0;
    bool m_sawNonFinite = false;
};
}  // namespace roc::host_numerics::detail
