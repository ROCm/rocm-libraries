// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "detail/comparison_accumulator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace roc::host_numerics::detail {
ComparisonAccumulator::ComparisonAccumulator(const ComparisonOptions& options, const Shape& shape)
    : m_options(options), m_shape(&shape) {
    m_result.allCloseEvaluated = m_options.allClose;
    m_result.frobeniusEvaluated = m_options.relativeFrobeniusTolerance.has_value();
    m_result.ulpEvaluated = m_options.maximumUlpTolerance.has_value();
    m_result.reportedMismatches.reserve(m_options.maxReportedMismatches);
    if (m_options.reportMatchingElements)
        m_result.reportedComparisons.reserve(m_options.maxReportedMismatches);
}

void ComparisonAccumulator::observeReal(size_t logicalIndex, ptrdiff_t observedOffset,
                                        ptrdiff_t expectedOffset, double observed, double expected,
                                        std::optional<bool> allCloseDecision,
                                        const ExactRealEvidence* exactEvidence) {
    ++m_result.compared;
    // ULP-only validation does not need the general elementwise or norm evidence. Most entries in
    // a passing numerical result are exactly equal, so count those directly and calculate a ULP
    // distance only for the exceptional values.
    if (!m_options.allClose && !m_options.computeElementwiseStatistics &&
        !m_options.computeFrobenius && m_options.computeUlp && !m_options.reportMatchingElements) {
        const ScalarCategory ulpCategory = scalarTypeInfo(*m_options.ulpType).category;
        if (exactEvidence != nullptr && (ulpCategory == ScalarCategory::Boolean ||
                                         ulpCategory == ScalarCategory::SignedInteger ||
                                         ulpCategory == ScalarCategory::UnsignedInteger)) {
            accumulateUlpDistance(static_cast<double>(exactEvidence->difference));
        } else if (observed == expected ||
                   (m_options.equalNaNs && std::isnan(observed) && std::isnan(expected))) {
            ++m_result.ulpCompared;
        } else {
            accumulateUlp(expected, observed);
        }
        return;
    }
    m_sawNonFinite = m_sawNonFinite || !std::isfinite(observed) || !std::isfinite(expected);
    if ((!allCloseDecision || *allCloseDecision) && observed == expected &&
        !m_options.computeFrobenius && !m_options.computeUlp && !m_options.reportMatchingElements) {
        if (m_options.computeElementwiseStatistics)
            m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed));
        return;
    }

    --m_result.compared;
    observe(logicalIndex, observedOffset, expectedOffset, ComparisonValue{observed, 0.0, false},
            ComparisonValue{expected, 0.0, false}, allCloseDecision, exactEvidence);
}

void ComparisonAccumulator::observe(size_t logicalIndex, ptrdiff_t observedOffset,
                                    ptrdiff_t expectedOffset, const ComparisonValue& observed,
                                    const ComparisonValue& expected,
                                    std::optional<bool> allCloseDecision,
                                    const ExactRealEvidence* exactEvidence) {
    ++m_result.compared;
    m_sawNonFinite = m_sawNonFinite || !std::isfinite(observed.real) ||
                     !std::isfinite(expected.real) ||
                     (observed.complex && !std::isfinite(observed.imaginary)) ||
                     (expected.complex && !std::isfinite(expected.imaginary));

    const bool complexValue = observed.complex || expected.complex;
    const bool exactReal = observed.real == expected.real;
    const bool exactImaginary = !complexValue || observed.imaginary == expected.imaginary;
    if ((!allCloseDecision || *allCloseDecision) && exactReal && exactImaginary &&
        !m_options.computeFrobenius && !m_options.computeUlp && !m_options.reportMatchingElements) {
        if (m_options.computeElementwiseStatistics) {
            if (complexValue &&
                m_options.complexComparisonMode == ComplexComparisonMode::Magnitude) {
                m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed.real) ||
                                                                  std::isinf(observed.imaginary));
            } else {
                m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed.real));
                if (complexValue)
                    m_result.matchedInfinities +=
                        static_cast<size_t>(std::isinf(observed.imaginary));
            }
        }
        return;
    }

    const ComponentResult real = exactEvidence != nullptr
                                     ? exactEvidence->component
                                     : compareComponent(observed.real, expected.real, m_options);
    const ComponentResult imaginary =
        complexValue ? compareComponent(observed.imaginary, expected.imaginary, m_options)
                     : ComponentResult{.close = true};
    const bool magnitudeMode =
        complexValue && m_options.complexComparisonMode == ComplexComparisonMode::Magnitude;
    const ComponentResult magnitude = magnitudeMode
                                          ? compareComplexMagnitude(observed, expected, m_options)
                                          : ComponentResult{.close = true};

    const bool close =
        allCloseDecision.value_or(magnitudeMode ? magnitude.close : real.close && imaginary.close);
    const bool nonFiniteMismatch = magnitudeMode
                                       ? magnitude.nonFiniteMismatch
                                       : real.nonFiniteMismatch || imaginary.nonFiniteMismatch;

    if (m_options.computeElementwiseStatistics) {
        if (magnitudeMode) {
            m_result.matchedNaNs += static_cast<size_t>(magnitude.matchedNaN);
            m_result.matchedInfinities += static_cast<size_t>(magnitude.matchedInfinity);
        } else {
            m_result.matchedNaNs +=
                static_cast<size_t>(real.matchedNaN) + static_cast<size_t>(imaginary.matchedNaN);
            m_result.matchedInfinities += static_cast<size_t>(real.matchedInfinity) +
                                          static_cast<size_t>(imaginary.matchedInfinity);
        }
        m_result.nonFiniteMismatches += nonFiniteMismatch;
    }

    double difference =
        magnitudeMode
            ? magnitude.difference
            : (complexValue ? std::hypot(real.difference, imaginary.difference) : real.difference);
    if (nonFiniteMismatch) difference = std::numeric_limits<double>::infinity();

    if (m_options.computeElementwiseStatistics || m_options.computeFrobenius)
        m_result.maxAbsoluteDifference = std::max(m_result.maxAbsoluteDifference, difference);

    if (m_options.computeElementwiseStatistics) {
        if (magnitudeMode) {
            m_result.maxRelativeDifference =
                std::max(m_result.maxRelativeDifference, magnitude.relativeDifference);
        } else {
            m_result.maxRelativeDifference =
                std::max({m_result.maxRelativeDifference, real.relativeDifference,
                          imaginary.relativeDifference});
        }
    }

    if (m_options.computeFrobenius) {
        if (!nonFiniteMismatch && std::isfinite(observed.real) && std::isfinite(expected.real) &&
            (!complexValue ||
             (std::isfinite(observed.imaginary) && std::isfinite(expected.imaginary)))) {
            const long double observedMagnitude = exactEvidence != nullptr
                                                      ? exactEvidence->observedMagnitude
                                                      : valueMagnitude(observed);
            const long double expectedMagnitude = exactEvidence != nullptr
                                                      ? exactEvidence->expectedMagnitude
                                                      : valueMagnitude(expected);
            m_result.maximumObservedMagnitude =
                std::max(m_result.maximumObservedMagnitude, static_cast<double>(observedMagnitude));
            m_result.maximumExpectedMagnitude =
                std::max(m_result.maximumExpectedMagnitude, static_cast<double>(expectedMagnitude));
            m_observedSquares += observedMagnitude * observedMagnitude;
            m_expectedSquares += expectedMagnitude * expectedMagnitude;
            const long double differenceMagnitude =
                exactEvidence != nullptr ? exactEvidence->difference : difference;
            m_differenceSquares += differenceMagnitude * differenceMagnitude;
        } else if (nonFiniteMismatch) {
            m_differenceSquares = std::numeric_limits<long double>::infinity();
        }
    }

    if (m_options.computeUlp) {
        if (real.matchedNaN || real.matchedInfinity)
            ++m_result.ulpCompared;
        else if (exactEvidence != nullptr &&
                 (scalarTypeInfo(*m_options.ulpType).category == ScalarCategory::Boolean ||
                  scalarTypeInfo(*m_options.ulpType).category == ScalarCategory::SignedInteger ||
                  scalarTypeInfo(*m_options.ulpType).category == ScalarCategory::UnsignedInteger))
            accumulateUlpDistance(static_cast<double>(exactEvidence->difference));
        else
            accumulateUlp(expected.real, observed.real);
        if (complexValue) {
            if (imaginary.matchedNaN || imaginary.matchedInfinity)
                ++m_result.ulpCompared;
            else
                accumulateUlp(expected.imaginary, observed.imaginary);
        }
    }

    const bool reportComparison =
        m_options.reportMatchingElements &&
        m_result.reportedComparisons.size() < m_options.maxReportedMismatches;
    const bool reportMismatch =
        m_options.allClose && !close &&
        m_result.reportedMismatches.size() < m_options.maxReportedMismatches;
    if (reportComparison || reportMismatch) {
        std::vector<size_t> coordinates(m_shape->rank(), 0);
        coordinatesForLinearIndex(logicalIndex, *m_shape, m_options.selection.indexOrder(),
                                  coordinates);
        const Mismatch sample{
            logicalIndex,
            std::move(coordinates),
            observedOffset,
            expectedOffset,
            observed.real,
            expected.real,
            observed.imaginary,
            expected.imaginary,
            difference,
            magnitudeMode ? magnitude.tolerance : std::max(real.tolerance, imaginary.tolerance),
            close,
        };
        if (reportComparison) m_result.reportedComparisons.push_back(sample);
        if (reportMismatch) m_result.reportedMismatches.push_back(sample);
    }

    if (m_options.allClose && !close) ++m_result.mismatches;
}

void ComparisonAccumulator::merge(ComparisonAccumulator&& other) {
    m_result.compared += other.m_result.compared;
    m_result.mismatches += other.m_result.mismatches;
    m_result.matchedNaNs += other.m_result.matchedNaNs;
    m_result.matchedInfinities += other.m_result.matchedInfinities;
    m_result.nonFiniteMismatches += other.m_result.nonFiniteMismatches;
    m_result.maxAbsoluteDifference =
        std::max(m_result.maxAbsoluteDifference, other.m_result.maxAbsoluteDifference);
    m_result.maxRelativeDifference =
        std::max(m_result.maxRelativeDifference, other.m_result.maxRelativeDifference);
    m_result.maximumObservedMagnitude =
        std::max(m_result.maximumObservedMagnitude, other.m_result.maximumObservedMagnitude);
    m_result.maximumExpectedMagnitude =
        std::max(m_result.maximumExpectedMagnitude, other.m_result.maximumExpectedMagnitude);
    m_result.maximumUlp = std::max(m_result.maximumUlp, other.m_result.maximumUlp);
    m_result.sumUlp += other.m_result.sumUlp;
    m_result.ulpCompared += other.m_result.ulpCompared;
    m_differenceSquares += other.m_differenceSquares;
    m_observedSquares += other.m_observedSquares;
    m_expectedSquares += other.m_expectedSquares;
    m_sawNonFinite = m_sawNonFinite || other.m_sawNonFinite;

    const auto appendSamples = [&](std::vector<Mismatch>& destination,
                                   std::vector<Mismatch>& source) {
        for (Mismatch& sample : source) {
            if (destination.size() >= m_options.maxReportedMismatches) break;
            destination.push_back(std::move(sample));
        }
    };
    appendSamples(m_result.reportedMismatches, other.m_result.reportedMismatches);
    appendSamples(m_result.reportedComparisons, other.m_result.reportedComparisons);
}

ComparisonReport ComparisonAccumulator::finish() {
    m_result.allClosePassed = !m_options.allClose || m_result.mismatches == 0;
    if (m_options.computeFrobenius) {
        m_result.frobeniusDifference = std::sqrt(static_cast<double>(m_differenceSquares));
        m_result.frobeniusObserved = std::sqrt(static_cast<double>(m_observedSquares));
        m_result.frobeniusExpected = std::sqrt(static_cast<double>(m_expectedSquares));
        if (m_options.nonFiniteValuesInvalidateRelativeNorms && m_sawNonFinite) {
            m_result.relativeFrobeniusError = std::numeric_limits<double>::quiet_NaN();
            m_result.relativeMaximumError = std::numeric_limits<double>::quiet_NaN();
        } else if (m_result.frobeniusExpected == 0.0) {
            m_result.relativeFrobeniusError =
                m_result.frobeniusDifference == 0.0
                    ? (m_options.zeroExpectedNormIsNaN ? std::numeric_limits<double>::quiet_NaN()
                                                       : 0.0)
                    : std::numeric_limits<double>::infinity();
            m_result.relativeMaximumError =
                m_result.maxAbsoluteDifference == 0.0
                    ? (m_options.zeroExpectedNormIsNaN ? std::numeric_limits<double>::quiet_NaN()
                                                       : 0.0)
                    : std::numeric_limits<double>::infinity();
        } else {
            m_result.relativeFrobeniusError =
                m_result.frobeniusDifference / m_result.frobeniusExpected;
            m_result.relativeMaximumError =
                m_result.maxAbsoluteDifference / m_result.maximumExpectedMagnitude;
        }
    }
    if (m_options.relativeFrobeniusTolerance)
        m_result.frobeniusPassed =
            m_result.relativeFrobeniusError <= *m_options.relativeFrobeniusTolerance;

    if (m_result.ulpCompared != 0)
        m_result.averageUlp = m_result.sumUlp / static_cast<double>(m_result.ulpCompared);
    if (m_options.maximumUlpTolerance)
        m_result.ulpPassed = m_result.maximumUlp <= *m_options.maximumUlpTolerance;
    return std::move(m_result);
}

void ComparisonAccumulator::accumulateUlpDistance(double distance) {
    m_result.maximumUlp = std::max(m_result.maximumUlp, distance);
    m_result.sumUlp += distance;
    ++m_result.ulpCompared;
}

void ComparisonAccumulator::accumulateUlp(double exact, double approximation) {
    const double distance =
        ulpDistanceForType(exact, approximation, *m_options.ulpType, m_options.ulpMode);
    accumulateUlpDistance(distance);
}
}  // namespace roc::host_numerics::detail
