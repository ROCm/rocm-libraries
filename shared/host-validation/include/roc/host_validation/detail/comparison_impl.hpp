// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace roc::host_validation {

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

inline ComparisonOptions defaultComparisonOptions(
    ScalarType type, std::optional<double> symmetricRelativeTolerance) {
    ComparisonOptions options;
    options.symmetricRelativeTolerance =
        symmetricRelativeTolerance.value_or(defaultSymmetricRelativeTolerance(type));
    options.strictTolerance = options.symmetricRelativeTolerance != 0.0;
    options.equalNaNs = false;
    return options;
}

inline ComparisonOptions nearComparisonOptions(double absoluteTolerance) {
    ComparisonOptions options;
    options.absoluteTolerance = absoluteTolerance;
    options.equalNaNs = true;
    return options;
}

inline ComparisonOptions allCloseComparisonOptions(double absoluteTolerance,
                                                   double relativeTolerance, bool equalNaNs) {
    ComparisonOptions options;
    options.absoluteTolerance = absoluteTolerance;
    options.relativeTolerance = relativeTolerance;
    options.equalNaNs = equalNaNs;
    return options;
}

inline int ulpMantissaBits(ScalarType type) {
    const auto& info = scalarTypeInfo(type);
    if (info.category == ScalarCategory::SignedInteger ||
        info.category == ScalarCategory::UnsignedInteger ||
        info.category == ScalarCategory::Boolean)
        return 0;
    if (info.category == ScalarCategory::Scale) return info.mantissaBits;
    return info.mantissaBits;
}

inline double ulpDistance(double exact, double approximation, int mantissaBits) {
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

namespace detail {
template <typename T>
ComparisonValue comparisonValue(const T& value) {
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

inline bool isZeroWithOppositeSign(double observed, double expected) {
    return observed == 0.0 && expected == 0.0 && std::signbit(observed) != std::signbit(expected);
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

    if (!options.equalSignedZero && isZeroWithOppositeSign(observed, expected)) {
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

inline double valueMagnitude(const ComparisonValue& value) {
    return value.complex ? std::hypot(value.real, value.imaginary) : std::abs(value.real);
}

inline uint64_t orderedFloatingEncoding(uint64_t raw, uint32_t bitCount) {
    if (bitCount == 0 || bitCount > 64)
        throw std::invalid_argument("Floating encoding width must be in [1, 64].");
    const uint64_t mask =
        bitCount == 64 ? std::numeric_limits<uint64_t>::max() : (uint64_t{1} << bitCount) - 1;
    const uint64_t sign = uint64_t{1} << (bitCount - 1);
    raw &= mask;
    return (raw & sign) != 0 ? (~raw + 1) & mask : raw | sign;
}

inline double encodedUlpDistance(double exact, double approximation, ScalarType type) {
    if (exact == approximation) return 0.0;
    if (!std::isfinite(exact) || !std::isfinite(approximation))
        return std::numeric_limits<double>::infinity();

    uint64_t exactEncoding = 0;
    uint64_t approximationEncoding = 0;
    switch (type) {
        case ScalarType::Float32:
        case ScalarType::ComplexFloat32:
            exactEncoding =
                orderedFloatingEncoding(std::bit_cast<uint32_t>(static_cast<float>(exact)), 32);
            approximationEncoding = orderedFloatingEncoding(
                std::bit_cast<uint32_t>(static_cast<float>(approximation)), 32);
            break;
        case ScalarType::Float64:
        case ScalarType::ComplexFloat64:
            exactEncoding = orderedFloatingEncoding(std::bit_cast<uint64_t>(exact), 64);
            approximationEncoding =
                orderedFloatingEncoding(std::bit_cast<uint64_t>(approximation), 64);
            break;
        case ScalarType::Float16:
            exactEncoding = orderedFloatingEncoding(encodeFloat16(static_cast<float>(exact)), 16);
            approximationEncoding =
                orderedFloatingEncoding(encodeFloat16(static_cast<float>(approximation)), 16);
            break;
        case ScalarType::BFloat16:
            exactEncoding = orderedFloatingEncoding(encodeBFloat16(static_cast<float>(exact)), 16);
            approximationEncoding =
                orderedFloatingEncoding(encodeBFloat16(static_cast<float>(approximation)), 16);
            break;
        case ScalarType::Float8E4M3:
        case ScalarType::Float8E5M2:
        case ScalarType::Float8E4M3Fnuz:
        case ScalarType::Float8E5M2Fnuz:
        case ScalarType::Float6E2M3:
        case ScalarType::Float6E3M2:
        case ScalarType::Float4E2M1:
            exactEncoding =
                orderedFloatingEncoding(encodeBinaryFloat(type, static_cast<float>(exact)),
                                        scalarTypeInfo(type).storageBits);
            approximationEncoding =
                orderedFloatingEncoding(encodeBinaryFloat(type, static_cast<float>(approximation)),
                                        scalarTypeInfo(type).storageBits);
            break;
        case ScalarType::Boolean:
        case ScalarType::UInt8:
        case ScalarType::Int8:
        case ScalarType::UInt16:
        case ScalarType::Int16:
        case ScalarType::UInt32:
        case ScalarType::Int32:
        case ScalarType::UInt64:
        case ScalarType::Int64:
        case ScalarType::Int4:
        case ScalarType::Int12:
            return std::abs(exact - approximation);
        default:
            return ulpDistance(exact, approximation, ulpMantissaBits(type));
    }

    return static_cast<double>(exactEncoding >= approximationEncoding
                                   ? exactEncoding - approximationEncoding
                                   : approximationEncoding - exactEncoding);
}

inline double ulpDistanceForType(double exact, double approximation, ScalarType type,
                                 UlpComparisonMode mode) {
    const auto category = scalarTypeInfo(type).category;
    if (category == ScalarCategory::Boolean || category == ScalarCategory::SignedInteger ||
        category == ScalarCategory::UnsignedInteger)
        return std::abs(exact - approximation);
    if (mode == UlpComparisonMode::EncodedDistance)
        return detail::encodedUlpDistance(exact, approximation, type);
    return ulpDistance(exact, approximation, ulpMantissaBits(type));
}

inline void coordinatesForLinearIndex(size_t logicalIndex, const Shape& shape,
                                      ComparisonIndexOrder order, std::span<size_t> coordinates);

class ComparisonAccumulator {
   public:
    ComparisonAccumulator(const ComparisonOptions& options, const Shape& shape)
        : m_options(options), m_shape(&shape) {
        if (m_options.selection.stride == 0)
            throw std::invalid_argument("Comparison selection stride must be non-zero.");
        m_result.reportedMismatches.reserve(m_options.maxReportedMismatches);
        if (m_options.reportMatchingElements)
            m_result.reportedComparisons.reserve(m_options.maxReportedMismatches);
    }

    void observeReal(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                     double observed, double expected) {
        ++m_result.compared;
        if (observed == expected &&
            (m_options.equalSignedZero || !isZeroWithOppositeSign(observed, expected)) &&
            !m_options.computeFrobenius && !m_options.computeUlp &&
            !m_options.reportMatchingElements) {
            m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed));
            return;
        }

        --m_result.compared;
        observe(logicalIndex, observedOffset, expectedOffset, ComparisonValue{observed, 0.0, false},
                ComparisonValue{expected, 0.0, false});
    }

    void observe(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                 const ComparisonValue& observed, const ComparisonValue& expected) {
        ++m_result.compared;

        const bool complexValue = observed.complex || expected.complex;
        const bool exactReal = observed.real == expected.real;
        const bool exactImaginary = !complexValue || observed.imaginary == expected.imaginary;
        const bool signedZeroMatches =
            m_options.equalSignedZero ||
            (!isZeroWithOppositeSign(observed.real, expected.real) &&
             (!complexValue || !isZeroWithOppositeSign(observed.imaginary, expected.imaginary)));
        if (exactReal && exactImaginary && signedZeroMatches && !m_options.computeFrobenius &&
            !m_options.computeUlp && !m_options.reportMatchingElements) {
            m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed.real));
            if (complexValue)
                m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed.imaginary));
            return;
        }

        const ComponentResult real = compareComponent(observed.real, expected.real, m_options);
        const ComponentResult imaginary =
            complexValue ? compareComponent(observed.imaginary, expected.imaginary, m_options)
                         : ComponentResult{.close = true};

        const bool close = real.close && imaginary.close;
        const bool nonFiniteMismatch = real.nonFiniteMismatch || imaginary.nonFiniteMismatch;
        const bool signedZeroMismatch = real.signedZeroMismatch || imaginary.signedZeroMismatch;

        if (m_options.computePointwiseStatistics) {
            m_result.matchedNaNs +=
                static_cast<size_t>(real.matchedNaN) + static_cast<size_t>(imaginary.matchedNaN);
            m_result.matchedInfinities += static_cast<size_t>(real.matchedInfinity) +
                                          static_cast<size_t>(imaginary.matchedInfinity);
            m_result.nonFiniteMismatches += nonFiniteMismatch;
            m_result.signedZeroMismatches += signedZeroMismatch;
        }

        double difference =
            complexValue ? std::hypot(real.difference, imaginary.difference) : real.difference;
        if (nonFiniteMismatch) difference = std::numeric_limits<double>::infinity();

        if (m_options.computePointwiseStatistics) {
            m_result.maxAbsoluteDifference = std::max(m_result.maxAbsoluteDifference, difference);
            m_result.maxRelativeDifference =
                std::max({m_result.maxRelativeDifference, real.relativeDifference,
                          imaginary.relativeDifference});
            m_result.maxSymmetricRelativeDifference =
                std::max({m_result.maxSymmetricRelativeDifference, real.symmetricRelativeDifference,
                          imaginary.symmetricRelativeDifference});
        }

        if (m_options.computeFrobenius) {
            if (!nonFiniteMismatch && std::isfinite(observed.real) &&
                std::isfinite(expected.real) &&
                (!complexValue ||
                 (std::isfinite(observed.imaginary) && std::isfinite(expected.imaginary)))) {
                const double observedMagnitude = valueMagnitude(observed);
                const double expectedMagnitude = valueMagnitude(expected);
                m_result.maximumObservedMagnitude =
                    std::max(m_result.maximumObservedMagnitude, observedMagnitude);
                m_result.maximumExpectedMagnitude =
                    std::max(m_result.maximumExpectedMagnitude, expectedMagnitude);
                m_observedSquares +=
                    static_cast<long double>(observedMagnitude) * observedMagnitude;
                m_expectedSquares +=
                    static_cast<long double>(expectedMagnitude) * expectedMagnitude;
                m_differenceSquares += static_cast<long double>(difference) * difference;
            } else if (nonFiniteMismatch) {
                m_differenceSquares = std::numeric_limits<long double>::infinity();
            }
        }

        if (m_options.computeUlp) {
            if (m_options.ulpType == ScalarType::Count)
                throw std::invalid_argument("ULP comparison requires an explicit scalar type.");
            if (real.matchedNaN || real.matchedInfinity)
                ++m_result.ulpCompared;
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
            m_options.pointwise && !close &&
            m_result.reportedMismatches.size() < m_options.maxReportedMismatches;
        if (reportComparison || reportMismatch) {
            std::vector<size_t> coordinates(m_shape->rank(), 0);
            coordinatesForLinearIndex(logicalIndex, *m_shape, m_options.selection.indexOrder,
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
                std::max(real.tolerance, imaginary.tolerance),
                close,
            };
            if (reportComparison) m_result.reportedComparisons.push_back(sample);
            if (reportMismatch) m_result.reportedMismatches.push_back(sample);
        }

        if (m_options.pointwise && !close) ++m_result.mismatches;
    }

    ComparisonResult finish() {
        m_result.pointwisePassed = !m_options.pointwise || m_result.mismatches == 0;
        if (m_options.computeFrobenius) {
            m_result.frobeniusDifference = std::sqrt(static_cast<double>(m_differenceSquares));
            m_result.frobeniusObserved = std::sqrt(static_cast<double>(m_observedSquares));
            m_result.frobeniusExpected = std::sqrt(static_cast<double>(m_expectedSquares));
            if (m_result.frobeniusExpected == 0.0) {
                m_result.relativeFrobeniusError = m_result.frobeniusDifference == 0.0
                                                      ? 0.0
                                                      : std::numeric_limits<double>::infinity();
            } else {
                m_result.relativeFrobeniusError =
                    m_result.frobeniusDifference / m_result.frobeniusExpected;
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

   private:
    void accumulateUlp(double exact, double approximation) {
        const double distance =
            ulpDistanceForType(exact, approximation, m_options.ulpType, m_options.ulpMode);
        m_result.maximumUlp = std::max(m_result.maximumUlp, distance);
        m_result.sumUlp += distance;
        ++m_result.ulpCompared;
    }

    ComparisonOptions m_options;
    const Shape* m_shape = nullptr;
    ComparisonResult m_result;
    long double m_differenceSquares = 0.0;
    long double m_observedSquares = 0.0;
    long double m_expectedSquares = 0.0;
};

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

template <typename Function>
void forEachSelectedIndex(const Shape& shape, const ComparisonSelection& selection,
                          Function&& function) {
    if (selection.stride == 0)
        throw std::invalid_argument("Comparison selection stride must be non-zero.");

    const size_t total = shape.elementCount();
    std::vector<size_t> coordinates(shape.rank(), 0);
    size_t selected = 0;
    for (size_t logicalIndex = selection.first;
         logicalIndex < total && selected < selection.maxElements; ++selected) {
        coordinatesForLinearIndex(logicalIndex, shape, selection.indexOrder, coordinates);
        function(logicalIndex, std::span<const size_t>(coordinates));
        if (selection.stride > std::numeric_limits<size_t>::max() - logicalIndex) break;
        logicalIndex += selection.stride;
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

inline ComparisonValue loadComparisonValue(TensorView view, ptrdiff_t logicalOffset) {
    const auto storage = view.storage();
    if (scalarTypeInfo(view.type()).category == ScalarCategory::Complex)
        return comparisonValue(
            decodeScalar<std::complex<double>>(view.type(), storage, logicalOffset));
    return comparisonValue(decodeScalar<double>(view.type(), storage, logicalOffset));
}

template <typename Tag>
ComparisonValue loadComparisonValueKnown(std::span<const std::byte> storage,
                                         ptrdiff_t logicalOffset) {
    if (logicalOffset < 0) throw std::out_of_range("Comparison logical offset is negative.");

    constexpr ScalarType type = Tag::type;
    constexpr size_t storageBits = scalarTypeInfo(type).storageBits;
    const size_t byteOffset = static_cast<size_t>(logicalOffset) * storageBits / 8;
    const auto readNativeUnchecked = [&]<typename T>() {
        T value;
        std::memcpy(&value, storage.data() + byteOffset, sizeof(T));
        return value;
    };

    if constexpr (type == ScalarType::Boolean)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint8_t>() != 0),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt8)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint8_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int8)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int8_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt16)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint16_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int16)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int16_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint32_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int32_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::UInt64)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<uint64_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Int64)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<int64_t>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float16)
        return {
            static_cast<double>(decodeFloat16(readNativeUnchecked.template operator()<uint16_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::BFloat16)
        return {
            static_cast<double>(
                decodeBFloat16(readNativeUnchecked.template operator()<uint16_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float32)
        return {
            static_cast<double>(readNativeUnchecked.template operator()<float>()),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::Float64)
        return { readNativeUnchecked.template operator()<double>(), 0.0, false };
    else if constexpr (type == ScalarType::ComplexFloat32) {
        const auto value = readNativeUnchecked.template operator()<std::complex<float>>();
        return {static_cast<double>(value.real()), static_cast<double>(value.imag()), true};
    } else if constexpr (type == ScalarType::ComplexFloat64) {
        const auto value = readNativeUnchecked.template operator()<std::complex<double>>();
        return {value.real(), value.imag(), true};
    } else if constexpr (type == ScalarType::Float8E4M3 || type == ScalarType::Float8E5M2 ||
                         type == ScalarType::Float8E4M3Fnuz || type == ScalarType::Float8E5M2Fnuz ||
                         type == ScalarType::E5M3)
        return {
            static_cast<double>(
                decodeBinaryFloat(type, readNativeUnchecked.template operator()<uint8_t>())),
            0.0,
            false
        };
    else if constexpr (type == ScalarType::E8M0)
        return {
            static_cast<double>(decodeE8M0(readNativeUnchecked.template operator()<uint8_t>())),
            0.0,
            false
        };
    else
        return comparisonValue(decodeScalarKnown<type, double>(storage, logicalOffset));
}

template <typename Tag>
auto loadFastComparisonReal(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    constexpr ScalarType type = Tag::type;
    if constexpr (type == ScalarType::Float32) {
        float value;
        std::memcpy(&value, storage.data() + static_cast<size_t>(logicalOffset) * sizeof(float),
                    sizeof(value));
        return value;
    } else if constexpr (type == ScalarType::Float64) {
        double value;
        std::memcpy(&value, storage.data() + static_cast<size_t>(logicalOffset) * sizeof(double),
                    sizeof(value));
        return value;
    } else {
        return loadComparisonValueKnown<Tag>(storage, logicalOffset).real;
    }
}

inline bool isUnwrittenSentinelValue(ScalarType type, const ComparisonValue& value) {
    const auto& info = scalarTypeInfo(type);
    switch (info.category) {
        case ScalarCategory::SignedInteger: {
            const double lowest = info.storageBits == 64
                                      ? static_cast<double>(std::numeric_limits<int64_t>::lowest())
                                      : -std::ldexp(1.0, info.storageBits - 1);
            return value.real == lowest;
        }
        case ScalarCategory::UnsignedInteger:
            return value.real == std::ldexp(1.0, info.storageBits) - 1.0;
        case ScalarCategory::FloatingPoint:
        case ScalarCategory::Scale:
            return info.supportsInfinity ? std::isinf(value.real) : std::isnan(value.real);
        case ScalarCategory::Complex:
            return std::isinf(value.real) && std::isinf(value.imaginary);
        case ScalarCategory::Boolean:
            return false;
    }
    return false;
}

template <typename Observed, typename Expected>
inline bool valuesCloseFast(Observed observed, Expected expected,
                            const ComparisonOptions& options) {
    if (observed == expected)
        return options.equalSignedZero || !isZeroWithOppositeSign(observed, expected);
    if (std::isnan(observed) || std::isnan(expected))
        return options.equalNaNs && std::isnan(observed) && std::isnan(expected);
    if (std::isinf(observed) || std::isinf(expected)) return false;

    const double difference = std::abs(observed - expected);
    const double tolerance =
        options.absoluteTolerance + options.relativeTolerance * std::abs(expected) +
        options.symmetricRelativeTolerance * (std::abs(observed) + std::abs(expected) + 1.0);
    return options.strictTolerance ? difference < tolerance : difference <= tolerance;
}

template <typename Tag>
ComparisonResult comparePointwiseOnlyKnown(TensorView observed, TensorView expected,
                                           const ComparisonOptions& options) {
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonResult result;
        forEachSelectedOffsetPair(
            observed.layout(), expected.layout(), options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                ++result.compared;
                bool close = false;
                if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                    const ComparisonValue observedValue =
                        loadComparisonValueKnown<Tag>(observed.storage(), observedOffset);
                    const ComparisonValue expectedValue =
                        loadComparisonValueKnown<Tag>(expected.storage(), expectedOffset);
                    close = predicate(observedValue.real, expectedValue.real);
                    close = close && predicate(observedValue.imaginary, expectedValue.imaginary);
                } else {
                    close =
                        predicate(loadFastComparisonReal<Tag>(observed.storage(), observedOffset),
                                  loadFastComparisonReal<Tag>(expected.storage(), expectedOffset));
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
            using Real = decltype(observedValue - expectedValue);
            const Real typedTolerance = static_cast<Real>(tolerance);
            return observedValue == expectedValue ||
                   std::abs(observedValue - expectedValue) <
                       typedTolerance * (std::abs(observedValue) + std::abs(expectedValue) +
                                         static_cast<Real>(1));
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

template <typename Observed, typename Expected>
ComparisonResult comparePointwiseOnlyTyped(std::span<const Observed> observedStorage,
                                           const Layout& observedLayout,
                                           std::span<const Expected> expectedStorage,
                                           const Layout& expectedLayout,
                                           const ComparisonOptions& options) {
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonResult result;
        if (options.selection.first == 0 && options.selection.stride == 1 &&
            options.selection.indexOrder == ComparisonIndexOrder::FirstDimensionFastest &&
            observedLayout.shape().rank() != 0) {
            const Shape& shape = observedLayout.shape();
            const size_t innerSize = shape[0];
            const size_t selectedTotal =
                std::min(shape.elementCount(), options.selection.maxElements);
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
                            comparisonValue(observedStorage[observedOffset]);
                        const ComparisonValue expectedValue =
                            comparisonValue(expectedStorage[expectedOffset]);
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
                        comparisonValue(observedStorage[observedOffset]);
                    const ComparisonValue expectedValue =
                        comparisonValue(expectedStorage[expectedOffset]);
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
            using Real = decltype(observedValue - expectedValue);
            const Real typedTolerance = static_cast<Real>(tolerance);
            return observedValue == expectedValue ||
                   std::abs(observedValue - expectedValue) <
                       typedTolerance * (std::abs(observedValue) + std::abs(expectedValue) +
                                         static_cast<Real>(1));
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
}  // namespace detail

inline double encodedUlpDistance(double exact, double approximation, ScalarType type) {
    return detail::encodedUlpDistance(exact, approximation, type);
}

template <typename Observed, typename Expected>
bool valuesClose(const Observed& observed, const Expected& expected,
                 const ComparisonOptions& options) {
    const ComparisonValue observedValue = detail::comparisonValue(observed);
    const ComparisonValue expectedValue = detail::comparisonValue(expected);
    const detail::ComponentResult real =
        detail::compareComponent(observedValue.real, expectedValue.real, options);
    if (!real.close) return false;
    if (observedValue.complex || expectedValue.complex)
        return detail::compareComponent(observedValue.imaginary, expectedValue.imaginary, options)
            .close;
    return true;
}

template <typename Observed, typename Expected>
ComparisonResult compare(std::span<const Observed> observedStorage, const Layout& observedLayout,
                         std::span<const Expected> expectedStorage, const Layout& expectedLayout,
                         const ComparisonOptions& options) {
    if (observedLayout.shape() != expectedLayout.shape())
        throw std::invalid_argument("Host validation comparison shape mismatch.");

    const auto [observedLower, observedUpper] = detail::elementBounds(observedLayout);
    const auto [expectedLower, expectedUpper] = detail::elementBounds(expectedLayout);
    if (observedLower < 0 || expectedLower < 0 ||
        (observedUpper >= 0 && static_cast<size_t>(observedUpper) >= observedStorage.size()) ||
        (expectedUpper >= 0 && static_cast<size_t>(expectedUpper) >= expectedStorage.size()))
        throw std::invalid_argument(
            "Host validation comparison storage is too small for its layout.");

    if (options.pointwise && !options.computePointwiseStatistics && !options.computeFrobenius &&
        !options.computeUlp && !options.relativeFrobeniusTolerance &&
        !options.maximumUlpTolerance) {
        ComparisonResult result = detail::comparePointwiseOnlyTyped(
            observedStorage, observedLayout, expectedStorage, expectedLayout, options);
        const bool needsSamples = options.maxReportedMismatches != 0 &&
                                  (options.reportMatchingElements || result.mismatches != 0);
        if (!needsSamples) return result;

        ComparisonOptions detailed = options;
        detailed.computePointwiseStatistics = true;
        return compare(observedStorage, observedLayout, expectedStorage, expectedLayout, detailed);
    }

    detail::ComparisonAccumulator accumulator(options, observedLayout.shape());
    detail::forEachSelectedOffsetPair(
        observedLayout, expectedLayout, options.selection,
        [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
            accumulator.observe(logicalIndex, observedOffset, expectedOffset,
                                detail::comparisonValue(observedStorage[observedOffset]),
                                detail::comparisonValue(expectedStorage[expectedOffset]));
        });
    return accumulator.finish();
}

template <typename Observed, typename Expected>
ComparisonResult compare(std::span<const Observed> observed, std::span<const Expected> expected,
                         const ComparisonOptions& options) {
    return compare(observed, Layout::contiguous(Shape{observed.size()}), expected,
                   Layout::contiguous(Shape{expected.size()}), options);
}

template <typename Observed, typename Expected>
ComparisonResult compare(ConstMatrixView<Observed> observed, ConstMatrixView<Expected> expected,
                         const ComparisonOptions& options) {
    if (observed.rows() != expected.rows() || observed.columns() != expected.columns())
        throw std::invalid_argument("Host validation matrix comparison shape mismatch.");

    const Shape shape{observed.rows(), observed.columns()};
    ComparisonOptions matrixOptions = options;
    matrixOptions.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    detail::ComparisonAccumulator accumulator(matrixOptions, shape);
    detail::forEachSelectedIndex(shape, matrixOptions.selection,
                                 [&](size_t logicalIndex, std::span<const size_t> coordinates) {
                                     const size_t row = coordinates[0];
                                     const size_t column = coordinates[1];
                                     accumulator.observe(
                                         logicalIndex, static_cast<ptrdiff_t>(logicalIndex),
                                         static_cast<ptrdiff_t>(logicalIndex),
                                         detail::comparisonValue(observed(row, column)),
                                         detail::comparisonValue(expected(row, column)));
                                 });
    return accumulator.finish();
}

inline ComparisonResult compare(TensorView observed, TensorView expected,
                                const ComparisonOptions& options) {
    if (observed.shape() != expected.shape())
        throw std::invalid_argument("Host validation tensor comparison shape mismatch.");

    if (observed.type() == expected.type() && options.pointwise &&
        !options.computePointwiseStatistics && !options.computeFrobenius && !options.computeUlp &&
        !options.relativeFrobeniusTolerance && !options.maximumUlpTolerance) {
        ComparisonResult result = visitScalarType(observed.type(), [&]<typename Tag>() {
            return detail::comparePointwiseOnlyKnown<Tag>(observed, expected, options);
        });
        const bool needsSamples = options.maxReportedMismatches != 0 &&
                                  (options.reportMatchingElements || result.mismatches != 0);
        if (!needsSamples) return result;

        ComparisonOptions detailed = options;
        detailed.computePointwiseStatistics = true;
        return compare(observed, expected, detailed);
    }

    detail::ComparisonAccumulator accumulator(options, observed.shape());
    if (observed.type() == expected.type()) {
        visitScalarType(observed.type(), [&]<typename Tag>() {
            detail::forEachSelectedOffsetPair(
                observed.layout(), expected.layout(), options.selection,
                [&](size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                        accumulator.observe(logicalIndex, observedOffset, expectedOffset,
                                            detail::loadComparisonValueKnown<Tag>(
                                                observed.storage(), observedOffset),
                                            detail::loadComparisonValueKnown<Tag>(
                                                expected.storage(), expectedOffset));
                    } else {
                        accumulator.observeReal(logicalIndex, observedOffset, expectedOffset,
                                                detail::loadComparisonValueKnown<Tag>(
                                                    observed.storage(), observedOffset)
                                                    .real,
                                                detail::loadComparisonValueKnown<Tag>(
                                                    expected.storage(), expectedOffset)
                                                    .real);
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

template <typename Observed, typename Expected>
std::optional<ComparisonTolerance> findAllCloseTolerance(std::span<const Observed> observedStorage,
                                                         const Layout& observedLayout,
                                                         std::span<const Expected> expectedStorage,
                                                         const Layout& expectedLayout,
                                                         std::span<const double> absoluteCandidates,
                                                         std::span<const double> relativeCandidates,
                                                         ComparisonOptions options) {
    for (const double absolute : absoluteCandidates) {
        for (const double relative : relativeCandidates) {
            options.absoluteTolerance = absolute;
            options.relativeTolerance = relative;
            options.symmetricRelativeTolerance = 0.0;
            if (compare(observedStorage, observedLayout, expectedStorage, expectedLayout, options)
                    .passed())
                return ComparisonTolerance{absolute, relative};
        }
    }
    return std::nullopt;
}

inline std::optional<ComparisonTolerance> findAllCloseTolerance(
    TensorView observed, TensorView expected, std::span<const double> absoluteCandidates,
    std::span<const double> relativeCandidates, ComparisonOptions options) {
    for (const double absolute : absoluteCandidates) {
        for (const double relative : relativeCandidates) {
            options.absoluteTolerance = absolute;
            options.relativeTolerance = relative;
            options.symmetricRelativeTolerance = 0.0;
            if (compare(observed, expected, options).passed())
                return ComparisonTolerance{absolute, relative};
        }
    }
    return std::nullopt;
}

inline SentinelResult checkUnwrittenSentinel(ScalarType type, std::span<const std::byte> storage,
                                             size_t firstElement, size_t elementCount,
                                             SentinelRegion region, size_t maxReportedMismatches) {
    const size_t storageBits = scalarTypeInfo(type).storageBits;
    if (storageBits == 0) throw std::invalid_argument("Sentinel scalar type has no storage.");
    const uint64_t requiredBits = static_cast<uint64_t>(firstElement + elementCount) * storageBits;
    if ((requiredBits + 7) / 8 > storage.size())
        throw std::invalid_argument(
            "Sentinel storage is too small for the requested element range.");

    SentinelResult result;
    result.checked = elementCount;
    result.reportedMismatches.reserve(std::min(maxReportedMismatches, elementCount));
    for (size_t index = 0; index < elementCount; ++index) {
        const ptrdiff_t offset = static_cast<ptrdiff_t>(firstElement + index);
        ComparisonValue value;
        if (scalarTypeInfo(type).category == ScalarCategory::Complex)
            value = detail::comparisonValue(
                detail::decodeScalar<std::complex<double>>(type, storage, offset));
        else
            value = detail::comparisonValue(detail::decodeScalar<double>(type, storage, offset));
        if (!detail::isUnwrittenSentinelValue(type, value)) {
            ++result.mismatches;
            if (result.reportedMismatches.size() < maxReportedMismatches)
                result.reportedMismatches.push_back({region, index, value});
        }
    }
    return result;
}

inline SentinelResult checkUnusedTensorStorage(TensorView logicalTensor, size_t allocatedElements,
                                               SentinelRegion region,
                                               size_t maxReportedMismatches) {
    const auto& layout = logicalTensor.layout();
    std::vector<bool> used(allocatedElements, false);
    detail::forEachSelectedIndex(
        layout.shape(), {}, [&](size_t, std::span<const size_t> coordinates) {
            const ptrdiff_t offset = layout.elementOffset(coordinates);
            if (offset < 0 || static_cast<size_t>(offset) >= allocatedElements)
                throw std::invalid_argument("Logical tensor addresses outside allocated storage.");
            used[static_cast<size_t>(offset)] = true;
        });

    SentinelResult result;
    result.reportedMismatches.reserve(maxReportedMismatches);
    for (size_t index = 0; index < allocatedElements; ++index) {
        if (used[index]) continue;
        const SentinelResult element =
            checkUnwrittenSentinel(logicalTensor.type(), logicalTensor.storage(), index, 1, region,
                                   maxReportedMismatches - result.reportedMismatches.size());
        result.append(element, maxReportedMismatches);
        if (!element.reportedMismatches.empty()) result.reportedMismatches.back().index = index;
    }
    return result;
}
}  // namespace roc::host_validation
