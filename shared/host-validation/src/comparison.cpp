// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/comparison_common.hpp"

namespace roc::host_validation {

ComparisonOptions defaultComparisonOptions(ScalarType type,
                                           std::optional<double> symmetricRelativeTolerance) {
    ComparisonOptions options;
    options.symmetricRelativeTolerance =
        symmetricRelativeTolerance.value_or(defaultSymmetricRelativeTolerance(type));
    options.strictTolerance = options.symmetricRelativeTolerance != 0.0;
    options.equalNaNs = false;
    return options;
}

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
    options.complexPointwiseMode = ComplexPointwiseMode::Magnitude;
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

namespace detail {
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
        case ScalarType::E5M3:
        case ScalarType::E4M3: {
            const BinaryFloatFormat format = binaryFloatFormat(type);
            const uint64_t exactRaw = encodeBinaryFloat(type, static_cast<float>(exact));
            const uint64_t approximationRaw =
                encodeBinaryFloat(type, static_cast<float>(approximation));
            exactEncoding =
                format.hasSign ? orderedFloatingEncoding(exactRaw, format.totalBits) : exactRaw;
            approximationEncoding =
                format.hasSign ? orderedFloatingEncoding(approximationRaw, format.totalBits)
                               : approximationRaw;
            break;
        }
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

class ComparisonAccumulator {
   public:
    ComparisonAccumulator(const ComparisonOptions& options, const Shape& shape)
        : m_options(options), m_shape(&shape) {
        m_result.pointwiseEvaluated = m_options.pointwise;
        m_result.frobeniusEvaluated = m_options.relativeFrobeniusTolerance.has_value();
        m_result.ulpEvaluated = m_options.maximumUlpTolerance.has_value();
        m_result.reportedMismatches.reserve(m_options.maxReportedMismatches);
        if (m_options.reportMatchingElements)
            m_result.reportedComparisons.reserve(m_options.maxReportedMismatches);
    }

    void observeReal(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                     double observed, double expected,
                     std::optional<bool> pointwiseDecision = std::nullopt) {
        ++m_result.compared;
        if ((!pointwiseDecision || *pointwiseDecision) && observed == expected &&
            (m_options.equalSignedZero || !oppositeZeroSigns(observed, expected)) &&
            !m_options.computeFrobenius && !m_options.computeUlp &&
            !m_options.reportMatchingElements) {
            if (m_options.computePointwiseStatistics)
                m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed));
            return;
        }

        --m_result.compared;
        observe(logicalIndex, observedOffset, expectedOffset, ComparisonValue{observed, 0.0, false},
                ComparisonValue{expected, 0.0, false}, pointwiseDecision);
    }

    void observe(size_t logicalIndex, ptrdiff_t observedOffset, ptrdiff_t expectedOffset,
                 const ComparisonValue& observed, const ComparisonValue& expected,
                 std::optional<bool> pointwiseDecision = std::nullopt) {
        ++m_result.compared;

        const bool complexValue = observed.complex || expected.complex;
        const bool exactReal = observed.real == expected.real;
        const bool exactImaginary = !complexValue || observed.imaginary == expected.imaginary;
        const bool signedZeroMatches =
            m_options.equalSignedZero ||
            (!oppositeZeroSigns(observed.real, expected.real) &&
             (!complexValue || !oppositeZeroSigns(observed.imaginary, expected.imaginary)));
        if ((!pointwiseDecision || *pointwiseDecision) && exactReal && exactImaginary &&
            signedZeroMatches && !m_options.computeFrobenius && !m_options.computeUlp &&
            !m_options.reportMatchingElements) {
            if (m_options.computePointwiseStatistics) {
                if (complexValue &&
                    m_options.complexPointwiseMode == ComplexPointwiseMode::Magnitude) {
                    m_result.matchedInfinities += static_cast<size_t>(
                        std::isinf(observed.real) || std::isinf(observed.imaginary));
                } else {
                    m_result.matchedInfinities += static_cast<size_t>(std::isinf(observed.real));
                    if (complexValue)
                        m_result.matchedInfinities +=
                            static_cast<size_t>(std::isinf(observed.imaginary));
                }
            }
            return;
        }

        const ComponentResult real = compareComponent(observed.real, expected.real, m_options);
        const ComponentResult imaginary =
            complexValue ? compareComponent(observed.imaginary, expected.imaginary, m_options)
                         : ComponentResult{.close = true};
        const bool magnitudeMode =
            complexValue && m_options.complexPointwiseMode == ComplexPointwiseMode::Magnitude;
        const ComponentResult magnitude =
            magnitudeMode ? compareComplexMagnitude(observed, expected, m_options)
                          : ComponentResult{.close = true};

        const bool close = pointwiseDecision.value_or(
            magnitudeMode ? magnitude.close : real.close && imaginary.close);
        const bool nonFiniteMismatch = magnitudeMode
                                           ? magnitude.nonFiniteMismatch
                                           : real.nonFiniteMismatch || imaginary.nonFiniteMismatch;

        if (m_options.computePointwiseStatistics) {
            if (magnitudeMode) {
                m_result.matchedNaNs += static_cast<size_t>(magnitude.matchedNaN);
                m_result.matchedInfinities += static_cast<size_t>(magnitude.matchedInfinity);
            } else {
                m_result.matchedNaNs += static_cast<size_t>(real.matchedNaN) +
                                        static_cast<size_t>(imaginary.matchedNaN);
                m_result.matchedInfinities += static_cast<size_t>(real.matchedInfinity) +
                                              static_cast<size_t>(imaginary.matchedInfinity);
            }
            m_result.nonFiniteMismatches += nonFiniteMismatch;
            m_result.signedZeroMismatches +=
                magnitudeMode ? magnitude.signedZeroMismatch
                              : real.signedZeroMismatch || imaginary.signedZeroMismatch;
        }

        double difference = magnitudeMode
                                ? magnitude.difference
                                : (complexValue ? std::hypot(real.difference, imaginary.difference)
                                                : real.difference);
        if (nonFiniteMismatch) difference = std::numeric_limits<double>::infinity();

        if (m_options.computePointwiseStatistics) {
            m_result.maxAbsoluteDifference = std::max(m_result.maxAbsoluteDifference, difference);
            if (magnitudeMode) {
                m_result.maxRelativeDifference =
                    std::max(m_result.maxRelativeDifference, magnitude.relativeDifference);
                m_result.maxSymmetricRelativeDifference = std::max(
                    m_result.maxSymmetricRelativeDifference, magnitude.symmetricRelativeDifference);
            } else {
                m_result.maxRelativeDifference =
                    std::max({m_result.maxRelativeDifference, real.relativeDifference,
                              imaginary.relativeDifference});
                m_result.maxSymmetricRelativeDifference = std::max(
                    {m_result.maxSymmetricRelativeDifference, real.symmetricRelativeDifference,
                     imaginary.symmetricRelativeDifference});
            }
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
                magnitudeMode ? magnitude.tolerance : std::max(real.tolerance, imaginary.tolerance),
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
            ulpDistanceForType(exact, approximation, *m_options.ulpType, m_options.ulpMode);
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

inline ComparisonValue loadComparisonValue(const Tensor& view, ptrdiff_t logicalOffset) {
    const auto storage = view.rawEncodedBackingStorage();
    if (scalarTypeInfo(view.type()).category == ScalarCategory::Complex)
        return typedComparisonValue(
            decodeScalar<std::complex<double>>(view.type(), storage, logicalOffset));
    return typedComparisonValue(decodeScalar<double>(view.type(), storage, logicalOffset));
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
                         type == ScalarType::E5M3 || type == ScalarType::E4M3)
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
        return typedComparisonValue(decodeScalarKnown<type, double>(storage, logicalOffset));
}

template <typename Tag>
auto loadFastComparisonReal(std::span<const std::byte> storage, ptrdiff_t logicalOffset) {
    constexpr ScalarType type = Tag::type;
    constexpr ScalarCategory category = scalarTypeInfo(type).category;

    const auto readNativeUnchecked = [&]<typename T>() {
        T value;
        std::memcpy(&value, storage.data() + static_cast<size_t>(logicalOffset) * sizeof(T),
                    sizeof(T));
        return value;
    };

    if constexpr (type == ScalarType::Float32) {
        return readNativeUnchecked.template operator()<float>();
    } else if constexpr (type == ScalarType::Float64) {
        return readNativeUnchecked.template operator()<double>();
    } else if constexpr (category == ScalarCategory::Boolean) {
        return readNativeUnchecked.template operator()<uint8_t>() != 0;
    } else if constexpr (category == ScalarCategory::SignedInteger) {
        if constexpr (std::is_void_v<typename Tag::Storage>)
            return decodeScalarKnown<type, int64_t>(storage, logicalOffset);
        else
            return readNativeUnchecked.template operator()<typename Tag::Storage>();
    } else if constexpr (category == ScalarCategory::UnsignedInteger) {
        return readNativeUnchecked.template operator()<typename Tag::Storage>();
    } else {
        return loadComparisonValueKnown<Tag>(storage, logicalOffset).real;
    }
}

template <typename Tag>
bool knownPointwiseDecision(const Tensor& observed, ptrdiff_t observedOffset,
                            const Tensor& expected, ptrdiff_t expectedOffset,
                            const ComparisonOptions& options) {
    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
        const ComparisonValue observedValue =
            loadComparisonValueKnown<Tag>(observed.rawEncodedBackingStorage(), observedOffset);
        const ComparisonValue expectedValue =
            loadComparisonValueKnown<Tag>(expected.rawEncodedBackingStorage(), expectedOffset);
        if (options.complexPointwiseMode == ComplexPointwiseMode::Magnitude)
            return compareComplexMagnitude(observedValue, expectedValue, options).close;
        return pointwiseValuesClose(observedValue.real, expectedValue.real, options) &&
               pointwiseValuesClose(observedValue.imaginary, expectedValue.imaginary, options);
    } else {
        return pointwiseValuesClose(
            loadFastComparisonReal<Tag>(observed.rawEncodedBackingStorage(), observedOffset),
            loadFastComparisonReal<Tag>(expected.rawEncodedBackingStorage(), expectedOffset),
            options);
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

inline void validateSentinelRange(ScalarType type, std::span<const std::byte> storage,
                                  size_t firstElement, size_t elementCount) {
    const size_t storageBits = scalarTypeInfo(type).storageBits;
    if (storageBits == 0) throw std::invalid_argument("Sentinel scalar type has no storage.");
    if (firstElement > std::numeric_limits<size_t>::max() - elementCount)
        throw std::invalid_argument("Sentinel element range overflows.");

    const size_t endElement = firstElement + elementCount;
    if (endElement != 0 &&
        endElement - 1 > static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()))
        throw std::invalid_argument("Sentinel element range exceeds supported element offsets.");
    if (endElement != 0 && storageBits > std::numeric_limits<size_t>::max() / endElement)
        throw std::invalid_argument("Sentinel element range overflows.");

    const size_t requiredBits = endElement * storageBits;
    const size_t requiredBytes = requiredBits / 8 + static_cast<size_t>(requiredBits % 8 != 0);
    if (requiredBytes > storage.size())
        throw std::invalid_argument(
            "Sentinel storage is too small for the requested element range.");
}

template <typename Tag>
ComparisonResult comparePointwiseOnlyKnown(const Tensor& observed, const Tensor& expected,
                                           const ComparisonOptions& options) {
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonResult result;
        result.pointwiseEvaluated = true;
        if (options.selection.first == 0 && options.selection.stride == 1 &&
            options.selection.indexOrder == IndexOrder::FirstDimensionFastest &&
            observed.shape().rank() != 0) {
            const Shape& shape = observed.shape();
            const size_t innerSize = shape[0];
            const size_t selectedTotal =
                std::min(shape.elementCount(), options.selection.maxElements);
            if (selectedTotal == 0) return result;
            const size_t outerCount = (selectedTotal + innerSize - 1) / innerSize;
            std::vector<size_t> coordinates(shape.rank(), 0);

            for (size_t outerIndex = 0; outerIndex < outerCount; ++outerIndex) {
                size_t remaining = outerIndex;
                ptrdiff_t observedBase = observed.layout().offset();
                ptrdiff_t expectedBase = expected.layout().offset();
                for (size_t dimension = 1; dimension < shape.rank(); ++dimension) {
                    coordinates[dimension] = remaining % shape[dimension];
                    remaining /= shape[dimension];
                    observedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    observed.layout().strides()[dimension];
                    expectedBase += static_cast<ptrdiff_t>(coordinates[dimension]) *
                                    expected.layout().strides()[dimension];
                }

                const size_t logicalBase = outerIndex * innerSize;
                const size_t count = std::min(innerSize, selectedTotal - logicalBase);
                for (size_t innerIndex = 0; innerIndex < count; ++innerIndex) {
                    const ptrdiff_t observedOffset =
                        observedBase +
                        static_cast<ptrdiff_t>(innerIndex) * observed.layout().strides()[0];
                    const ptrdiff_t expectedOffset =
                        expectedBase +
                        static_cast<ptrdiff_t>(innerIndex) * expected.layout().strides()[0];
                    bool close = false;
                    if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                        const ComparisonValue observedValue = loadComparisonValueKnown<Tag>(
                            observed.rawEncodedBackingStorage(), observedOffset);
                        const ComparisonValue expectedValue = loadComparisonValueKnown<Tag>(
                            expected.rawEncodedBackingStorage(), expectedOffset);
                        if (options.complexPointwiseMode == ComplexPointwiseMode::Magnitude) {
                            close = compareComplexMagnitude(observedValue, expectedValue, options)
                                        .close;
                        } else {
                            close = predicate(observedValue.real, expectedValue.real);
                            close = close &&
                                    predicate(observedValue.imaginary, expectedValue.imaginary);
                        }
                    } else {
                        close = predicate(loadFastComparisonReal<Tag>(
                                              observed.rawEncodedBackingStorage(), observedOffset),
                                          loadFastComparisonReal<Tag>(
                                              expected.rawEncodedBackingStorage(), expectedOffset));
                    }
                    result.mismatches += static_cast<size_t>(!close);
                }
            }
            result.compared = selectedTotal;
            result.pointwisePassed = result.mismatches == 0;
            return result;
        }

        forEachSelectedOffsetPair(
            observed.layout(), expected.layout(), options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                ++result.compared;
                bool close = false;
                if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                    const ComparisonValue observedValue = loadComparisonValueKnown<Tag>(
                        observed.rawEncodedBackingStorage(), observedOffset);
                    const ComparisonValue expectedValue = loadComparisonValueKnown<Tag>(
                        expected.rawEncodedBackingStorage(), expectedOffset);
                    if (options.complexPointwiseMode == ComplexPointwiseMode::Magnitude) {
                        close =
                            compareComplexMagnitude(observedValue, expectedValue, options).close;
                    } else {
                        close = predicate(observedValue.real, expectedValue.real);
                        close =
                            close && predicate(observedValue.imaginary, expectedValue.imaginary);
                    }
                } else {
                    close = predicate(loadFastComparisonReal<Tag>(
                                          observed.rawEncodedBackingStorage(), observedOffset),
                                      loadFastComparisonReal<Tag>(
                                          expected.rawEncodedBackingStorage(), expectedOffset));
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
}  // namespace detail

double encodedUlpDistance(double exact, double approximation, ScalarType type) {
    return detail::encodedUlpDistance(exact, approximation, type);
}

ComparisonResult compare(const Tensor& observed, const Tensor& expected,
                         const ComparisonOptions& options) {
    detail::validateComparisonOptions(options);
    if (observed.shape() != expected.shape())
        throw std::invalid_argument("Host validation tensor comparison shape mismatch.");

    if (observed.type() == expected.type() && detail::pointwiseOnlyComparison(options)) {
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
                    std::optional<bool> decision;
                    if (options.pointwise)
                        decision.emplace(detail::knownPointwiseDecision<Tag>(
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

std::optional<ComparisonTolerance> findAllCloseTolerance(const Tensor& observed,
                                                         const Tensor& expected,
                                                         std::span<const double> absoluteCandidates,
                                                         std::span<const double> relativeCandidates,
                                                         ComparisonOptions options) {
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

SentinelResult checkUnwrittenSentinel(ScalarType type, std::span<const std::byte> storage,
                                      size_t firstElement, size_t elementCount,
                                      SentinelRegion region, size_t maxReportedMismatches) {
    detail::validateSentinelRange(type, storage, firstElement, elementCount);

    SentinelResult result;
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

SentinelResult checkUnusedTensorStorage(const Tensor& logicalTensor, size_t allocatedElements,
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

    SentinelResult result;
    result.reportedMismatches.reserve(maxReportedMismatches);
    for (size_t index = 0; index < allocatedElements; ++index) {
        if (used[index]) continue;
        const SentinelResult element = checkUnwrittenSentinel(
            logicalTensor.type(), logicalTensor.rawEncodedBackingStorage(), index, 1, region,
            maxReportedMismatches - result.reportedMismatches.size());
        result.append(element, maxReportedMismatches);
    }
    return result;
}
}  // namespace roc::host_validation
