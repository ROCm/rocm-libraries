// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

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

#include "comparison_common.hpp"
#include "comparison_iteration.hpp"

namespace roc::host_numerics::detail {
inline bool encodedBitRangesAreIdentical(std::span<const std::byte> observed,
                                         std::span<const std::byte> expected, uint64_t firstBit,
                                         uint64_t comparedBits) {
    if (firstBit > std::numeric_limits<uint64_t>::max() - comparedBits) return false;
    const uint64_t pastLastBit = firstBit + comparedBits;
    const uint64_t requiredBytes = pastLastBit / 8 + static_cast<uint64_t>(pastLastBit % 8 != 0);
    if (requiredBytes > observed.size() || requiredBytes > expected.size()) return false;

    const auto bitsEqual = [&](uint64_t bit) {
        const size_t byte = static_cast<size_t>(bit / 8);
        const unsigned mask = 1U << (bit % 8);
        return (std::to_integer<unsigned>(observed[byte]) & mask) ==
               (std::to_integer<unsigned>(expected[byte]) & mask);
    };
    while (comparedBits != 0 && firstBit % 8 != 0) {
        if (!bitsEqual(firstBit)) return false;
        ++firstBit;
        --comparedBits;
    }

    const size_t firstByte = static_cast<size_t>(firstBit / 8);
    const size_t comparedBytes = static_cast<size_t>(comparedBits / 8);
    if (comparedBytes != 0 &&
        std::memcmp(observed.data() + firstByte, expected.data() + firstByte, comparedBytes) != 0)
        return false;
    firstBit += static_cast<uint64_t>(comparedBytes) * 8;
    comparedBits -= static_cast<uint64_t>(comparedBytes) * 8;
    while (comparedBits != 0) {
        if (!bitsEqual(firstBit)) return false;
        ++firstBit;
        --comparedBits;
    }
    return true;
}

inline bool encodedValuesAreIdentical(const Tensor& observed, const Tensor& expected,
                                      const ComparisonOptions& options) {
    // This is only an early success test. A byte mismatch falls through to the numerical
    // comparison because different encodings can still compare equal (for example, signed zero).
    if (!options.selection.selectsAll() || observed.type() != expected.type() ||
        observed.layout() != expected.layout() ||
        (!options.equalNaNs && scalarTypeInfo(observed.type()).supportsNaN))
        return false;

    const size_t elementCount = observed.elementCount();
    if (elementCount == 0) return true;
    const uint64_t storageBits = scalarTypeInfo(observed.type()).storageBits;
    const auto observedStorage = observed.rawEncodedBackingStorage();
    const auto expectedStorage = expected.rawEncodedBackingStorage();

    if (hasProvablyDistinctElementOffsets(observed.layout())) {
        const auto [firstElement, lastElement] = elementBounds(observed.layout());
        if (firstElement >= 0 && lastElement >= firstElement &&
            static_cast<uintmax_t>(lastElement - firstElement) + 1 == elementCount &&
            static_cast<uint64_t>(firstElement) <=
                std::numeric_limits<uint64_t>::max() / storageBits &&
            static_cast<uint64_t>(elementCount) <=
                std::numeric_limits<uint64_t>::max() / storageBits)
            return encodedBitRangesAreIdentical(observedStorage, expectedStorage,
                                                static_cast<uint64_t>(firstElement) * storageBits,
                                                static_cast<uint64_t>(elementCount) * storageBits);
    }

    const Shape& shape = observed.shape();
    size_t contiguousDimension = shape.rank();
    for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
        if ((observed.layout().stride(dimension) == 1 ||
             observed.layout().stride(dimension) == -1) &&
            (contiguousDimension == shape.rank() || shape[dimension] > shape[contiguousDimension]))
            contiguousDimension = dimension;
    }
    if (contiguousDimension == shape.rank()) return false;

    const size_t contiguousElements = shape[contiguousDimension];
    const size_t blockCount = elementCount / contiguousElements;
    std::vector<size_t> coordinates(shape.rank(), 0);
    for (size_t block = 0; block < blockCount; ++block) {
        size_t remaining = block;
        for (size_t dimension = 0; dimension < shape.rank(); ++dimension) {
            if (dimension == contiguousDimension) continue;
            coordinates[dimension] = remaining % shape[dimension];
            remaining /= shape[dimension];
        }
        ptrdiff_t firstElement = observed.layout().elementOffset(coordinates);
        if (observed.layout().stride(contiguousDimension) < 0)
            firstElement -= static_cast<ptrdiff_t>(contiguousElements - 1);
        if (firstElement < 0 ||
            static_cast<uint64_t>(firstElement) >
                std::numeric_limits<uint64_t>::max() / storageBits ||
            static_cast<uint64_t>(contiguousElements) >
                std::numeric_limits<uint64_t>::max() / storageBits ||
            !encodedBitRangesAreIdentical(observedStorage, expectedStorage,
                                          static_cast<uint64_t>(firstElement) * storageBits,
                                          static_cast<uint64_t>(contiguousElements) * storageBits))
            return false;
    }
    return true;
}

template <typename Tag>
ComparisonReport compareAllCloseOnlyKnown(const Tensor& observed, const Tensor& expected,
                                          const ComparisonOptions& options) {
    if (encodedValuesAreIdentical(observed, expected, options)) {
        ComparisonReport result;
        result.compared = observed.elementCount();
        result.allCloseEvaluated = true;
        return result;
    }

    const auto observedStorage = observed.rawEncodedBackingStorage();
    const auto expectedStorage = expected.rawEncodedBackingStorage();
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonReport result;
        result.allCloseEvaluated = true;
        if ((options.selection.selectsAll() ||
             (options.selection.first() == 0 && options.selection.stride() == 1)) &&
            options.selection.indexOrder() == IndexOrder::FirstDimensionFastest &&
            observed.shape().rank() != 0) {
            const Shape& shape = observed.shape();
            const size_t innerSize = shape[0];
            const size_t selectedTotal =
                options.selection.selectsAll()
                    ? shape.elementCount()
                    : std::min(shape.elementCount(), options.selection.maxElements());
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
                        const ComparisonValue observedValue =
                            loadComparisonValueKnown<Tag>(observedStorage, observedOffset);
                        const ComparisonValue expectedValue =
                            loadComparisonValueKnown<Tag>(expectedStorage, expectedOffset);
                        if (options.complexComparisonMode == ComplexComparisonMode::Magnitude) {
                            close = compareComplexMagnitude(observedValue, expectedValue, options)
                                        .close;
                        } else {
                            close = predicate(observedValue.real, expectedValue.real);
                            close = close &&
                                    predicate(observedValue.imaginary, expectedValue.imaginary);
                        }
                    } else {
                        close =
                            predicate(loadFastComparisonReal<Tag>(observedStorage, observedOffset),
                                      loadFastComparisonReal<Tag>(expectedStorage, expectedOffset));
                    }
                    result.mismatches += static_cast<size_t>(!close);
                }
            }
            result.compared = selectedTotal;
            result.allClosePassed = result.mismatches == 0;
            return result;
        }

        forEachSelectedOffsetPair(
            observed.layout(), expected.layout(), options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                ++result.compared;
                bool close = false;
                if constexpr (scalarTypeInfo(Tag::type).category == ScalarCategory::Complex) {
                    const ComparisonValue observedValue =
                        loadComparisonValueKnown<Tag>(observedStorage, observedOffset);
                    const ComparisonValue expectedValue =
                        loadComparisonValueKnown<Tag>(expectedStorage, expectedOffset);
                    if (options.complexComparisonMode == ComplexComparisonMode::Magnitude) {
                        close =
                            compareComplexMagnitude(observedValue, expectedValue, options).close;
                    } else {
                        close = predicate(observedValue.real, expectedValue.real);
                        close =
                            close && predicate(observedValue.imaginary, expectedValue.imaginary);
                    }
                } else {
                    close = predicate(loadFastComparisonReal<Tag>(observedStorage, observedOffset),
                                      loadFastComparisonReal<Tag>(expectedStorage, expectedOffset));
                }
                result.mismatches += static_cast<size_t>(!close);
            });
        result.allClosePassed = result.mismatches == 0;
        return result;
    };

    if (!options.equalNaNs && options.absoluteTolerance == 0.0 && options.relativeTolerance == 0.0)
        return run(
            [](auto observedValue, auto expectedValue) { return observedValue == expectedValue; });

    return run([&options](auto observedValue, auto expectedValue) {
        return valuesCloseFast(observedValue, expectedValue, options);
    });
}
}  // namespace roc::host_numerics::detail
