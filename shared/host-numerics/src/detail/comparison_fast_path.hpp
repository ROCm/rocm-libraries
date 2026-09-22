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
#include <functional>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "comparison_common.hpp"
#include "comparison_iteration.hpp"
#include "threading.hpp"

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

inline std::optional<ComparisonReport> exactEncodedComparisonReport(
    const Tensor& observed, const Tensor& expected, const ComparisonOptions& options) {
    // Exact encoded equality determines all-close and zero-distance ULP evidence without decoding.
    // Statistics, norms, and matching-element reports still require visiting the logical values.
    if (options.computeElementwiseStatistics || options.computeFrobenius ||
        options.reportMatchingElements || !encodedValuesAreIdentical(observed, expected, options))
        return std::nullopt;

    ComparisonReport result;
    result.compared = observed.elementCount();
    result.allCloseEvaluated = options.allClose;
    result.ulpEvaluated = options.maximumUlpTolerance.has_value();
    if (options.computeUlp) {
        const size_t componentsPerElement =
            scalarTypeInfo(observed.type()).category == ScalarCategory::Complex ? 2 : 1;
        result.ulpCompared = result.compared * componentsPerElement;
    }
    return result;
}

template <typename Tag>
ComparisonReport compareAllCloseOnlyKnown(const Tensor& observed, const Tensor& expected,
                                          const ComparisonOptions& options) {
    const auto observedStorage = observed.rawEncodedBackingStorage();
    const auto expectedStorage = expected.rawEncodedBackingStorage();
    const auto run = [&]<typename Predicate>(Predicate predicate) {
        ComparisonReport result;
        result.allCloseEvaluated = true;
        const bool selectsContiguousPrefix =
            options.selection.kind() == OutputSelectionKind::Strided &&
            options.selection.first() == 0 && options.selection.stride() == 1;
        if ((options.selection.selectsAll() || selectsContiguousPrefix) &&
            observed.shape().rank() != 0) {
            const Shape& shape = observed.shape();
            const size_t selectedTotal =
                options.selection.selectsAll()
                    ? shape.elementCount()
                    : std::min(shape.elementCount(), options.selection.maxElements());
            if (selectedTotal == 0) return result;
            constexpr size_t minimumElementsPerThread = 500'000;
            const size_t partitionCount =
                static_cast<size_t>(operationThreadCount(selectedTotal, minimumElementsPerThread));
            result.mismatches = transformReduceParallelIndices(
                partitionCount, selectedTotal, true, minimumElementsPerThread, size_t{0},
                [&](size_t partition) {
                    size_t mismatches = 0;
                    const auto [first, pastLast] =
                        evenlyPartitionedRange(selectedTotal, partitionCount, partition);
                    forEachLayoutOffsetPairRange(
                        observed.layout(), expected.layout(), options.selection.indexOrder(), first,
                        pastLast, [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                            bool close = false;
                            if constexpr (scalarTypeInfo(Tag::type).category ==
                                          ScalarCategory::Complex) {
                                const ComparisonValue observedValue =
                                    loadComparisonValueKnown<Tag>(observedStorage, observedOffset);
                                const ComparisonValue expectedValue =
                                    loadComparisonValueKnown<Tag>(expectedStorage, expectedOffset);
                                if (options.complexComparisonMode ==
                                    ComplexComparisonMode::Magnitude) {
                                    close = compareComplexMagnitude(observedValue, expectedValue,
                                                                    options)
                                                .close;
                                } else {
                                    close = predicate(observedValue.real, expectedValue.real);
                                    close = close && predicate(observedValue.imaginary,
                                                               expectedValue.imaginary);
                                }
                            } else {
                                close = predicate(
                                    loadFastComparisonReal<Tag>(observedStorage, observedOffset),
                                    loadFastComparisonReal<Tag>(expectedStorage, expectedOffset));
                            }
                            mismatches += static_cast<size_t>(!close);
                        });
                    return mismatches;
                },
                std::plus<>{});
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
