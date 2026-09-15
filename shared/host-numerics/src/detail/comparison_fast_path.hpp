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
template <typename Tag>
ComparisonReport compareAllCloseOnlyKnown(const Tensor& observed, const Tensor& expected,
                                          const ComparisonOptions& options) {
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
                        const ComparisonValue observedValue = loadComparisonValueKnown<Tag>(
                            observed.rawEncodedBackingStorage(), observedOffset);
                        const ComparisonValue expectedValue = loadComparisonValueKnown<Tag>(
                            expected.rawEncodedBackingStorage(), expectedOffset);
                        if (options.complexComparisonMode == ComplexComparisonMode::Magnitude) {
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
            result.allClosePassed = result.mismatches == 0;
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
                    if (options.complexComparisonMode == ComplexComparisonMode::Magnitude) {
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
