// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <roc/host_validation/detail/tensor_views.hpp>
#include <span>
#include <stdexcept>
#include <vector>

namespace roc::host_validation {
struct ComparisonOptions {
    double absoluteTolerance = 0.0;
    double relativeTolerance = 0.0;
    double symmetricRelativeTolerance = 0.0;
    size_t maxReportedMismatches = 10;
};

struct Mismatch {
    size_t index = 0;
    double observed = 0.0;
    double expected = 0.0;
    double absoluteDifference = 0.0;
};

struct ComparisonResult {
    size_t compared = 0;
    size_t mismatches = 0;
    double maxAbsoluteDifference = 0.0;
    std::vector<Mismatch> reportedMismatches;

    bool passed() const {
        return mismatches == 0;
    }
};

namespace detail {
inline void compareValue(ComparisonResult& result, size_t index, double observedValue,
                         double expectedValue, const ComparisonOptions& options) {
    const bool exactMatch = observedValue == expectedValue;
    const bool bothFinite = std::isfinite(observedValue) && std::isfinite(expectedValue);
    const double difference = exactMatch   ? 0.0
                              : bothFinite ? std::abs(observedValue - expectedValue)
                                           : std::numeric_limits<double>::infinity();
    const double tolerance = options.absoluteTolerance +
                             options.relativeTolerance * std::abs(expectedValue) +
                             options.symmetricRelativeTolerance *
                                 (std::abs(observedValue) + std::abs(expectedValue) + 1.0);

    result.maxAbsoluteDifference = std::max(result.maxAbsoluteDifference, difference);

    if (!exactMatch && (!bothFinite || !(difference <= tolerance))) {
        ++result.mismatches;
        if (result.reportedMismatches.size() < options.maxReportedMismatches) {
            result.reportedMismatches.push_back({index, observedValue, expectedValue, difference});
        }
    }
}
}  // namespace detail

template <typename Observed, typename Expected>
ComparisonResult compare(std::span<const Observed> observed, std::span<const Expected> expected,
                         const ComparisonOptions& options = {}) {
    if (observed.size() != expected.size())
        throw std::invalid_argument("Host validation comparison size mismatch.");

    ComparisonResult result;
    result.compared = observed.size();
    result.reportedMismatches.reserve(std::min(options.maxReportedMismatches, observed.size()));

    for (size_t i = 0; i < observed.size(); ++i) {
        detail::compareValue(result, i, static_cast<double>(observed[i]),
                             static_cast<double>(expected[i]), options);
    }

    return result;
}

template <typename Observed, typename Expected>
ComparisonResult compare(ConstMatrixView<Observed> observed, ConstMatrixView<Expected> expected,
                         const ComparisonOptions& options = {}) {
    if (observed.rows() != expected.rows() || observed.columns() != expected.columns())
        throw std::invalid_argument("Host validation matrix comparison shape mismatch.");

    ComparisonResult result;
    result.compared = observed.rows() * observed.columns();
    result.reportedMismatches.reserve(std::min(options.maxReportedMismatches, result.compared));

    for (size_t column = 0; column < observed.columns(); ++column) {
        for (size_t row = 0; row < observed.rows(); ++row) {
            const size_t logicalIndex = row + column * observed.rows();
            detail::compareValue(result, logicalIndex, static_cast<double>(observed(row, column)),
                                 static_cast<double>(expected(row, column)), options);
        }
    }

    return result;
}

inline ComparisonResult compare(TensorView observed, TensorView expected,
                                const ComparisonOptions& options = {}) {
    if (observed.shape() != expected.shape())
        throw std::invalid_argument("Host validation tensor comparison shape mismatch.");
    if (scalarTypeInfo(observed.type()).category == ScalarCategory::Complex ||
        scalarTypeInfo(expected.type()).category == ScalarCategory::Complex)
        throw std::invalid_argument(
            "Runtime tensor comparison does not yet support complex values.");

    ComparisonResult result;
    result.compared = observed.shape().elementCount();
    result.reportedMismatches.reserve(
        std::min(options.maxReportedMismatches, result.compared));

    std::vector<size_t> indices(observed.shape().rank(), 0);
    for (size_t linearIndex = 0; linearIndex < result.compared; ++linearIndex) {
        detail::compareValue(result, linearIndex, observed.loadAs<double>(indices),
                             expected.loadAs<double>(indices), options);
        for (size_t dimension = observed.shape().rank(); dimension > 0; --dimension) {
            const size_t index = dimension - 1;
            if (++indices[index] < observed.shape()[index]) break;
            indices[index] = 0;
        }
    }
    return result;
}
}  // namespace roc::host_validation
