// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <roc/host_validation/comparison.hpp>
#include <span>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

template <typename Function>
double bestSeconds(Function&& function) {
    double best = std::numeric_limits<double>::infinity();
    for (int iteration = 0; iteration < 4; ++iteration) {
        const auto start = Clock::now();
        function();
        const auto end = Clock::now();
        best = std::min(best, std::chrono::duration<double>(end - start).count());
    }
    return best;
}
}  // namespace

int main(int argc, char** argv) {
    using namespace roc::host_validation;

    const size_t rows = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 4096;
    const size_t columns = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 4096;
    const size_t leadingDimension = rows + 16;
    const size_t elements = rows * columns;
    const size_t storageElements = leadingDimension * columns;
    std::vector<float> expected(storageElements);
    std::vector<float> observed(storageElements);
    for (size_t column = 0; column < columns; ++column) {
        for (size_t row = 0; row < rows; ++row) {
            const size_t logicalIndex = row + column * rows;
            const size_t storageIndex = row + column * leadingDimension;
            expected[storageIndex] = static_cast<float>(static_cast<int>(logicalIndex % 101) - 50);
            observed[storageIndex] = expected[storageIndex];
        }
    }

    const Layout layout(Shape{rows, columns}, {1, static_cast<ptrdiff_t>(leadingDimension)});
    const TensorView expectedView =
        TensorView::fromNative<float>(layout, std::span<const float>(expected));
    const TensorView observedView =
        TensorView::fromNative<float>(layout, std::span<const float>(observed));
    const TypedTensorView<float> typedExpectedView(layout, std::span<const float>(expected));
    const TypedTensorView<float> typedObservedView(layout, std::span<const float>(observed));
    ComparisonOptions options = defaultComparisonOptions(ScalarType::Float32);
    options.computePointwiseStatistics = false;
    options.computeFrobenius = false;
    options.maxReportedMismatches = 0;
    options.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;

    ComparisonResult report;
    const double componentSeconds =
        bestSeconds([&] { report = compare(observedView, expectedView, options); });
    if (!report.passed() || report.compared != elements) return 1;

    ComparisonResult typedReport;
    const double typedComponentSeconds =
        bestSeconds([&] { typedReport = compare(typedObservedView, typedExpectedView, options); });
    if (!typedReport.passed() || typedReport.compared != elements) return 1;

    size_t offsetTraversalMismatches = 0;
    const double offsetTraversalSeconds = bestSeconds([&] {
        size_t mismatches = 0;
        detail::forEachSelectedOffsetPair(
            layout, layout, options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                mismatches += observed[observedOffset] != expected[expectedOffset];
            });
        offsetTraversalMismatches = mismatches;
    });
    if (offsetTraversalMismatches != 0) return 1;

    size_t decodeTraversalMismatches = 0;
    const double decodeTraversalSeconds = bestSeconds([&] {
        size_t mismatches = 0;
        detail::forEachSelectedOffsetPair(
            layout, layout, options.selection,
            [&](size_t, ptrdiff_t observedOffset, ptrdiff_t expectedOffset) {
                const auto a = detail::loadComparisonValueKnown<detail::Float32Tag>(
                    observedView.storage(), observedOffset);
                const auto b = detail::loadComparisonValueKnown<detail::Float32Tag>(
                    expectedView.storage(), expectedOffset);
                mismatches += a.real != b.real;
            });
        decodeTraversalMismatches = mismatches;
    });
    if (decodeTraversalMismatches != 0) return 1;

    size_t baselineMismatches = 0;
    const double tolerance = defaultSymmetricRelativeTolerance(ScalarType::Float32);
    const double baselineSeconds = bestSeconds([&] {
        size_t mismatches = 0;
        for (size_t column = 0; column < columns; ++column) {
            for (size_t row = 0; row < rows; ++row) {
                const size_t index = row + column * leadingDimension;
                const double a = observed[index];
                const double b = expected[index];
                const double difference = std::abs(a - b);
                mismatches +=
                    !(a == b || difference < tolerance * (std::abs(a) + std::abs(b) + 1.0));
            }
        }
        baselineMismatches = mismatches;
    });
    if (baselineMismatches != 0) return 1;

    const double bytes = 2.0 * elements * sizeof(float);
    std::cout << "component_seconds=" << componentSeconds
              << " component_GBps=" << bytes / componentSeconds / 1e9
              << " baseline_seconds=" << baselineSeconds
              << " baseline_GBps=" << bytes / baselineSeconds / 1e9
              << " component_over_baseline=" << componentSeconds / baselineSeconds
              << " typed_component_seconds=" << typedComponentSeconds
              << " typed_component_over_baseline=" << typedComponentSeconds / baselineSeconds
              << " offset_traversal_seconds=" << offsetTraversalSeconds
              << " decode_traversal_seconds=" << decodeTraversalSeconds << std::endl;
    return 0;
}
