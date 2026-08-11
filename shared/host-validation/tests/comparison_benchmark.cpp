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
#include <roc/host_validation/typed_comparison.hpp>
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

    ComparisonOptions statisticsOptions = defaultComparisonOptions(ScalarType::Float32);
    statisticsOptions.computeFrobenius = false;
    statisticsOptions.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    ComparisonResult statisticsReport;
    const double statisticsComponentSeconds = bestSeconds(
        [&] { statisticsReport = compare(observedView, expectedView, statisticsOptions); });
    if (!statisticsReport.passed() || statisticsReport.compared != elements) return 1;

    ComparisonResult typedStatisticsReport;
    const double typedStatisticsComponentSeconds = bestSeconds([&] {
        typedStatisticsReport = compare(typedObservedView, typedExpectedView, statisticsOptions);
    });
    if (!typedStatisticsReport.passed() || typedStatisticsReport.compared != elements) return 1;

    ComparisonOptions detailedOptions = defaultComparisonOptions(ScalarType::Float32);
    detailedOptions.selection.indexOrder = ComparisonIndexOrder::FirstDimensionFastest;
    ComparisonResult detailedReport;
    const double detailedComponentSeconds =
        bestSeconds([&] { detailedReport = compare(observedView, expectedView, detailedOptions); });
    if (!detailedReport.passed() || detailedReport.compared != elements) return 1;

    ComparisonResult typedDetailedReport;
    const double typedDetailedComponentSeconds = bestSeconds([&] {
        typedDetailedReport = compare(typedObservedView, typedExpectedView, detailedOptions);
    });
    if (!typedDetailedReport.passed() || typedDetailedReport.compared != elements) return 1;

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
              << " statistics_component_seconds=" << statisticsComponentSeconds
              << " typed_statistics_component_seconds=" << typedStatisticsComponentSeconds
              << " typed_statistics_over_runtime="
              << typedStatisticsComponentSeconds / statisticsComponentSeconds
              << " detailed_component_seconds=" << detailedComponentSeconds
              << " typed_detailed_component_seconds=" << typedDetailedComponentSeconds
              << " typed_detailed_over_runtime="
              << typedDetailedComponentSeconds / detailedComponentSeconds << std::endl;
    return 0;
}
