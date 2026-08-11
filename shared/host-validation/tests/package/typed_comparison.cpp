// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <concepts>
#include <roc/host_validation/typed_comparison.hpp>
#include <span>

namespace {
struct ExternalFloat {
    float value;

    operator float() const {
        return value;
    }
};

template <typename T>
concept HasTypedComparison = requires(const roc::host_validation::TypedTensorView<T>& view,
                                      const roc::host_validation::ComparisonOptions& options) {
    {
        roc::host_validation::compare(view, view, options)
    } -> std::same_as<roc::host_validation::ComparisonResult>;
};

static_assert(HasTypedComparison<ExternalFloat>);
}  // namespace

int main() {
    using namespace roc::host_validation;

    const std::array<ExternalFloat, 2> observed{ExternalFloat{1.0f}, ExternalFloat{2.0f}};
    const std::array<ExternalFloat, 2> expected{ExternalFloat{1.0f}, ExternalFloat{3.0f}};
    ComparisonOptions options;
    options.computePointwiseStatistics = false;
    options.computeFrobenius = false;
    options.maxReportedMismatches = 1;
    const ComparisonResult result =
        compare(TypedTensorView<ExternalFloat>(std::span<const ExternalFloat>(observed)),
                TypedTensorView<ExternalFloat>(std::span<const ExternalFloat>(expected)), options);
    return !result.passed() && result.mismatches == 1 && result.reportedMismatches.size() == 1 ? 0
                                                                                               : 1;
}
