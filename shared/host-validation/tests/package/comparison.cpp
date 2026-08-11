// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <concepts>
#include <roc/host_validation/comparison.hpp>
#include <span>

template <typename T>
concept HasTypedComparison = requires(const roc::host_validation::TypedTensorView<T>& view,
                                      const roc::host_validation::ComparisonOptions& options) {
    {
        roc::host_validation::compare(view, view, options)
    } -> std::same_as<roc::host_validation::ComparisonResult>;
};

static_assert(!HasTypedComparison<float>);

int main() {
    using namespace roc::host_validation;

    const std::array<float, 3> observed{1.0f, 2.0f, 3.0f};
    const std::array<float, 3> expected{1.0f, 2.0f, 4.0f};
    const ComparisonResult result = compare(
        TensorView::fromNative(std::span<const float>(observed)),
        TensorView::fromNative(std::span<const float>(expected)), nearComparisonOptions(0.25));
    return !result.passed() && result.mismatches == 1 ? 0 : 1;
}
