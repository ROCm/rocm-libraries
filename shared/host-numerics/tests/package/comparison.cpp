// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <array>
#include <roc/host_numerics/comparison.hpp>
#include <span>

int main() {
    using namespace roc::host_numerics;

    const std::array<float, 3> observed{1.0f, 2.0f, 3.0f};
    const std::array<float, 3> expected{1.0f, 2.0f, 4.0f};
    const ComparisonResult result = compare(
        Tensor::copyNativeStorage(std::span<const float>(observed)),
        Tensor::copyNativeStorage(std::span<const float>(expected)), nearComparisonOptions(0.25));
    return !result.passed() && result.mismatches == 1 ? 0 : 1;
}
