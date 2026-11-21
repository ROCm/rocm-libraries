// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <array>
#include <roc/hostnumerics/comparison.hpp>
#include <span>

TEST(Package, Comparison) {
    using namespace roc::hostnumerics;

    const std::array<float, 3> observed{1.0f, 2.0f, 3.0f};
    const std::array<float, 3> expected{1.0f, 2.0f, 4.0f};
    const ComparisonReport result = compare(
        Tensor::copyNativeStorage(std::span<const float>(observed)),
        Tensor::copyNativeStorage(std::span<const float>(expected)), nearComparisonOptions(0.25));
    EXPECT_FALSE(result.passed());
    EXPECT_EQ(result.mismatches, 1);
}
