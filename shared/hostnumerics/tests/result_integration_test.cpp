// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/tensor.hpp>

namespace {
using namespace roc::hostnumerics;
using namespace roc::hostnumerics::detail;

TEST(ResultIntegration, GeometryFailuresPropagateWithoutFormatting) {
    const auto contiguous =
        Layout::getContiguousLayoutLastDimensionFastest(Shape{std::numeric_limits<size_t>::max()});
    ASSERT_FALSE(contiguous.has_value());
    EXPECT_EQ(contiguous.error().error, LayoutError::ExtentOutOfRange);
    const Layout overflowing(Shape{3}, {std::numeric_limits<ptrdiff_t>::max()}, 0);
    const auto bounds = getElementBounds(overflowing);
    ASSERT_FALSE(bounds.has_value());
    EXPECT_EQ(bounds.error().error, LayoutError::OffsetMultiplyOverflow);
    const auto storage = getStorageBytesForLayout(ScalarType::Float32, overflowing);
    ASSERT_FALSE(storage.has_value());
    ASSERT_TRUE(std::holds_alternative<LayoutFailure>(storage.error().cause));
    const auto& cause = std::get<LayoutFailure>(storage.error().cause);
    EXPECT_EQ(cause.error, bounds.error().error);
    EXPECT_EQ(cause.dimension, bounds.error().dimension);

    const Layout reversed(Shape{3}, {-1}, 2);
    const auto reversedBounds = getElementBounds(reversed);
    ASSERT_TRUE(reversedBounds);
    EXPECT_EQ(reversedBounds->first, 0);
    EXPECT_EQ(reversedBounds->second, 2);
    const auto bytes = getStorageBytesForLayout(ScalarType::Float32, reversed);
    ASSERT_TRUE(bytes);
    EXPECT_EQ(*bytes, 12);
}
}  // namespace
