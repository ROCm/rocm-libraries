// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <roc/hostnumerics/tensor.hpp>

namespace {
using namespace roc::hostnumerics;
using namespace roc::hostnumerics::detail;

TEST(ResultIntegration, ConversionFailuresCarryDestinationContext) {
    const auto result = getConvertToInteger<int8_t>(300.0, ScalarConversionOptions{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ConversionError::RoundedOutOfRange);
    EXPECT_EQ(result.error().destinationBits, 8);
    EXPECT_TRUE(result.error().signedDestination);
    const auto nonreal = getConvertToInteger<uint32_t>(std::complex<double>(1, 2), {});
    ASSERT_FALSE(nonreal.has_value());
    EXPECT_EQ(nonreal.error().code, ConversionError::Nonreal);
    EXPECT_EQ(nonreal.error().destinationBits, 32);
    EXPECT_FALSE(nonreal.error().signedDestination);
}

TEST(ResultIntegration, GeometryFailuresPropagateWithoutFormatting) {
    const auto contiguous = Layout::getContiguousLayout(Shape{std::numeric_limits<size_t>::max()});
    ASSERT_FALSE(contiguous.has_value());
    EXPECT_EQ(contiguous.error(), LayoutError::ExtentOutOfRange);
    const Layout overflowing(Shape{3}, {std::numeric_limits<ptrdiff_t>::max()}, 0);
    const auto bounds = getElementBounds(overflowing);
    ASSERT_FALSE(bounds.has_value());
    EXPECT_EQ(bounds.error(), LayoutError::OffsetMultiplyOverflow);
    const auto storage = getStorageBytesForLayout(ScalarType::Float32, overflowing);
    ASSERT_FALSE(storage.has_value());
    ASSERT_TRUE(std::holds_alternative<LayoutError>(storage.error().cause));
    EXPECT_EQ(std::get<LayoutError>(storage.error().cause), bounds.error());

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
