// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <memory>
#include <roc/hostnumerics/layout.hpp>
#include <roc/hostnumerics/result.hpp>
#include <roc/hostnumerics/scalar_integer_conversion.hpp>
#include <roc/hostnumerics/tensor.hpp>

namespace {
using namespace roc::hostnumerics;
using namespace roc::hostnumerics::detail;

enum class TestFailure { Missing };

Expected<int, TestFailure> readResult(bool available) {
    if (!available) return Unexpected(TestFailure::Missing);
    return 17;
}
Expected<long, TestFailure> propagateResult(bool available) {
    auto result = readResult(available);
    if (!result) return Unexpected(result.error());
    return static_cast<long>(*result) + 1;
}

TEST(Result, PropagatesFailureAcrossValueTypes) {
    const auto success = propagateResult(true);
    ASSERT_TRUE(success.has_value());
    EXPECT_EQ(*success, 18);
    const auto failure = propagateResult(false);
    ASSERT_FALSE(failure.has_value());
    EXPECT_EQ(failure.error(), TestFailure::Missing);
}

TEST(Result, OwnsMoveOnlyValueAndPreservesItsIdentity) {
    auto owner = std::make_unique<int>(37);
    int* identity = owner.get();
    Expected<std::unique_ptr<int>, TestFailure> result(std::move(owner));
    auto moved = std::move(result);
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->get(), identity);
    auto extracted = *std::move(moved);
    EXPECT_EQ(extracted.get(), identity);
}

TEST(Result, DistinguishesValuesAndErrorsOfTheSameType) {
    constexpr Expected<int, int> success = 17;
    constexpr Expected<int, int> failure = Unexpected(17);
    static_assert(success.has_value() && *success == 17);
    static_assert(!failure.has_value() && failure.error() == 17);
    const Expected<int, int> defaultValue;
    ASSERT_TRUE(defaultValue.has_value());
    EXPECT_EQ(*defaultValue, 0);

    const Unexpected<int> error(23);
    const Expected<int, long> copiedError = error;
    ASSERT_FALSE(copiedError.has_value());
    EXPECT_EQ(copiedError.error(), 23);
    static_assert(std::same_as<decltype(*std::move(success)), const int&&>);
    static_assert(std::same_as<decltype(std::move(failure).error()), const int&&>);
    static_assert(!std::is_constructible_v<Expected<bool, int>, Expected<int, int>>);
}

TEST(Result, ConstructsValuesInPlaceAndMovesErrorOwnership) {
    struct Immovable {
        explicit Immovable(int input) : value(input) {}
        Immovable(const Immovable&) = delete;
        Immovable(Immovable&&) = delete;
        int value;
    };
    Expected<Immovable, TestFailure> success(std::in_place, 29);
    ASSERT_TRUE(success);
    EXPECT_EQ(success->value, 29);

    auto owner = std::make_unique<int>(37);
    int* identity = owner.get();
    Expected<int, std::unique_ptr<int>> failure = Unexpected(std::move(owner));
    auto moved = std::move(failure);
    ASSERT_FALSE(moved.has_value());
    auto extracted = std::move(moved).error();
    EXPECT_EQ(extracted.get(), identity);
}

TEST(Result, PayloadConstructionExceptionsPropagate) {
    struct PayloadFailure {};
    struct Payload {
        Payload() = default;
        Payload(const Payload&) {
            throw PayloadFailure{};
        }
    };
    const Payload payload;
    EXPECT_THROW((Expected<Payload, TestFailure>(payload)), PayloadFailure);
    static_assert(!std::is_nothrow_constructible_v<Expected<Payload, TestFailure>, const Payload&>);
    static_assert(std::is_nothrow_constructible_v<Expected<int, TestFailure>, int>);
    const Unexpected<Payload> error(std::in_place);
    EXPECT_THROW((Expected<int, Payload>(error)), PayloadFailure);
    static_assert(
        !std::is_nothrow_constructible_v<Expected<int, Payload>, const Unexpected<Payload>&>);
}

TEST(Result, ConversionFailuresCarryDestinationContext) {
    const auto result = tryConvertToInteger<int8_t>(300.0, ScalarConversionOptions{});
    ASSERT_FALSE(result.has_value());
    EXPECT_EQ(result.error().code, ConversionError::RoundedOutOfRange);
    EXPECT_EQ(result.error().destinationBits, 8);
    EXPECT_TRUE(result.error().signedDestination);
    const auto nonreal = tryConvertToInteger<uint32_t>(std::complex<double>(1, 2), {});
    ASSERT_FALSE(nonreal.has_value());
    EXPECT_EQ(nonreal.error().code, ConversionError::Nonreal);
    EXPECT_EQ(nonreal.error().destinationBits, 32);
    EXPECT_FALSE(nonreal.error().signedDestination);
}

TEST(Result, GeometryFailuresPropagateWithoutFormatting) {
    const auto contiguous = tryContiguousLayout(Shape{std::numeric_limits<size_t>::max()}, false);
    ASSERT_FALSE(contiguous.has_value());
    EXPECT_EQ(contiguous.error(), LayoutError::ExtentOutOfRange);
    const Layout overflowing(Shape{3}, {std::numeric_limits<ptrdiff_t>::max()});
    const auto bounds = tryElementBounds(overflowing);
    ASSERT_FALSE(bounds.has_value());
    EXPECT_EQ(bounds.error(), LayoutError::OffsetMultiplyOverflow);
    const auto storage = tryStorageBytesForLayout(ScalarType::Float32, overflowing);
    ASSERT_FALSE(storage.has_value());
    ASSERT_TRUE(std::holds_alternative<LayoutError>(storage.error().cause));
    EXPECT_EQ(std::get<LayoutError>(storage.error().cause), bounds.error());

    const Layout reversed(Shape{3}, {-1}, 2);
    const auto reversedBounds = tryElementBounds(reversed);
    ASSERT_TRUE(reversedBounds);
    EXPECT_EQ(reversedBounds->first, 0);
    EXPECT_EQ(reversedBounds->second, 2);
    const auto bytes = tryStorageBytesForLayout(ScalarType::Float32, reversed);
    ASSERT_TRUE(bytes);
    EXPECT_EQ(*bytes, 12);
}
}  // namespace
