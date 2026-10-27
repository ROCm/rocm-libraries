// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#if defined(HOSTNUMERICS_TEST_STD_EXPECTED)
#include <expected>
#else
#include <roc/hostnumerics/expected.hpp>
#endif

namespace {
#if defined(HOSTNUMERICS_TEST_STD_EXPECTED)
template <typename T, typename E>
using Expected = std::expected<T, E>;
template <typename E>
using Unexpected = std::unexpected<E>;
#else
using roc::hostnumerics::detail::Expected;
using roc::hostnumerics::detail::Unexpected;
#endif

enum class TestFailure { Missing };

Expected<int, TestFailure> readExpected(bool available) {
    if (!available) return Unexpected<TestFailure>(TestFailure::Missing);
    return 17;
}
Expected<long, TestFailure> propagateExpected(bool available) {
    auto result = readExpected(available);
    if (!result) return Unexpected<TestFailure>(result.error());
    return static_cast<long>(*result) + 1;
}

TEST(ExpectedTest, PropagatesFailureAcrossValueTypes) {
    const auto success = propagateExpected(true);
    ASSERT_TRUE(success.has_value());
    EXPECT_EQ(*success, 18);
    const auto failure = propagateExpected(false);
    ASSERT_FALSE(failure.has_value());
    EXPECT_EQ(failure.error(), TestFailure::Missing);
}

TEST(ExpectedTest, OwnsMoveOnlyValueAndPreservesItsIdentity) {
    auto owner = std::make_unique<int>(37);
    int* identity = owner.get();
    Expected<std::unique_ptr<int>, TestFailure> result(std::move(owner));
    auto moved = std::move(result);
    ASSERT_TRUE(moved);
    EXPECT_EQ(moved->get(), identity);
    auto extracted = *std::move(moved);
    EXPECT_EQ(extracted.get(), identity);
}

TEST(ExpectedTest, DistinguishesValuesAndErrorsOfTheSameType) {
    constexpr Expected<int, int> success = 17;
    constexpr Expected<int, int> failure = Unexpected<int>(17);
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
#if !defined(HOSTNUMERICS_TEST_STD_EXPECTED)
    // This subset requires explicit propagation between different result types.
    static_assert(!std::is_constructible_v<Expected<bool, int>, Expected<int, int>>);
#endif
}

TEST(ExpectedTest, ConstructsValuesInPlaceAndMovesErrorOwnership) {
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
    Expected<int, std::unique_ptr<int>> failure =
        Unexpected<std::unique_ptr<int>>(std::move(owner));
    auto moved = std::move(failure);
    ASSERT_FALSE(moved.has_value());
    auto extracted = std::move(moved).error();
    EXPECT_EQ(extracted.get(), identity);
}

TEST(ExpectedTest, PayloadConstructionExceptionsPropagate) {
    struct PayloadFailure {};
    struct Payload {
        Payload() = default;
        Payload(const Payload&) {
            throw PayloadFailure{};
        }
    };
    const Payload payload;
    EXPECT_THROW(static_cast<void>(Expected<Payload, TestFailure>(payload)), PayloadFailure);
    static_assert(!std::is_nothrow_constructible_v<Expected<Payload, TestFailure>, const Payload&>);
    static_assert(std::is_nothrow_constructible_v<Expected<int, TestFailure>, int>);
    const Unexpected<Payload> error(std::in_place);
    EXPECT_THROW(static_cast<void>(Expected<int, Payload>(error)), PayloadFailure);
    static_assert(
        !std::is_nothrow_constructible_v<Expected<int, Payload>, const Unexpected<Payload>&>);
}

TEST(ExpectedTest, ThrowingValueConversionRetainsItsException) {
    struct PayloadFailure {};
    struct Payload {
        explicit Payload(int) {
            throw PayloadFailure{};
        }
    };
    static_assert(!std::is_nothrow_constructible_v<Expected<Payload, int>, int>);
    EXPECT_THROW(static_cast<void>(Expected<Payload, int>(7)), PayloadFailure);
}

TEST(ExpectedTest, CopyAndMoveKeepOnlyTheActivePayloadAlive) {
    struct Payload {
        explicit Payload(int& count) : alive(&count) {
            ++*alive;
        }
        Payload(const Payload& other) : alive(other.alive) {
            ++*alive;
        }
        Payload(Payload&& other) noexcept : alive(std::exchange(other.alive, nullptr)) {}
        ~Payload() {
            if (alive) --*alive;
        }
        int* alive;
    };
    struct Source {
        operator Payload() {
            return Payload(alive);
        }
        int alive = 0;
    };
#if !defined(HOSTNUMERICS_TEST_STD_EXPECTED)
    // This subset requires explicit propagation between different result types.
    static_assert(!std::is_constructible_v<Expected<Payload, int>, Expected<Source, int>&&>);
#endif
    static_assert(!std::is_copy_assignable_v<Expected<Payload, int>>);
    static_assert(!std::is_move_assignable_v<Expected<Payload, int>>);
    int valuesAlive = 0;
    int errorsAlive = 0;
    {
        Expected<Payload, Payload> value(std::in_place, valuesAlive);
        Expected<Payload, Payload> error = Unexpected<Payload>(std::in_place, errorsAlive);
        auto copy = value;
        auto moved = std::move(error);
        EXPECT_EQ(valuesAlive, 2);
        EXPECT_EQ(errorsAlive, 1);
        ASSERT_TRUE(copy);
        ASSERT_FALSE(moved);
        EXPECT_EQ(copy->alive, &valuesAlive);
        EXPECT_EQ(moved.error().alive, &errorsAlive);
    }
    EXPECT_EQ(valuesAlive, 0);
    EXPECT_EQ(errorsAlive, 0);
}

TEST(ExpectedTest, AssignmentSupportsFallbackAndChangingBothAlternatives) {
    auto result = readExpected(false);
    if (!result) result = readExpected(true);
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 17);
    const auto failure = readExpected(false);
    result = failure;
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), TestFailure::Missing);
    const auto success = readExpected(true);
    result = success;
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 17);
    result = readExpected(false);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error(), TestFailure::Missing);
    static_assert(std::is_nothrow_move_assignable_v<Expected<int, TestFailure>>);
}

TEST(ExpectedTest, MoveAssignmentTransfersOwnershipBetweenAlternatives) {
    using Owner = std::unique_ptr<int>;
    Expected<Owner, Owner> result = Unexpected<Owner>(std::make_unique<int>(23));
    Expected<Owner, Owner> success = std::make_unique<int>(37);
    auto* valueIdentity = success->get();
    result = std::move(success);
    ASSERT_TRUE(result);
    EXPECT_EQ(result->get(), valueIdentity);
    Expected<Owner, Owner> failure = Unexpected<Owner>(std::make_unique<int>(41));
    auto* errorIdentity = failure.error().get();
    result = std::move(failure);
    ASSERT_FALSE(result);
    EXPECT_EQ(result.error().get(), errorIdentity);
    static_assert(!std::is_copy_assignable_v<Expected<Owner, Owner>>);
    static_assert(std::is_nothrow_move_assignable_v<Expected<Owner, Owner>>);
}

TEST(ExpectedTest, ThrowingCopyAssignmentPreservesAnActivePayload) {
    struct PayloadFailure {};
    struct Payload {
        Payload() = default;
        Payload(const Payload&) {
            throw PayloadFailure{};
        }
        Payload(Payload&&) noexcept = default;
        Payload& operator=(const Payload&) {
            throw PayloadFailure{};
        }
        Payload& operator=(Payload&&) noexcept = default;
    };
    Expected<Payload, int> valueDestination = Unexpected<int>(23);
    const Expected<Payload, int> valueSource(std::in_place);
    EXPECT_THROW(valueDestination = valueSource, PayloadFailure);
    ASSERT_FALSE(valueDestination);
    EXPECT_EQ(valueDestination.error(), 23);
    Expected<int, Payload> errorDestination = 37;
    const Expected<int, Payload> errorSource = Unexpected<Payload>(std::in_place);
    EXPECT_THROW(errorDestination = errorSource, PayloadFailure);
    ASSERT_TRUE(errorDestination);
    EXPECT_EQ(*errorDestination, 37);
    Expected<Payload, int> sameValue(std::in_place);
    EXPECT_THROW(sameValue = valueSource, PayloadFailure);
    EXPECT_TRUE(sameValue.has_value());
    Expected<int, Payload> sameError = Unexpected<Payload>(std::in_place);
    EXPECT_THROW(sameError = errorSource, PayloadFailure);
    EXPECT_FALSE(sameError.has_value());
}

#if !defined(HOSTNUMERICS_TEST_STD_EXPECTED)
TEST(ExpectedTest, RejectsAssignmentWhenEitherMoveConstructorCanThrow) {
    struct Payload {
        Payload() = default;
        Payload(const Payload&) = default;
        Payload(Payload&&) noexcept(false) {}
        Payload& operator=(const Payload&) = default;
        Payload& operator=(Payload&&) = default;
    };
    static_assert(!std::is_copy_assignable_v<Expected<Payload, int>>);
    static_assert(!std::is_move_assignable_v<Expected<Payload, int>>);
    static_assert(!std::is_copy_assignable_v<Expected<int, Payload>>);
    static_assert(!std::is_move_assignable_v<Expected<int, Payload>>);
    Unexpected error(TestFailure::Missing);
    static_assert(std::same_as<decltype(error), Unexpected<TestFailure>>);
}

TEST(ExpectedDeathTest, ChecksWrongSideAccessInDebugBuilds) {
#ifndef NDEBUG
    Expected<int, TestFailure> failure = Unexpected(TestFailure::Missing);
    const auto& constFailure = failure;
    EXPECT_DEBUG_DEATH(static_cast<void>(*failure), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(*constFailure), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(failure.operator->()), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(constFailure.operator->()), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(*std::move(failure)), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(*std::move(constFailure)), "");
    auto success = readExpected(true);
    const auto& constSuccess = success;
    EXPECT_DEBUG_DEATH(static_cast<void>(success.error()), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(constSuccess.error()), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(std::move(success).error()), "");
    EXPECT_DEBUG_DEATH(static_cast<void>(std::move(constSuccess).error()), "");
#else
    GTEST_SKIP() << "Wrong-side access is checked only when assertions are enabled.";
#endif
}
#endif

}  // namespace
