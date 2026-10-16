// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

namespace roc::hostnumerics::detail {
// Explicitly marks an error, including when the value and error types match.
template <typename E>
class Unexpected {
   public:
    using error_type = E;

    template <typename G = E>
        requires(!std::same_as<std::remove_cvref_t<G>, Unexpected> &&
                 !std::same_as<std::remove_cvref_t<G>, std::in_place_t> &&
                 std::constructible_from<E, G &&>)
    constexpr explicit Unexpected(G&& error) noexcept(std::is_nothrow_constructible_v<E, G&&>)
        : m_error(std::forward<G>(error)) {}

    template <typename... Args>
        requires std::constructible_from<E, Args&&...>
    constexpr explicit Unexpected(std::in_place_t, Args&&... args) noexcept(
        std::is_nothrow_constructible_v<E, Args&&...>)
        : m_error(std::forward<Args>(args)...) {}

    constexpr E& error() & noexcept {
        return m_error;
    }
    constexpr const E& error() const& noexcept {
        return m_error;
    }
    constexpr E&& error() && noexcept {
        return std::move(m_error);
    }
    constexpr const E&& error() const&& noexcept {
        return std::move(m_error);
    }

   private:
    E m_error;
};

template <typename E>
Unexpected(E) -> Unexpected<E>;

template <typename T>
inline constexpr bool IsUnexpectedV = false;
template <typename E>
inline constexpr bool IsUnexpectedV<Unexpected<E>> = true;

template <typename T, typename E>
class Expected;
template <typename T>
inline constexpr bool IsExpectedV = false;
template <typename T, typename E>
inline constexpr bool IsExpectedV<Expected<T, E>> = true;

// Internal value-or-error result, following the core std::expected<T, E> model.
// Kept in C++20 because std::expected requires C++23 and compatible standard
// library support that we cannot yet assume for HostNumerics consumers.
// This is a small subset, not a drop-in replacement: assignment is deleted, and
// void values, converting result constructors, throwing value() access, fallback
// access, comparisons and monadic operations are omitted.
// Stores the value or failure inline; the wrapper adds no allocation or exception
// translation.
// Owning payloads can still allocate/throw. Access requires the matching state,
// like dereferencing an optional; checking the wrong state is a programming error.
template <typename T, typename E>
class [[nodiscard]] Expected {
    static_assert(std::is_object_v<T> && std::is_object_v<E>);
    static_assert(!IsUnexpectedV<std::remove_cv_t<T>> &&
                  !std::same_as<std::remove_cv_t<T>, std::in_place_t>);

   public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = Unexpected<E>;

    constexpr Expected() noexcept(std::is_nothrow_default_constructible_v<T>)
        requires std::is_default_constructible_v<T>
        : m_result(std::in_place_index<0>) {}

    template <typename U = T>
        requires(!std::same_as<std::remove_cvref_t<U>, Expected> &&
                 (!std::same_as<std::remove_cv_t<T>, bool> ||
                  !IsExpectedV<std::remove_cvref_t<U>>) &&
                 !IsUnexpectedV<std::remove_cvref_t<U>> &&
                 !std::same_as<std::remove_cvref_t<U>, std::in_place_t> &&
                 std::constructible_from<T, U &&>)
    constexpr explicit(!std::is_convertible_v<U&&, T>)
        Expected(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : m_result(std::in_place_index<0>, std::forward<U>(value)) {}

    template <typename G>
        requires std::constructible_from<E, const G&>
    constexpr explicit(!std::is_convertible_v<const G&, E>)
        Expected(const Unexpected<G>& error) noexcept(std::is_nothrow_constructible_v<E, const G&>)
        : m_result(std::in_place_index<1>, error.error()) {}

    template <typename G>
        requires std::constructible_from<E, G&&>
    constexpr explicit(!std::is_convertible_v<G&&, E>)
        Expected(Unexpected<G>&& error) noexcept(std::is_nothrow_constructible_v<E, G&&>)
        : m_result(std::in_place_index<1>, std::move(error).error()) {}

    template <typename... Args>
        requires std::constructible_from<T, Args&&...>
    constexpr explicit Expected(std::in_place_t, Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>)
        : m_result(std::in_place_index<0>, std::forward<Args>(args)...) {}

    Expected(const Expected&) = default;
    Expected(Expected&&) = default;
    // Results cannot be reassigned. Avoid variant's valueless state after a
    // throwing payload assignment; construct another result instead.
    Expected& operator=(const Expected&) = delete;
    Expected& operator=(Expected&&) = delete;

    constexpr bool has_value() const noexcept {
        return m_result.index() == 0;
    }
    constexpr explicit operator bool() const noexcept {
        return has_value();
    }

    constexpr T& operator*() & noexcept {
        assert(has_value());
        return *std::get_if<0>(&m_result);
    }
    constexpr const T& operator*() const& noexcept {
        assert(has_value());
        return *std::get_if<0>(&m_result);
    }
    constexpr T&& operator*() && noexcept {
        assert(has_value());
        return std::move(*std::get_if<0>(&m_result));
    }
    constexpr const T&& operator*() const&& noexcept {
        assert(has_value());
        return std::move(*std::get_if<0>(&m_result));
    }
    constexpr T* operator->() noexcept {
        return std::addressof(**this);
    }
    constexpr const T* operator->() const noexcept {
        return std::addressof(**this);
    }

    constexpr E& error() & noexcept {
        assert(!has_value());
        return *std::get_if<1>(&m_result);
    }
    constexpr const E& error() const& noexcept {
        assert(!has_value());
        return *std::get_if<1>(&m_result);
    }
    constexpr E&& error() && noexcept {
        assert(!has_value());
        return std::move(*std::get_if<1>(&m_result));
    }
    constexpr const E&& error() const&& noexcept {
        assert(!has_value());
        return std::move(*std::get_if<1>(&m_result));
    }

   private:
    std::variant<T, E> m_result;
};
}  // namespace roc::hostnumerics::detail
