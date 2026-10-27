// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

// Expected<T, E> holds either a value of type T or an error of type E, such as
// an enum that names the failure. HostNumerics returns it for failures that a
// caller can handle, for example an unsupported layout that has a fallback.
// Exceptions are reserved for terminal errors: HostNumerics never catches one to
// choose another path, so recoverable failures must be part of the return type.
//
// This is a subset of C++23 std::expected. HostNumerics and its consumers build
// as C++20, and inline headers that return Expected would otherwise raise their
// language standard. Existing backports such as TartanLlama/expected implement
// the full interface; one would add a large external header for the few
// operations used here.
//
// Each operation kept here behaves as in std::expected. Assignment requires
// nonthrowing move construction for both payload types. Conversion between
// result types, void values, value(), value_or(), and monadic operations are
// omitted. When HostNumerics can require C++23, replace Expected and Unexpected
// with std::expected and std::unexpected, raise the exported C++ standard, and
// delete this header.

namespace roc::hostnumerics::detail {
template <typename E>
class Unexpected;

namespace expected_detail {
template <typename T>
inline constexpr bool IsUnexpectedV = false;
template <typename E>
inline constexpr bool IsUnexpectedV<Unexpected<E>> = true;

template <typename E>
inline constexpr bool IsErrorV = std::is_object_v<E> && !std::is_array_v<E> &&
                                 !std::is_const_v<E> && !std::is_volatile_v<E> && !IsUnexpectedV<E>;
}  // namespace expected_detail

// Explicitly marks an error, including when the value and error types match.
template <typename E>
class Unexpected {
    static_assert(expected_detail::IsErrorV<E>,
                  "E must be an unqualified, non-array object type other than Unexpected.");

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

template <typename T, typename E>
class Expected;

namespace expected_detail {
template <typename T>
inline constexpr bool IsExpectedV = false;
template <typename T, typename E>
inline constexpr bool IsExpectedV<Expected<T, E>> = true;
}  // namespace expected_detail

// std::variant manages storage and lifetimes. Callers check has_value() before
// accessing a payload; accessing the inactive one is checked by assertions.
template <typename T, typename E>
class [[nodiscard]] Expected {
    static_assert(std::is_object_v<T> && !std::is_array_v<T>,
                  "T must be a non-array object type; void values are not supported.");
    static_assert(expected_detail::IsErrorV<E>,
                  "E must be an unqualified, non-array object type other than Unexpected.");
    static_assert(!expected_detail::IsUnexpectedV<std::remove_cv_t<T>> &&
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
                 !expected_detail::IsExpectedV<std::remove_cvref_t<U>> &&
                 !expected_detail::IsUnexpectedV<std::remove_cvref_t<U>> &&
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
    // Nonthrowing moves let variant preserve its active payload when a copy
    // throws during a state change, keeping its valueless state unreachable.
    Expected& operator=(const Expected&)
        requires(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_constructible_v<E>)
    = default;
    Expected& operator=(Expected&&)
        requires(std::is_nothrow_move_constructible_v<T> && std::is_nothrow_move_constructible_v<E>)
    = default;
    Expected& operator=(const Expected&)
        requires(!(std::is_nothrow_move_constructible_v<T> &&
                   std::is_nothrow_move_constructible_v<E>))
    = delete;
    Expected& operator=(Expected&&)
        requires(!(std::is_nothrow_move_constructible_v<T> &&
                   std::is_nothrow_move_constructible_v<E>))
    = delete;

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
