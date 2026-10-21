// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <roc/hostnumerics/compat/tl/expected.hpp>
#include <type_traits>
#include <utility>

namespace roc::hostnumerics::compat {
// Explicitly marks an error, including when the value and error types match.
template <typename E>
class unexpected {
   public:
    using error_type = E;

    template <typename G = E>
        requires(!std::same_as<std::remove_cvref_t<G>, unexpected> &&
                 !std::same_as<std::remove_cvref_t<G>, std::in_place_t> &&
                 std::constructible_from<E, G &&>)
    constexpr explicit unexpected(G&& error) noexcept(std::is_nothrow_constructible_v<E, G&&>)
        : m_error(std::forward<G>(error)) {}

    template <typename... Args>
        requires std::constructible_from<E, Args&&...>
    constexpr explicit unexpected(std::in_place_t, Args&&... args) noexcept(
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
unexpected(E) -> unexpected<E>;

template <typename T, typename E>
class expected;

namespace detail {
template <typename T>
inline constexpr bool isUnexpectedV = false;
template <typename E>
inline constexpr bool isUnexpectedV<unexpected<E>> = true;

template <typename T>
inline constexpr bool isExpectedV = false;
template <typename T, typename E>
inline constexpr bool isExpectedV<expected<T, E>> = true;
template <typename T, typename E>
inline constexpr bool isExpectedV<tl::expected<T, E>> = true;
}  // namespace detail

// A restricted std::expected interface for C++20 callers. The upstream base owns
// the active payload; its converting-result constructors and mutating operations
// stay inaccessible. Explicit propagation constructs a value or tagged error.
template <typename T, typename E>
class [[nodiscard]] expected : private tl::expected<T, E> {
    using base_type = tl::expected<T, E>;
    static_assert(std::is_object_v<T> && std::is_object_v<E>);
    static_assert(!detail::isUnexpectedV<std::remove_cv_t<T>> &&
                  !std::same_as<std::remove_cv_t<T>, std::in_place_t>);

   public:
    using value_type = T;
    using error_type = E;
    using unexpected_type = unexpected<E>;

    constexpr expected() noexcept(std::is_nothrow_default_constructible_v<T>)
        requires std::is_default_constructible_v<T>
        : base_type() {}

    template <typename U = T>
        requires(!std::same_as<std::remove_cvref_t<U>, expected> &&
                 !detail::isExpectedV<std::remove_cvref_t<U>> &&
                 !detail::isUnexpectedV<std::remove_cvref_t<U>> &&
                 !std::same_as<std::remove_cvref_t<U>, std::in_place_t> &&
                 std::constructible_from<T, U &&>)
    constexpr explicit(!std::is_convertible_v<U&&, T>)
        expected(U&& value) noexcept(std::is_nothrow_constructible_v<T, U&&>)
        : base_type(tl::in_place, std::forward<U>(value)) {}

    template <typename G>
        requires std::constructible_from<E, const G&>
    constexpr explicit(!std::is_convertible_v<const G&, E>)
        expected(const unexpected<G>& error) noexcept(std::is_nothrow_constructible_v<E, const G&>)
        : base_type(tl::unexpect, error.error()) {}

    template <typename G>
        requires std::constructible_from<E, G&&>
    constexpr explicit(!std::is_convertible_v<G&&, E>)
        expected(unexpected<G>&& error) noexcept(std::is_nothrow_constructible_v<E, G&&>)
        : base_type(tl::unexpect, std::move(error).error()) {}

    template <typename... Args>
        requires std::constructible_from<T, Args&&...>
    constexpr explicit expected(std::in_place_t, Args&&... args) noexcept(
        std::is_nothrow_constructible_v<T, Args&&...>)
        : base_type(tl::in_place, std::forward<Args>(args)...) {}

    expected(const expected&) = default;
    expected(expected&&) = default;
    // Construct another result so payload assignment cannot change its state.
    expected& operator=(const expected&) = delete;
    expected& operator=(expected&&) = delete;

    using base_type::has_value;
    using base_type::operator bool;
    using base_type::operator*;
    using base_type::operator->;
    using base_type::error;
};
}  // namespace roc::hostnumerics::compat
