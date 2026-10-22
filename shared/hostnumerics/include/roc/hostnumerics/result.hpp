// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>
#include <variant>

// Design history, tradeoffs, and C++23 migration:
//
// Layout, scalar conversion, storage, and descriptor checks need a value or a
// typed failure so callers can choose a fallback without classifying exception
// messages. bool or optional would lose the failure reason and context; separate
// result structs for each domain would duplicate construction/access conventions.
// Callers need construction, state checks, payload access/move, and explicit
// propagation, rather than the full std::expected interface.
//
// The original helper wrapped std::variant. We then tried an established backport
// to reduce the amount of result code we maintain: TartanLlama/expected v1.3.1
// behind a restricted facade. Its converting-result lifetime defect (upstream
// https://github.com/TartanLlama/expected/issues/180) and boolean-conversion
// pitfalls required explicit propagation.
// expected-lite v0.10.0 was also considered, but a throwing value conversion
// incorrectly terminated in our probe because of its noexcept specification.
// At evaluation, Beman/expected described itself as in development and did not
// support MSVC. These findings concern the evaluated versions, not future releases.
// Backport constexpr/noexcept differences were acceptable where consumers allowed
// them; passing upstream tests alone did not establish our consumers' contract.
//
// We returned to this small variant wrapper after checking the actual consumers.
// The backport covered a much larger API than they need and still required a
// local facade and pinned dependency. This wrapper leaves less code to maintain.
// std::variant supplies storage, lifetimes, copy, move, and destruction without
// maintaining a manual union. Assignment is disabled to prevent throwing changes
// between alternatives from leaving a live result valueless. The wrapper adds no
// allocation or exception translation; payloads may still allocate or throw.
// TartanLlama's CC0 license permitted bundling without another license file, so
// licensing did not rule it out. The chosen wrapper copies no third-party code
// and uses the existing HostNumerics MIT license.
//
// std::expected is the intended replacement once all affected C++ consumers and
// compiler/standard-library combinations can use C++23. ROCm libraries/systems
// still include C++17/20 consumers, and inline layout/scalar helpers include this
// header: putting it in detail does not hide a C++23 parsing requirement. Public
// C ABIs need not change, but the affected C++ source baseline would. Moving these
// helpers behind a compiled C++23 implementation could keep C++20 headers, at the
// cost of a larger refactor, so we retain C++20 here for now.
//
// At that transition, migrate this internal interface and its callers together to
// std::expected/std::unexpected, update exported CMake requirements, and rerun the
// contract, package, sanitizer, and consumer tests on supported toolchains.
// Recheck constexpr/noexcept, ownership, conversion/propagation, and the scalar
// no-exceptions/no-allocation leaf. Retain explicit propagation until converting
// result behavior is verified; do not require a drop-in alias or expose additional
// standard APIs just to replace storage. No source or ABI compatibility is promised
// for this internal helper, and no automatic per-toolchain backend switch is used.

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

template <typename T, typename E>
class Expected;

namespace result_detail {
template <typename T>
inline constexpr bool IsUnexpectedV = false;
template <typename E>
inline constexpr bool IsUnexpectedV<Unexpected<E>> = true;

template <typename T>
inline constexpr bool IsExpectedV = false;
template <typename T, typename E>
inline constexpr bool IsExpectedV<Expected<T, E>> = true;
}  // namespace result_detail

// A value or typed failure for internal C++20 helpers. std::variant manages
// storage and lifetimes; callers check the state before accessing a payload.
template <typename T, typename E>
class [[nodiscard]] Expected {
    static_assert(std::is_object_v<T> && std::is_object_v<E>);
    static_assert(!result_detail::IsUnexpectedV<std::remove_cv_t<T>> &&
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
                 !result_detail::IsExpectedV<std::remove_cvref_t<U>> &&
                 !result_detail::IsUnexpectedV<std::remove_cvref_t<U>> &&
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
    // Keep variant's valueless state unreachable by disabling state changes.
    // A throwing construction prevents the result from coming into existence.
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
