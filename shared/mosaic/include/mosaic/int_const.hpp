// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "mosaic/config.hpp"

namespace mosaic {

/**
 * @brief Compile-time integral/value constant wrapper.
 *
 * Allows us to use constant values for layout arguments and operations and perform
 * math at compile time.
 *
 */
template <auto v>
struct C {
    using type = C<v>;
    using value_type = decltype(v);
    static constexpr value_type value = v;

    MOSAIC_HOST_DEVICE constexpr operator value_type() const noexcept {
        return value;
    }
    MOSAIC_HOST_DEVICE constexpr value_type operator()() const noexcept {
        return value;
    }
};

// Alias matching CuTe's naming. Int<16>{} and C<16>{} are the same type.
template <auto v>
using Int = C<v>;

// ---------------------------------------------------------------------------
// Type traits
// ---------------------------------------------------------------------------

// is_static: true for types with no runtime state (empty types like C<v>)
template <class T>
struct is_static : std::bool_constant<std::is_empty_v<T>> {};

template <class T>
inline constexpr bool is_static_v = is_static<T>::value;

// is_c: detect C<v> specifically
template <class T>
struct is_c : std::false_type {};

template <auto v>
struct is_c<C<v>> : std::true_type {};

template <class T>
inline constexpr bool is_c_v = is_c<T>::value;

// is_integral: matches built-in integrals AND C<v>
template <class T>
struct is_integral : std::bool_constant<std::is_integral_v<T> || is_c_v<T>> {};

template <class T>
inline constexpr bool is_integral_v = is_integral<T>::value;

// is_constant: check if a type statically equals a specific value
template <auto n, class T>
struct is_constant : std::false_type {};

template <auto n, auto v>
struct is_constant<n, C<v>> : std::bool_constant<n == v> {};

template <auto n, class T>
inline constexpr bool is_constant_v = is_constant<n, T>::value;

// ---------------------------------------------------------------------------
// Arithmetic on C<> — returns C<result> (compile-time propagation)
// ---------------------------------------------------------------------------

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr C<(a + b)> operator+(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr C<(a - b)> operator-(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr C<(a * b)> operator*(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr C<(a / b)> operator/(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr C<(a % b)> operator%(C<a>, C<b>) {
    return {};
}

// Comparison
template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr std::bool_constant<(a == b)> operator==(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr std::bool_constant<(a != b)> operator!=(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr std::bool_constant<(a < b)> operator<(C<a>, C<b>) {
    return {};
}

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr std::bool_constant<(a <= b)> operator<=(C<a>, C<b>) {
    return {};
}

// ---------------------------------------------------------------------------
// Mixed arithmetic: C<v> op runtime → runtime
// ---------------------------------------------------------------------------

// C<v> + runtime
template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator+(C<v>, T b) {
    return static_cast<T>(v) + b;
}

template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator+(T a, C<v>) {
    return a + static_cast<T>(v);
}

// C<v> - runtime
template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator-(C<v>, T b) {
    return static_cast<T>(v) - b;
}

template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator-(T a, C<v>) {
    return a - static_cast<T>(v);
}

// C<v> * runtime
template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator*(C<v>, T b) {
    return static_cast<T>(v) * b;
}

template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator*(T a, C<v>) {
    return a * static_cast<T>(v);
}

// C<v> / runtime
template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator/(C<v>, T b) {
    return static_cast<T>(v) / b;
}

template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator/(T a, C<v>) {
    return a / static_cast<T>(v);
}

// C<v> % runtime
template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator%(C<v>, T b) {
    return static_cast<T>(v) % b;
}

template <auto v, class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto operator%(T a, C<v>) {
    return a % static_cast<T>(v);
}

// ---------------------------------------------------------------------------
// divmod — single division yielding both quotient and remainder.
// On GPU hardware, a single div instruction gives both.
// ---------------------------------------------------------------------------

template <class A, class B>
struct DivMod {
    A quot;
    B rem;
};

// Both static
template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr auto divmod(C<a>, C<b>) {
    return DivMod<C<a / b>, C<a % b>>{};
}

// Mixed / runtime
template <class A, class B>
    requires(std::is_integral_v<A> && std::is_integral_v<B>)
MOSAIC_HOST_DEVICE constexpr auto divmod(A a, B b) {
    return DivMod<A, A>{static_cast<A>(a / b), static_cast<A>(a % b)};
}

// C<v> with runtime
template <auto v, class B>
    requires(std::is_integral_v<B> && !is_c_v<B>)
MOSAIC_HOST_DEVICE constexpr auto divmod(C<v>, B b) {
    using R = decltype(static_cast<B>(v) / b);
    return DivMod<R, R>{static_cast<B>(v) / b, static_cast<B>(v) % b};
}

template <class A, auto v>
    requires(std::is_integral_v<A> && !is_c_v<A>)
MOSAIC_HOST_DEVICE constexpr auto divmod(A a, C<v>) {
    return DivMod<A, A>{static_cast<A>(a / v), static_cast<A>(a % v)};
}

// ---------------------------------------------------------------------------
// abs — branchless absolute value, static-preserving for C<>
// ---------------------------------------------------------------------------

template <auto v>
MOSAIC_HOST_DEVICE constexpr auto abs(C<v>) {
    if constexpr (v < 0)
        return C<-v>{};
    else
        return C<v>{};
}

template <class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto abs(T x) {
    return x < 0 ? -x : x;
}

// ---------------------------------------------------------------------------
// signum — sign as {-1, 0, 1}, static-preserving for C<>
// Defined for v==0 (returns C<0>) so composition's fold can safely use it
// even though branch 3 short-circuits the rhs-stride-0 case upstream.
// ---------------------------------------------------------------------------

template <auto v>
MOSAIC_HOST_DEVICE constexpr auto signum(C<v>) {
    if constexpr (v > 0)
        return C<1>{};
    else if constexpr (v < 0)
        return C<-1>{};
    else
        return C<0>{};
}

template <class T>
    requires(std::is_integral_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr int signum(T x) {
    return (x > 0) - (x < 0);
}

// ---------------------------------------------------------------------------
// ceil_div(a, b) = (a + b - 1) / b for non-negative a, b > 0.
// Static-preserving when both inputs are C<>.
// ---------------------------------------------------------------------------

template <auto a, auto b>
MOSAIC_HOST_DEVICE constexpr auto ceil_div(C<a>, C<b>) {
    static_assert(b != 0, "ceil_div: divisor cannot be zero");
    return C<(a + b - 1) / b>{};
}

template <auto v, class B>
    requires(std::is_integral_v<B> && !is_c_v<B>)
MOSAIC_HOST_DEVICE constexpr auto ceil_div(C<v>, B b) {
    return (static_cast<B>(v) + b - 1) / b;
}

template <class A, auto v>
    requires(std::is_integral_v<A> && !is_c_v<A>)
MOSAIC_HOST_DEVICE constexpr auto ceil_div(A a, C<v>) {
    return (a + static_cast<A>(v) - 1) / static_cast<A>(v);
}

template <class A, class B>
    requires(std::is_integral_v<A> && std::is_integral_v<B> && !is_c_v<A> && !is_c_v<B>)
MOSAIC_HOST_DEVICE constexpr auto ceil_div(A a, B b) {
    return (a + b - 1) / b;
}

// ---------------------------------------------------------------------------
// Convenience aliases — the most commonly used compile-time constants
// ---------------------------------------------------------------------------

using _0 = C<0>;
using _1 = C<1>;
using _2 = C<2>;
using _3 = C<3>;
using _4 = C<4>;
using _5 = C<5>;
using _6 = C<6>;
using _7 = C<7>;
using _8 = C<8>;
using _9 = C<9>;
using _10 = C<10>;
using _12 = C<12>;
using _16 = C<16>;
using _24 = C<24>;
using _32 = C<32>;
using _48 = C<48>;
using _64 = C<64>;
using _96 = C<96>;
using _128 = C<128>;
using _192 = C<192>;
using _256 = C<256>;

}  // namespace mosaic
