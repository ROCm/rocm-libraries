// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "mosaic/int_const.hpp"
#include "mosaic/int_tuple.hpp"

// ===================================================================
// CuTe-style underscore slicing.
//
//   _                — placeholder marking "keep this mode"
//   slice(coord, x)  — returns the elements of x paired with `_` in coord
//   dice(coord, x)   — opposite: returns elements paired with non-`_`
//
// Works on:
//   - tuples (filtering tuples by an underscore-coord pattern)
//   - layouts (defined in layout.hpp, on top of the tuple version)
//
// Inspired by CuTe underscore slicing (Cecka; NVIDIA CUTLASS cute::underscore).
// ===================================================================

namespace mosaic {

// ---------------------------------------------------------------------------
// Underscore — the placeholder. Inheriting Int<0> matches CuTe and makes _
// usable as a degenerate integer in coord-eval contexts.
// ---------------------------------------------------------------------------

struct Underscore : Int<0> {};

inline constexpr Underscore _ = {};

// ---------------------------------------------------------------------------
// Traits
// ---------------------------------------------------------------------------

template <class T>
struct is_underscore : std::false_type {};

template <>
struct is_underscore<Underscore> : std::true_type {};

template <class T>
inline constexpr bool is_underscore_v = is_underscore<std::decay_t<T>>::value;

// has_underscore<T>: true if T is Underscore, OR if T is a tuple that
// (recursively) contains an Underscore at any nesting level.
template <class T>
struct has_underscore : is_underscore<std::decay_t<T>> {};

template <class... Ts>
struct has_underscore<std::tuple<Ts...>> : std::bool_constant<(has_underscore<Ts>::value || ...)> {
};

template <class T>
inline constexpr bool has_underscore_v = has_underscore<std::decay_t<T>>::value;

// ---------------------------------------------------------------------------
// filter_tuple — apply f(a_i, b_i) to each pair and concatenate.
// Each f result is expected to be a (possibly empty) tuple. The
// concatenation is what makes filter "drop" an element when f returns ().
// ---------------------------------------------------------------------------

namespace detail {

template <class A, class B, class F>
MOSAIC_HOST_DEVICE constexpr auto filter_tuple(A const& a, B const& b, F&& f) {
    static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>,
                  "filter_tuple: rank mismatch between A and B");
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::tuple_cat(f(std::get<I>(a), std::get<I>(b))...);
    }(std::make_index_sequence<std::tuple_size_v<A>>{});
}

}  // namespace detail

// ---------------------------------------------------------------------------
// slice(coord, x)
//
// Examples:
//   slice(_,        5)       → 5            (single _: return whole thing)
//   slice(3,        5)       → ()           (no _: empty)
//   slice((_, 3),   (a, b))  → (a,)         (mixed: keep only paired-with-_)
//   slice((_, _),   (a, b))  → (a, b)
//   slice(((_,3),_), ((a,b),c)) → (a, c)    (nested)
//
// "lift_slice" always returns a tuple so the recursive descent can
// concatenate. Top-level `slice` unwraps the single-_ case.
// ---------------------------------------------------------------------------

namespace detail {

template <class A, class B>
MOSAIC_HOST_DEVICE constexpr auto lift_slice(A const& a, B const& b) {
    if constexpr (is_tuple_v<A>) {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>,
                      "slice: rank mismatch between coord and target");
        return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_slice(x, y); });
    } else if constexpr (is_underscore_v<A>) {
        return std::make_tuple(b);
    } else {
        return std::tuple<>{};
    }
}

}  // namespace detail

template <class A, class B>
MOSAIC_HOST_DEVICE constexpr auto slice(A const& a, B const& b) {
    if constexpr (is_tuple_v<A>) {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>,
                      "slice: rank mismatch between coord and target");
        return detail::filter_tuple(
            a, b, [](auto const& x, auto const& y) { return detail::lift_slice(x, y); });
    } else if constexpr (is_underscore_v<A>) {
        return b;
    } else {
        return std::tuple<>{};
    }
}

// ---------------------------------------------------------------------------
// dice(coord, x) — opposite filter: keep elements paired with NON-`_`.
//
// Examples:
//   dice(_,        5)       → ()
//   dice(3,        5)       → 5
//   dice((_, 3),   (a, b))  → (b,)
// ---------------------------------------------------------------------------

namespace detail {

template <class A, class B>
MOSAIC_HOST_DEVICE constexpr auto lift_dice(A const& a, B const& b) {
    if constexpr (is_tuple_v<A>) {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>,
                      "dice: rank mismatch between coord and target");
        return filter_tuple(a, b, [](auto const& x, auto const& y) { return lift_dice(x, y); });
    } else if constexpr (is_underscore_v<A>) {
        return std::tuple<>{};
    } else {
        return std::make_tuple(b);
    }
}

}  // namespace detail

template <class A, class B>
MOSAIC_HOST_DEVICE constexpr auto dice(A const& a, B const& b) {
    if constexpr (is_tuple_v<A>) {
        static_assert(std::tuple_size_v<A> == std::tuple_size_v<B>,
                      "dice: rank mismatch between coord and target");
        return detail::filter_tuple(
            a, b, [](auto const& x, auto const& y) { return detail::lift_dice(x, y); });
    } else if constexpr (is_underscore_v<A>) {
        return std::tuple<>{};
    } else {
        return b;
    }
}

}  // namespace mosaic
