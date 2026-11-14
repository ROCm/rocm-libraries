// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <type_traits>
#include <utility>

#include "mosaic/int_const.hpp"

namespace mosaic {

// TODO: CuTe uses their own custom tuple type.
//       There seems to be some advantages to this, but we're building off of std::tuple for now.
// TODO: congruent, coalesce, compatible,

// ---------------------------------------------------------------------------
// is_tuple trait
// ---------------------------------------------------------------------------

template <class T>
struct is_tuple : std::false_type {};

template <class... Ts>
struct is_tuple<std::tuple<Ts...>> : std::true_type {};

template <class T>
inline constexpr bool is_tuple_v = is_tuple<std::decay_t<T>>::value;

template <class T>
concept IsIntTuple = std::is_integral_v<T> || is_c_v<T> || is_tuple_v<T>;

/**
 * @brief The rank of an IntTuple is the number of top-level modes i.e its length.
 *
 * Optional template params Is... let you query a sub-element's rank:
 *   rank(t)        → rank of the whole tuple
 *   rank<2>(t)     → rank of get<2>(t)
 *   rank<2, 0>(t)  → rank of get<0>(get<2>(t))
 */
template <int... Is, IsIntTuple IntTuple>
MOSAIC_HOST_DEVICE constexpr auto rank(IntTuple const& t) {
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple_v<IntTuple>) {
            return Int<std::tuple_size_v<IntTuple>>{};
        } else {
            return Int<1>{};
        }
    } else {
        return rank(get<Is...>(t));
    }
}

// Type-trait aliases derived from the function. Lets you write rank_v<MyShape>
// in template contexts where you don't have a value to call rank() on.
template <class IntTuple>
using rank_t = decltype(rank(std::declval<IntTuple>()));

template <class IntTuple>
inline constexpr auto rank_v = rank_t<IntTuple>::value;

/**
 * @brief max — variadic, tuple-aware, compile-time-preserving for Int<v>
 */

// Compile-time binary base case: returns Int<max(A,B)>, preserves type info.
// Picked over the runtime binary below when both args are Int<v> (more specialized).
template <auto A, auto B>
MOSAIC_HOST_DEVICE constexpr auto max(Int<A>, Int<B>) {
    if constexpr (A >= B)
        return Int<A>{};
    else
        return Int<B>{};
}

// Binary min: static base case for two C<> args. The variadic max above has
// no binary-min counterpart; composition needs min(next_shape, rest_shape)
// to be static-preserving so the result layout's leaves stay C<>.
template <auto A, auto B>
MOSAIC_HOST_DEVICE constexpr auto min(Int<A>, Int<B>) {
    if constexpr (A <= B)
        return Int<A>{};
    else
        return Int<B>{};
}

// Runtime binary min for any other arithmetic types.
template <class T, class U>
    requires(std::is_arithmetic_v<T> && std::is_arithmetic_v<U> && !is_c_v<T> && !is_c_v<U>)
MOSAIC_HOST_DEVICE constexpr auto min(T const& t, U const& u) {
    return t < u ? t : u;
}

// Mixed min: C<v> with runtime. Result is runtime since the comparison
// can't be resolved statically.
template <auto v, class T>
    requires(std::is_arithmetic_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto min(Int<v>, T const& t) {
    return static_cast<T>(v) < t ? static_cast<T>(v) : t;
}

template <class T, auto v>
    requires(std::is_arithmetic_v<T> && !is_c_v<T>)
MOSAIC_HOST_DEVICE constexpr auto min(T const& t, Int<v>) {
    return t < static_cast<T>(v) ? t : static_cast<T>(v);
}

// Runtime binary base case: any other arithmetic types (built-in ints, etc).
template <class T, class U>
    requires(std::is_arithmetic_v<T> && std::is_arithmetic_v<U> && !is_c_v<T> && !is_c_v<U>)
MOSAIC_HOST_DEVICE constexpr auto max(T const& t, U const& u) {
    return t < u ? u : t;
}

// Variadic + tuple-aware: handles N-ary calls and recurses into tuple args.
// Always delegates the actual "which is bigger" decision to a binary overload.
template <class T0, class... Ts>
MOSAIC_HOST_DEVICE constexpr auto max(T0 const& t0, Ts const&... ts) {
    if constexpr (is_tuple_v<T0>) {
        // First arg is a tuple: collapse it via max over its leaves, then continue
        return max(std::apply([](auto const&... a) { return max(a...); }, t0), ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        // Single scalar
        return t0;
    } else {
        // N-ary: peel one off, combine via binary max
        return max(t0, max(ts...));
    }
}

// Variadic + tuple-aware min, mirroring the variadic max above. Needed by
// complement's sort-by-stride step (min over remaining-mode strides).
template <class T0, class... Ts>
MOSAIC_HOST_DEVICE constexpr auto min(T0 const& t0, Ts const&... ts) {
    if constexpr (is_tuple_v<T0>) {
        return min(std::apply([](auto const&... a) { return min(a...); }, t0), ts...);
    } else if constexpr (sizeof...(Ts) == 0) {
        return t0;
    } else {
        return min(t0, min(ts...));
    }
}

/**
 * @brief Let X = (X_0, X_1, ...) be an IntTuple. depth(X) = max(depth(X_0, X_1, ...)) + 1.
 *
 * Like rank, we have optional template params Is... to query a sub-element's depth:
 *   depth(X)        → depth of the whole tuple
 *   depth<2>(X)     → depth of get<2>(X)
 *   depth<2, 0>(X)  → depth of get<0>(get<2>(X))
 */
template <int... Is, IsIntTuple IntTuple>
MOSAIC_HOST_DEVICE constexpr auto depth(IntTuple const& t) {
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple_v<IntTuple>) {
            return Int<1>{} + std::apply([](auto const&... v) { return max(depth(v)...); }, t);
        } else {
            // depth of a scalar is 0
            return Int<0>{};
        }
    } else {
        return depth(get<Is...>(t));
    }
}

template <class IntTuple>
using depth_t = decltype(depth(std::declval<IntTuple>()));

template <class IntTuple>
inline constexpr auto depth_v = depth_t<IntTuple>::value;

/**
 * @brief product — recursive product of all leaf values
 *
 * Like rank and depth, we have optional template params Is... to query a sub-element's product:
 *   product(X)        → product of the whole tuple
 *   product<2>(X)     → product of get<2>(X)
 *   product<2, 0>(X)  → product of get<0>(get<2>(X))
 */
template <int... Is, IsIntTuple IntTuple>
MOSAIC_HOST_DEVICE constexpr auto product(IntTuple const& t) {
    if constexpr (sizeof...(Is) == 0) {
        if constexpr (is_tuple_v<IntTuple>) {
            return std::apply([](auto const&... args) { return (product(args) * ...); }, t);
        } else {
            return t;
        }
    } else {
        return product(get<Is...>(t));
    }
}

/**
 * @brief size — alias for product
 */
template <int... Is, IsIntTuple IntTuple>
MOSAIC_HOST_DEVICE constexpr auto size(IntTuple const& t) {
    return product<Is...>(t);
}

template <class IntTuple>
inline constexpr auto size_v = decltype(size(std::declval<IntTuple>()))::value;

// Single-index get: t[I]. Scalars only support I==0.
template <int I, class T>
MOSAIC_HOST_DEVICE constexpr auto get(T const& t) {
    if constexpr (is_tuple_v<T>) {
        return std::get<I>(t);
    } else {
        static_assert(I == 0, "Scalar only has index 0");
        return t;
    }
}

// Multi-index get: get<I0, I1, ...>(t) == get<I1, ...>(get<I0>(t)).
// Lets you drill into nested tuples in one call.
template <int I0, int I1, int... Is, class T>
MOSAIC_HOST_DEVICE constexpr auto get(T const& t) {
    return get<I1, Is...>(get<I0>(t));
}

// ---------------------------------------------------------------------------
// flatten — recursively flatten nested tuples to a single level.
// Used by algebra ops that need to walk leaves uniformly (coalesce,
// complement, right_inverse).
// ---------------------------------------------------------------------------

namespace detail {

template <class T>
MOSAIC_HOST_DEVICE constexpr auto flatten_impl(T const& t) {
    if constexpr (is_tuple_v<T>) {
        return std::apply([](auto const&... args) { return std::tuple_cat(flatten_impl(args)...); },
                          t);
    } else {
        return std::make_tuple(t);
    }
}

}  // namespace detail

template <class T>
MOSAIC_HOST_DEVICE constexpr auto flatten(T const& t) {
    return detail::flatten_impl(t);
}

// ---------------------------------------------------------------------------
// for_each — execute a function on each element of a tuple (or scalar).
// ---------------------------------------------------------------------------

template <class T, class F>
MOSAIC_HOST_DEVICE constexpr void for_each(T const& t, F&& f) {
    if constexpr (is_tuple_v<T>) {
        std::apply([&f](auto const&... args) { (f(args), ...); }, t);
    } else {
        f(t);
    }
}

// ---------------------------------------------------------------------------
// congruent — two tuples have the same nesting profile
// ---------------------------------------------------------------------------

template <class A, class B>
struct congruent : std::false_type {};

template <class A, class B>
    requires(!is_tuple_v<A> && !is_tuple_v<B>)
struct congruent<A, B> : std::true_type {};

template <class... As, class... Bs>
    requires(sizeof...(As) == sizeof...(Bs))
struct congruent<std::tuple<As...>, std::tuple<Bs...>>
    : std::bool_constant<(congruent<As, Bs>::value && ...)> {};

template <class A, class B>
inline constexpr bool congruent_v = congruent<std::decay_t<A>, std::decay_t<B>>::value;

// ---------------------------------------------------------------------------
// compatible — two shapes have the same total product (size)
// ---------------------------------------------------------------------------

template <class A, class B>
MOSAIC_HOST_DEVICE constexpr bool compatible(A const& a, B const& b) {
    return product(a) == product(b);
}

// ---------------------------------------------------------------------------
// transform — apply a function element-wise to a tuple
// ---------------------------------------------------------------------------

template <class T, class F>
MOSAIC_HOST_DEVICE constexpr auto transform(T const& t, F&& f) {
    if constexpr (is_tuple_v<T>) {
        return std::apply([&f](auto const&... args) { return std::make_tuple(f(args)...); }, t);
    } else {
        return f(t);
    }
}

template <class A, class B, class F>
MOSAIC_HOST_DEVICE constexpr auto transform(A const& a, B const& b, F&& f) {
    if constexpr (is_tuple_v<A> && is_tuple_v<B>) {
        static_assert(rank_v<A> == rank_v<B>, "Rank mismatch in binary transform");
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(f(std::get<I>(a), std::get<I>(b))...);
        }(std::make_index_sequence<rank_v<A>>{});
    } else {
        return f(a, b);
    }
}

// ---------------------------------------------------------------------------
// fold — left fold over tuple elements
// ---------------------------------------------------------------------------

template <class T, class Init, class F>
MOSAIC_HOST_DEVICE constexpr auto fold(T const& t, Init init, F&& f) {
    if constexpr (is_tuple_v<T>) {
        auto result = init;
        std::apply([&](auto const&... args) { ((result = f(result, args)), ...); }, t);
        return result;
    } else {
        return f(init, t);
    }
}

// ---------------------------------------------------------------------------
// tuple_cat / append / prepend — convenience wrappers over std::tuple_cat
// ---------------------------------------------------------------------------

using std::tuple_cat;

template <class T, class E>
MOSAIC_HOST_DEVICE constexpr auto append(T const& t, E const& e) {
    if constexpr (is_tuple_v<T>) {
        return std::apply([&e](auto const&... args) { return std::make_tuple(args..., e); }, t);
    } else {
        return std::make_tuple(t, e);
    }
}

template <class E, class T>
MOSAIC_HOST_DEVICE constexpr auto prepend(E const& e, T const& t) {
    if constexpr (is_tuple_v<T>) {
        return std::apply([&e](auto const&... args) { return std::make_tuple(e, args...); }, t);
    } else {
        return std::make_tuple(e, t);
    }
}

// Replace the last element of a non-empty tuple. Used by coalesce to merge
// the trailing accumulator entry with the next incoming mode.
template <class T, class E>
MOSAIC_HOST_DEVICE constexpr auto replace_back(T const& t, E const& e) {
    static_assert(is_tuple_v<T>, "replace_back: T must be a tuple");
    constexpr std::size_t N = std::tuple_size_v<T>;
    static_assert(N >= 1, "replace_back: tuple must be non-empty");
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(std::get<I>(t)..., e);
    }(std::make_index_sequence<N - 1>{});
}

// unwrap — extract the single element from a rank-1 tuple; identity otherwise.
// Used by composition's wrap-up case B to deflate a result that ended up with
// exactly one mode back to flat scalar shape/stride form.
template <class T>
MOSAIC_HOST_DEVICE constexpr decltype(auto) unwrap(T const& t) {
    if constexpr (is_tuple_v<T> && std::tuple_size_v<T> == 1) {
        return std::get<0>(t);
    } else {
        return t;
    }
}

// ---------------------------------------------------------------------------
// zip2_by — regroup paired modes according to a profile.
//
// `modes` is a tuple. `profile` names which leading positions are pairs:
//   - a tuple profile entry: recurse on (modes[i], profile[i])
//   - a scalar profile entry: modes[i] is already a rank-2 (head, tail) pair
// Positions past the profile's rank are leftovers and join the tails.
//
// The result is always rank 2:
//   (head of each profiled pair, tail of each profiled pair, then leftovers)
//
// Only consumer is detail::tile_unzip, which turns logical_divide's per-mode
// (tile, rest) pairs into ((tiles...), (rests...)).
// ---------------------------------------------------------------------------

template <class Modes, class Profile>
MOSAIC_HOST_DEVICE constexpr auto zip2_by(Modes const& modes, Profile const& profile);

namespace detail {

// Heads of the first J already-zipped pairs, as one tuple.
template <std::size_t J, class Pairs>
MOSAIC_HOST_DEVICE constexpr auto zip2_halves(Pairs const& pairs) {
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(std::get<0>(std::get<I>(pairs))...);
    }(std::make_index_sequence<J>{});
}

// Tails of the zipped pairs, then leftover modes starting at Skip.
template <std::size_t Skip, class Pairs, class Modes>
MOSAIC_HOST_DEVICE constexpr auto zip2_leftovers(Pairs const& pairs, Modes const& modes) {
    constexpr std::size_t n_pairs = std::tuple_size_v<Pairs>;
    constexpr std::size_t n_modes = std::tuple_size_v<Modes>;
    return [&]<std::size_t... I, std::size_t... K>(std::index_sequence<I...>,
                                                   std::index_sequence<K...>) {
        return std::make_tuple(std::get<1>(std::get<I>(pairs))..., std::get<Skip + K>(modes)...);
    }(std::make_index_sequence<n_pairs>{}, std::make_index_sequence<n_modes - Skip>{});
}

// Recurse on each profiled position. Returns a tuple of (head, tail) pairs.
template <class Modes, class Profile, std::size_t... I>
MOSAIC_HOST_DEVICE constexpr auto zip2_pairs(Modes const& modes, Profile const& profile,
                                             std::index_sequence<I...>) {
    return std::make_tuple(zip2_by(std::get<I>(modes), std::get<I>(profile))...);
}

}  // namespace detail

template <class Modes, class Profile>
MOSAIC_HOST_DEVICE constexpr auto zip2_by(Modes const& modes, Profile const& profile) {
    if constexpr (!is_tuple_v<Profile>) {
        static_assert(is_tuple_v<Modes> && std::tuple_size_v<Modes> == 2,
                      "zip2_by: scalar profile requires a rank-2 mode pair");
        return modes;
    } else {
        constexpr std::size_t MR = std::tuple_size_v<Modes>;
        constexpr std::size_t PR = std::tuple_size_v<Profile>;
        static_assert(MR >= PR, "zip2_by: mode rank must be >= profile rank");
        auto pairs = detail::zip2_pairs(modes, profile, std::make_index_sequence<PR>{});
        return std::tuple_cat(std::make_tuple(detail::zip2_halves<PR>(pairs)),
                              std::make_tuple(detail::zip2_leftovers<PR>(pairs, modes)));
    }
}

// ---------------------------------------------------------------------------
// repeat<N>(x) — build a tuple of N copies of x.
// For N == 1, returns x scalar (matches CUTLASS convention so callers like
// tiled_divide can use `result(_, repeat<R1>(_))` uniformly across ranks).
// ---------------------------------------------------------------------------

template <int N, class X>
MOSAIC_HOST_DEVICE constexpr auto repeat(X const& x) {
    if constexpr (N == 1) {
        return x;
    } else {
        return [&]<std::size_t... I>(std::index_sequence<I...>) {
            return std::make_tuple(((void)I, x)...);
        }(std::make_index_sequence<N>{});
    }
}

// ---------------------------------------------------------------------------
// find — compile-time index of the first element in a tuple equal to a value.
// Returns Int<idx> for a match, Int<-1> for no match. Static-only: both the
// tuple elements and the search value must be C<v> for the comparison to
// resolve at compile time.
// ---------------------------------------------------------------------------

namespace detail {

template <std::size_t I, class T, class V>
MOSAIC_HOST_DEVICE constexpr auto find_impl(T const& t, V const& v) {
    if constexpr (I >= std::tuple_size_v<T>) {
        return Int<-1>{};
    } else {
        auto elem = std::get<I>(t);
        if constexpr (is_static_v<decltype(elem)> && is_static_v<V> &&
                      decltype(elem)::value == V::value) {
            return Int<int(I)>{};
        } else {
            return find_impl<I + 1>(t, v);
        }
    }
}

}  // namespace detail

template <class T, class V>
MOSAIC_HOST_DEVICE constexpr auto find(T const& t, V const& v) {
    static_assert(is_tuple_v<T>, "find: T must be a tuple");
    return detail::find_impl<0>(t, v);
}

// ---------------------------------------------------------------------------
// remove<I> — return a new tuple with the I-th element removed.
// Implemented via an index sequence that skips position I.
// ---------------------------------------------------------------------------

template <std::size_t I, class T>
MOSAIC_HOST_DEVICE constexpr auto remove(T const& t) {
    static_assert(is_tuple_v<T>, "remove: T must be a tuple");
    constexpr std::size_t N = std::tuple_size_v<T>;
    static_assert(I < N, "remove: index out of range");
    return [&]<std::size_t... J>(std::index_sequence<J...>) {
        return std::make_tuple(std::get<(J < I ? J : J + 1)>(t)...);
    }(std::make_index_sequence<N - 1>{});
}

}  // namespace mosaic
