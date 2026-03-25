// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/integer.hpp"
#include "ck_tile/core/numeric/integral_constant.hpp"
#include "ck_tile/core/container/sequence.hpp"
#include <stdint.h>
#include <utility>

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"
namespace ck_tile {

namespace detail {

struct swallow
{
    template <typename... Ts>
    CK_TILE_HOST_DEVICE constexpr swallow(Ts&&...)
    {
    }
};

template <class>
struct static_for_impl;

template <index_t... Is>
struct static_for_impl<sequence<Is...>>
{
    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        swallow{(f(number<Is>{}), 0)...};
    }
};

} // namespace detail

// F signature: F(number<Iter>)
template <index_t NBegin, index_t NEnd, index_t Increment>
struct static_for
{
    CK_TILE_HOST_DEVICE constexpr static_for()
    {
        static_assert(Increment != 0 && (NEnd - NBegin) % Increment == 0,
                      "Wrong! should satisfy (NEnd - NBegin) % Increment == 0");
        static_assert((Increment > 0 && NBegin <= NEnd) || (Increment < 0 && NBegin >= NEnd),
                      "wrongs! should (Increment > 0 && NBegin <= NEnd) || (Increment < 0 && "
                      "NBegin >= NEnd)");
    }

    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        detail::static_for_impl<typename arithmetic_sequence_gen<NBegin, NEnd, Increment>::type>{}(
            f);
    }
};

namespace detail {

template <typename T, T... Is>
struct applier
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        // tweak -fbracket-depth if compilation fails. Clang default limit is 256
        (f(number<Is>{}), ...);
    }
};

template <int32_t Size> // == sizeof...(Is)
using make_applier = __make_integer_seq<applier, index_t, Size>;

} // namespace detail

template <index_t N>
struct static_for<0, N, 1> : detail::make_applier<N>
{
    using detail::make_applier<N>::operator();
};

template <typename... Ts>
struct static_for_product;
template <index_t... Is>
struct static_for_product<static_for<Is...>> : public static_for<Is...>
{
};
template <index_t... Is>
struct static_for_product<sequence<Is...>> : public static_for<Is...>
{
};
template <index_t I>
struct static_for_product<number<I>> : public static_for<0, I, 1>
{
};
template <typename First, typename... Rest>
struct static_for_product<First, Rest...>
{
    template <typename F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_for_product<First>{}([=](auto I) {
            static_for_product<Rest...>{}([=](auto... Is) { //
                f(I, Is...);
            });
        });
    }
};

struct identity
{
    template <typename T>
    CK_TILE_HOST_DEVICE constexpr T&& operator()(T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

// Similar to identity, but takes an additional index parameter as the first argument.
// The index is ignored and only the second argument (value) is forwarded.
// Useful for indexed element-wise operations where the functor signature requires an index.
struct idx_identity
{
    template <typename I, typename T>
    CK_TILE_HOST_DEVICE constexpr T&& operator()(I&& /*idx*/, T&& arg) const noexcept
    {
        return std::forward<T>(arg);
    }
};

namespace detail {

// Computes the inverse of a permutation as a constexpr array.
// Avoids the sequence_map_inverse -> is_valid_sequence_map -> sequence_sort chain.
template <class Perm>
struct inverse_perm;

template <index_t... Ps>
struct inverse_perm<sequence<Ps...>>
{
    static constexpr auto compute()
    {
        constexpr index_t n = sizeof...(Ps);
        static_array<index_t, n> result{};
        constexpr index_t input[] = {Ps...};
        for(index_t i = 0; i < n; ++i)
        {
            result[input[i]] = i;
        }
        return result;
    }
    static constexpr auto value = compute();
};

// Decomposes a linear index into multi-dimensional indices using pre-computed
// strides. Uses a single flat static_for instead of recursive nesting, which
// eliminates intermediate lambda closure instantiations.
template <class OrderedLengths, class IndexSeq>
struct index_decomposer;

template <index_t... Ls, index_t... Is>
struct index_decomposer<sequence<Ls...>, sequence<Is...>>
{
    static constexpr index_t n_dim                        = sizeof...(Ls);
    static constexpr static_array<index_t, n_dim> lengths = {{Ls...}};

    static constexpr static_array<index_t, n_dim> compute_all_strides()
    {
        static_array<index_t, n_dim> result{};
        if constexpr(n_dim > 0)
        {
            result[n_dim - 1] = 1;
            for(index_t i = n_dim - 1; i > 0; --i)
            {
                result[i - 1] = result[i] * lengths[i];
            }
        }
        return result;
    }

    static constexpr static_array<index_t, n_dim> strides = compute_all_strides();

    // Compile-time decomposition: linear index -> sequence of per-dimension indices
    template <index_t LinearIdx>
    using decompose = sequence<((LinearIdx / strides[Is]) % lengths[Is])...>;

    // Decompose AND reorder in one step using a pre-computed inverse permutation.
    // Produces the unordered multi-index directly, avoiding per-iteration
    // reorder_old_to_new member function instantiations on each unique sequence type.
    template <index_t LinearIdx, class New2Old>
    using decompose_reordered = sequence<((LinearIdx / strides[inverse_perm<New2Old>::value[Is]]) %
                                          lengths[inverse_perm<New2Old>::value[Is]])...>;
};

} // namespace detail

// Compile-time N-dimensional loop with static multi-indices.
// Uses a single flat static_for<0, TotalSize, 1> with index decomposition,
// eliminating intermediate lambda closures that nested static_for creates.
template <class Lengths,
          class Orders = typename arithmetic_sequence_gen<0, Lengths::size(), 1>::type>
struct static_ford
{
    static constexpr index_t n_dim = Lengths::size();
    static constexpr index_t total_size =
        reduce_on_sequence(Lengths{}, multiplies<>{}, number<1>{});

    static constexpr auto ordered_lengths = Lengths::reorder_new_to_old(Orders{});
    using OrderedLengthsType              = remove_cvref_t<decltype(ordered_lengths)>;
    using Decomposer = detail::index_decomposer<OrderedLengthsType, make_index_sequence<n_dim>>;

    CK_TILE_HOST_DEVICE constexpr static_ford()
    {
        static_assert(Lengths::size() > 0, "wrong! Lengths is empty");
        static_assert(Lengths::size() == Orders::size(), "wrong! inconsistent size");
    }

    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_for<0, total_size, 1>{}([&](auto linear_idx) {
            f(typename Decomposer::template decompose_reordered<linear_idx.value, Orders>{});
        });
    }
};

// Specialization for default (identity) iteration order.
// Skips reordering entirely since OrderedLengths == Lengths.
template <class Lengths>
struct static_ford<Lengths, make_index_sequence<Lengths::size()>>
{
    static constexpr index_t n_dim = Lengths::size();
    static constexpr index_t total_size =
        reduce_on_sequence(Lengths{}, multiplies<>{}, number<1>{});

    using Decomposer = detail::index_decomposer<Lengths, make_index_sequence<n_dim>>;

    CK_TILE_HOST_DEVICE constexpr static_ford()
    {
        static_assert(n_dim > 0, "wrong! Lengths is empty");
    }

    template <class F>
    CK_TILE_HOST_DEVICE constexpr void operator()(F f) const
    {
        static_for<0, total_size, 1>{}([&](auto linear_idx) {
            f(typename Decomposer::template decompose<linear_idx.value>{});
        });
    }
};

namespace detail {

template <typename Indices>
struct unpack_impl;

template <index_t... Is>
struct unpack_impl<sequence<Is...>>
{
    template <typename F, typename X>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...);
#endif
    }
};

template <typename Seq0, typename Seq1>
struct unpack2_impl;

// TODO: remove this, after properly implementing unpack that takes any number of containers
template <index_t... Is, index_t... Js>
struct unpack2_impl<sequence<Is...>, sequence<Js...>>
{
    template <typename F, typename X, typename Y>
    CK_TILE_HOST_DEVICE constexpr auto operator()(F&& f, X&& x, Y&& y) const
    {
#if 0
        return std::forward<F>(f)(std::forward<X>(x).at(number<Is>{})...,
                                  std::forward<Y>(y).at(number<Js>{})...);
#else
        return std::forward<F>(f)(std::forward<X>(x).template at<Is>()...,
                                  std::forward<Y>(y).template at<Js>()...);
#endif
    }
};

} // namespace detail

template <typename F, typename X>
CK_TILE_HOST_DEVICE constexpr auto unpack(F&& f, X&& x)
{
    using X_ = remove_reference_t<X>;
    return detail::unpack_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x));
}

// TODO: properly implement unpack that takes any number of containers
template <typename F, typename X, typename Y>
CK_TILE_HOST_DEVICE constexpr auto unpack2(F&& f, X&& x, Y&& y)
{
    using X_ = remove_reference_t<X>;
    using Y_ = remove_reference_t<Y>;
    return detail::unpack2_impl<typename arithmetic_sequence_gen<0, X_::size(), 1>::type,
                                typename arithmetic_sequence_gen<0, Y_::size(), 1>::type>{}(
        std::forward<F>(f), std::forward<X>(x), std::forward<Y>(y));
}

// z = predicate ? x : y
template <bool predicate, typename X, typename Y>
constexpr auto conditional_expr(X&& x, Y&& y)
{
    if constexpr(predicate)
    {
        return std::forward<X>(x);
    }
    else
    {
        return std::forward<Y>(y);
    }
}

} // namespace ck_tile
#pragma clang diagnostic pop
