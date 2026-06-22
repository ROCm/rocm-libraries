// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>

namespace ck {

namespace detail {
template <typename T1, typename T2>
struct tuple_cat_types;
template <typename... T1s, typename... T2s>
struct tuple_cat_types<std::tuple<T1s...>, std::tuple<T2s...>>
{
    using type = std::tuple<T1s..., T2s...>;
};
} // namespace detail

template <typename T1, typename T2>
using tuple_cat_t = typename detail::tuple_cat_types<T1, T2>::type;

} // namespace ck
