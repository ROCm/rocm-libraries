// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

// Minimal type traits for kernels compiled with hipRTC. Runtime-only ROCm installs (notably on
// Windows) do not ship the host STL headers, so kernel sources must not include <type_traits>.

namespace hip_kernel_provider::traits
{

template <typename T, T V>
struct integral_constant
{
    static constexpr T value = V;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template <typename A, typename B>
struct is_same : false_type
{
};

template <typename A>
struct is_same<A, A> : true_type
{
};

template <typename A, typename B>
inline constexpr bool is_same_v = is_same<A, B>::value;

template <bool Cond, typename T, typename F>
struct conditional
{
    using type = T;
};

template <typename T, typename F>
struct conditional<false, T, F>
{
    using type = F;
};

template <bool Cond, typename T, typename F>
using conditional_t = typename conditional<Cond, T, F>::type;

template <bool Cond, typename T = void>
struct enable_if
{
};

template <typename T>
struct enable_if<true, T>
{
    using type = T;
};

template <bool Cond, typename T = void>
using enable_if_t = typename enable_if<Cond, T>::type;

} // namespace hip_kernel_provider::traits
