// SPDX-FileCopyrightText: Copyright (c) 2008-2020, NVIDIA Corporation. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#include _THRUST_STD_INCLUDE(__type_traits/void_t.h)

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace pointer_traits_detail
{
template <typename Ptr, typename Enable = void>
struct pointer_raw_pointer_impl
{};

template <typename T>
struct pointer_raw_pointer_impl<T*>
{
  using type = T*;
};

template <typename Ptr>
struct pointer_raw_pointer_impl<Ptr, _THRUST_STD::void_t<typename Ptr::raw_pointer>>
{
  using type = typename Ptr::raw_pointer;
};
} // namespace pointer_traits_detail

template <typename T>
struct pointer_raw_pointer : pointer_traits_detail::pointer_raw_pointer_impl<T>
{};

// Check whether we are dealing with either a raw pointer or a thrust smart pointer
template <typename T, typename = void>
inline constexpr bool is_thrust_pointer_v = false;

template <typename T>
inline constexpr bool is_thrust_pointer_v<T*> = true;

template <typename T>
inline constexpr bool is_thrust_pointer_v<T, _THRUST_STD::void_t<typename T::raw_pointer>> = true;
} // namespace detail

THRUST_NAMESPACE_END
