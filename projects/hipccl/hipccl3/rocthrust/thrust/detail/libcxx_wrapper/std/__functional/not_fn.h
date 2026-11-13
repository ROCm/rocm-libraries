// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__FUNCTIONAL_NOT_FN_H
#define LIBCXX_WRAPPER_STD__FUNCTIONAL_NOT_FN_H

// TODO(libhipcxx): remove this file and replace ::internal with _THRUST_STD in rocThrust
// once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>
#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__functional/not_fn.h)
#else
#  include <thrust/detail/libcxx_wrapper/std/__cccl/attributes.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/__functional/invoke.h>
#  include <thrust/detail/type_traits.h>

#  include <type_traits>
#  include <utility>
#endif

namespace internal
{

#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename Fn>
using not_fn_t = _THRUST_STD::__not_fn_t<Fn>;
using _THRUST_STD::not_fn;

#else

namespace detail
{
template <typename Fn, typename... Args>
auto can_invoke_and_negate_test(int)
  -> decltype((void) (!::internal::detail::invoke(::std::declval<Fn>(), ::std::declval<Args>()...)),
              ::std::true_type{});

template <typename Fn, typename... Args>
auto can_invoke_and_negate_test(...) -> ::std::false_type;

template <typename Fn, typename... Args>
inline constexpr bool can_invoke_and_negate = decltype(can_invoke_and_negate_test<Fn, Args...>(0))::value;
} // namespace detail

template <typename Fn>
struct not_fn_t
{
  Fn f;

  template <typename... Args,
            bool cccl_true                                                                 = true,
            ::std::enable_if_t<::std::is_constructible_v<Fn, Args&&...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE explicit constexpr not_fn_t(Args&&... args) noexcept(
    ::std::is_nothrow_constructible_v<Fn, Args&&...>)
      : f(::std::forward<Args>(args)...)
  {}

  THRUST_EXEC_CHECK_DISABLE
  template <typename... Args,
            bool cccl_true                                                                    = true,
            ::std::enable_if_t<detail::can_invoke_and_negate<Fn&, Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto operator()(Args&&... args) & noexcept(noexcept(
    !detail::invoke(f, ::std::forward<Args>(args)...))) -> decltype(!detail::invoke(f, ::std::forward<Args>(args)...))
  {
    return !detail::invoke(f, ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                     = true,
            ::std::enable_if_t<!detail::can_invoke_and_negate<Fn&, Args...> && cccl_true, int> = 0>
  void operator()(Args&&...) & = delete;

  THRUST_EXEC_CHECK_DISABLE
  template <typename... Args,
            bool cccl_true                                                                          = true,
            ::std::enable_if_t<detail::can_invoke_and_negate<const Fn&, Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto operator()(Args&&... args) const& noexcept(noexcept(
    !detail::invoke(f, ::std::forward<Args>(args)...))) -> decltype(!detail::invoke(f, ::std::forward<Args>(args)...))
  {
    return !detail::invoke(f, ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                           = true,
            ::std::enable_if_t<!detail::can_invoke_and_negate<const Fn&, Args...> && cccl_true, int> = 0>
  void operator()(Args&&...) const& = delete;

  THRUST_EXEC_CHECK_DISABLE
  template <typename... Args,
            bool cccl_true                                                                   = true,
            ::std::enable_if_t<detail::can_invoke_and_negate<Fn, Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto
  operator()(Args&&... args) && noexcept(noexcept(!detail::invoke(::std::move(f), ::std::forward<Args>(args)...)))
    -> decltype(!detail::invoke(::std::move(f), ::std::forward<Args>(args)...))
  {
    return !detail::invoke(::std::move(f), ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                    = true,
            ::std::enable_if_t<!detail::can_invoke_and_negate<Fn, Args...> && cccl_true, int> = 0>
  void operator()(Args&&...) && = delete;

  THRUST_EXEC_CHECK_DISABLE
  template <typename... Args,
            bool cccl_true                                                                         = true,
            ::std::enable_if_t<detail::can_invoke_and_negate<const Fn, Args...> && cccl_true, int> = 0>
  inline THRUST_HOST_DEVICE constexpr auto
  operator()(Args&&... args) const&& noexcept(noexcept(!detail::invoke(::std::move(f), ::std::forward<Args>(args)...)))
    -> decltype(!detail::invoke(::std::move(f), ::std::forward<Args>(args)...))
  {
    return !detail::invoke(::std::move(f), ::std::forward<Args>(args)...);
  }

  template <typename... Args,
            bool cccl_true                                                                          = true,
            ::std::enable_if_t<!detail::can_invoke_and_negate<const Fn, Args...> && cccl_true, int> = 0>
  void operator()(Args&&...) const&& = delete;
};

template <typename Fn,
          bool cccl_true                                                                    = true,
          ::std::enable_if_t<::std::is_constructible_v<decay_t<Fn>, Fn> && cccl_true, int>  = 0,
          ::std::enable_if_t<::std::is_move_constructible_v<decay_t<Fn>> && cccl_true, int> = 0>
THRUST_NODISCARD inline THRUST_HOST_DEVICE constexpr auto not_fn(Fn&& f)
{
  return not_fn_t<decay_t<Fn>>(::std::forward<Fn>(f));
}

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_STD__FUNCTIONAL_NOT_FN_H
