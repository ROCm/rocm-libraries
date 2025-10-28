// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_WRAPPER_STD__FUNCTIONAL_INVOKE_H
#define LIBCXX_WRAPPER_STD__FUNCTIONAL_INVOKE_H

// TODO(libhipcxx): remove this file once libhipcxx gets ready

#include <thrust/detail/config/libcxx.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_STD_INCLUDE(__functional/invoke.h)
#else
#  include <rocprim/type_traits_functions.hpp>

#  include <thrust/detail/libcxx_wrapper/std/__cccl/dialect.h>
#  include <thrust/detail/libcxx_wrapper/std/__cccl/execution_space.h>
#  include <thrust/detail/libcxx_wrapper/std/__type_traits/is_reference_wrapper.h>
#  include <thrust/detail/libcxx_wrapper/std/__type_traits/nat.h>
#  include <thrust/detail/type_traits.h>

#  include <any>
#  include <type_traits>
#  include <utility>
#endif

namespace internal
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD

template <typename Invokable, typename InputT, typename InitT = InputT>
using accumulator_t = _THRUST_STD::__accumulator_t<Invokable, InputT, InitT>;

#else
namespace detail
{
template <typename DecayedFp>
struct member_pointer_class_type
{};

template <typename Ret, typename ClassType>
struct member_pointer_class_type<Ret ClassType::*>
{
  using type = ClassType;
};

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet1 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value&& ::std::is_base_of<ClassT, DecayA0>::value>;

template <typename Fp, typename A0, typename DecayFp = ::internal::decay_t<Fp>, typename DecayA0 = ::internal::decay_t<A0>>
using enable_if_bullet2 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value && is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet3 =
  ::std::enable_if_t<::std::is_member_function_pointer<DecayFp>::value && !::std::is_base_of<ClassT, DecayA0>::value
                     && !is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet4 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value&& ::std::is_base_of<ClassT, DecayA0>::value>;

template <typename Fp, typename A0, typename DecayFp = ::internal::decay_t<Fp>, typename DecayA0 = ::internal::decay_t<A0>>
using enable_if_bullet5 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value && is_reference_wrapper<DecayA0>::value>;

template <typename Fp,
          typename A0,
          typename DecayFp = ::internal::decay_t<Fp>,
          typename DecayA0 = ::internal::decay_t<A0>,
          typename ClassT  = typename member_pointer_class_type<DecayFp>::type>
using enable_if_bullet6 =
  ::std::enable_if_t<::std::is_member_object_pointer<DecayFp>::value && !::std::is_base_of<ClassT, DecayA0>::value
                     && !is_reference_wrapper<DecayA0>::value>;

template <typename... Args>
inline THRUST_HOST_DEVICE nat __invoke(::std::any, Args&&... args);

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet1<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype((::std::declval<A0>().*::std::declval<Fp>())(::std::declval<Args>()...))
__invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept((static_cast<A0&&>(a0).*f)(static_cast<Args&&>(args)...)))
{
  return (static_cast<A0&&>(a0).*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet2<Fp, A0>>
inline
  THRUST_HOST_DEVICE constexpr decltype((::std::declval<A0>().get().*::std::declval<Fp>())(::std::declval<Args>()...))
  __invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept((a0.get().*f)(static_cast<Args&&>(args)...)))
{
  return (a0.get().*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename... Args, typename = enable_if_bullet3<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(((*::std::declval<A0>()).*::std::declval<Fp>())(::std::declval<Args>()...))
__invoke(Fp&& f, A0&& a0, Args&&... args) noexcept(noexcept(((*static_cast<A0&&>(a0)).*f)(static_cast<Args&&>(args)...)))
{
  return ((*static_cast<A0&&>(a0)).*f)(static_cast<Args&&>(args)...);
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet4<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<A0>().*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept(static_cast<A0&&>(a0).*f))
{
  return static_cast<A0&&>(a0).*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet5<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<A0>().get().*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept(a0.get().*f))
{
  return a0.get().*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename A0, typename = enable_if_bullet6<Fp, A0>>
inline THRUST_HOST_DEVICE constexpr decltype((*::std::declval<A0>()).*::std::declval<Fp>())
__invoke(Fp&& f, A0&& a0) noexcept(noexcept((*static_cast<A0&&>(a0)).*f))
{
  return (*static_cast<A0&&>(a0)).*f;
}

THRUST_EXEC_CHECK_DISABLE
template <typename Fp, typename... Args>
inline THRUST_HOST_DEVICE constexpr decltype(::std::declval<Fp>()(::std::declval<Args>()...))
__invoke(Fp&& f, Args&&... args) noexcept(noexcept(static_cast<Fp&&>(f)(static_cast<Args&&>(args)...)))
{
  return static_cast<Fp&&>(f)(static_cast<Args&&>(args)...);
}

template <typename Fn, typename... Args>
inline THRUST_HOST_DEVICE constexpr ::rocprim::invoke_result_t<Fn, Args...>
invoke(Fn&& f, Args&&... args) noexcept(THRUST_TRAIT(::std::is_nothrow_invocable, Fn, Args...))
{
  return ::internal::detail::__invoke(::std::forward<Fn>(f), ::std::forward<Args>(args)...);
}
} // namespace detail

template <typename Invokable, typename InputT, typename InitT = InputT>
#  if _THRUST_USE_ROCPRIM
using accumulator_t = ::rocprim::accumulator_t<Invokable, InputT, InitT>;
#  else
using accumulator_t = decay_t<::std::invoke_result_t<Invokable, InitT, InputT>>;
#  endif

#endif

} // namespace internal

#endif // LIBCXX_WRAPPER_STD__FUNCTIONAL_INVOKE_H
