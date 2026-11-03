// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// allocator_traits::rebind_alloc and allocator::rebind_traits are from libc++,
// dual licensed under the MIT and the University of Illinois Open Source
// Licenses.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/type_traits.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/iterator/iterator_traits.h>

#include _THRUST_STD_INCLUDE(__iterator/iterator_traits.h)
#include _THRUST_STD_INCLUDE(__memory/allocator_traits.h)
#include _THRUST_STD_INCLUDE(__type_traits/add_lvalue_reference.h)
#include _THRUST_STD_INCLUDE(__type_traits/conditional.h)
#include _THRUST_STD_INCLUDE(__type_traits/integral_constant.h)
#include _THRUST_STD_INCLUDE(__type_traits/type_identity.h)
#include _THRUST_STD_INCLUDE(__type_traits/void_t.h)
#include _THRUST_STD_INCLUDE(__utility/declval.h)

THRUST_NAMESPACE_BEGIN
namespace detail
{
// forward declaration for has_member_system
template <typename Alloc>
struct allocator_system;

namespace allocator_traits_detail
{
template <class T, class = void>
inline constexpr bool has_system_type = false;
template <class T>
inline constexpr bool has_system_type<T, _THRUST_STD::void_t<typename T::system_type>> = true;

template <class T>
struct nested_system_type
{
  using type = typename T::system_type;
};

template <class T, class = void>
inline constexpr bool has_member_system = false;
template <class T>
inline constexpr bool has_member_system<T, _THRUST_STD::void_t<decltype(_THRUST_STD::declval<T>().system())>> =
  _THRUST_STD::is_same_v<decltype(_THRUST_STD::declval<T>().system()), typename allocator_system<T>::type&>;

template <typename Alloc>
[[nodiscard]] _CCCL_API decltype(auto) system(Alloc& a)
{
  if constexpr (has_member_system<Alloc>)
  { // return the allocator's system
    return a.system();
  }
  else
  { // return a copy of a value-initialized system
    return typename allocator_system<Alloc>::type{};
  }
}
} // namespace allocator_traits_detail

// XXX consider moving this non-standard functionality inside allocator_traits
template <typename Alloc>
struct allocator_system
{
  // the type of the allocator's system
  using type = typename eval_if<allocator_traits_detail::has_system_type<Alloc>,
                                allocator_traits_detail::nested_system_type<Alloc>,
                                thrust::iterator_system<typename _THRUST_STD::allocator_traits<Alloc>::pointer>>::type;

  // the type that get returns
  using get_result_type =
    typename eval_if<allocator_traits_detail::has_member_system<Alloc>,
                     _THRUST_STD::add_lvalue_reference<type>,
                     _THRUST_STD::type_identity<type>>::type;

  THRUST_HOST_DEVICE inline static get_result_type get(Alloc& a)
  {
    return allocator_traits_detail::system(a);
  }
};
} // namespace detail
THRUST_NAMESPACE_END
