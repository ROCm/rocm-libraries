// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator/allocator_system.h>
#include <thrust/detail/allocator/destroy_range.h>
#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/for_each.h>

#include _THRUST_STD_INCLUDE(__host_stdlib/memory)
#include _THRUST_STD_INCLUDE(__memory/allocator_traits.h)

THRUST_NAMESPACE_BEGIN
namespace detail
{
// destroy_range has three cases:
// if Allocator has an effectful member function destroy:
//   1. destroy via the allocator
// else
//   2. if T has a non-trivial destructor, destroy the range without using the allocator
//   3. if T has a trivial destructor, do a no-op

template <typename Allocator, typename T>
inline constexpr bool has_effectful_member_destroy = _THRUST_STD::__has_destroy<Allocator, T*>;

// std::allocator::destroy's only effect is to invoke its argument's destructor
template <typename U, typename T>
inline constexpr bool has_effectful_member_destroy<std::allocator<U>, T> = false;

template <typename Allocator>
struct destroy_via_allocator
{
  Allocator& a;

  template <typename T>
  THRUST_HOST_DEVICE void operator()(T& x) noexcept
  {
    _THRUST_STD::allocator_traits<Allocator>::destroy(a, &x);
  }
};

// we must prepare for His coming
struct gozer
{
  THRUST_EXEC_CHECK_DISABLE
  template <typename T>
  inline THRUST_HOST_DEVICE void operator()(T& x) noexcept
  {
    x.~T();
  }
};

template <typename Allocator, typename Pointer, typename Size>
THRUST_HOST_DEVICE void
destroy_range([[maybe_unused]] Allocator& a, [[maybe_unused]] Pointer p, [[maybe_unused]] Size n) noexcept
{
  using pe_t = typename pointer_element<Pointer>::type;

  // case 1: destroy via allocator
  if constexpr (has_effectful_member_destroy<Allocator, pe_t>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, destroy_via_allocator<Allocator>{a});
  }
  // case 2: destroy without the allocator
  else if constexpr (!_THRUST_STD::is_trivially_destructible_v<pe_t>)
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), p, n, gozer());
  }
  // case 3: Allocator has no member function "destroy", and T has a trivial destructor, nothing to be done
}
} // namespace detail
THRUST_NAMESPACE_END
