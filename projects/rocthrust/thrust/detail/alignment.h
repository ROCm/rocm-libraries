/*
 *  Copyright 2024 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/// \file alignment.h
/// \brief Type-alignment utilities.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include _THRUST_STD_INCLUDE(cstddef) // For `std::size_t` and `std::max_align_t`.
#include _THRUST_STD_INCLUDE(type_traits) // For `std::alignment_of`.

#if _THRUST_HAS_DEVICE_SYSTEM_STD
#  include _THRUST_LIBCXX_INCLUDE(cmath)
#else
#  include <rocprim/detail/various.hpp>
#endif

THRUST_NAMESPACE_BEGIN
namespace detail
{
/// \p alignment_of provides the member constant `value` which is equal to the
/// alignment requirement of the type `T`, as if obtained by a C++11 `alignof`
/// expression.
///
/// It is an implementation of C++11's \p std::alignment_of.
template <typename T>
using alignment_of = _THRUST_STD::alignment_of<T>;

/// \p aligned_type provides the nested type `type`, which is a trivial
/// type whose alignment requirement is a divisor of `Align`.
///
/// The behavior is undefined if `Align` is not a power of 2.
template <_THRUST_STD::size_t Align>
struct aligned_type
{
  struct alignas(Align) type
  {};
};

/// \p max_align_t is a trivial type whose alignment requirement is at least as
/// strict (as large) as that of every scalar type.
///
/// It is an implementation of C++11's \p std::max_align_t.
using max_align_t = _THRUST_STD::max_align_t;

/// \p aligned_reinterpret_cast `reinterpret_cast`s \p u of type \p U to `void*`
/// and then `reinterpret_cast`s the result to \p T. The indirection through
/// `void*` suppresses compiler warnings when the alignment requirement of \p *u
/// is less than the alignment requirement of \p *t. The caller of
/// \p aligned_reinterpret_cast is responsible for ensuring that the alignment
/// requirements are actually satisfied.
template <typename T, typename U>
THRUST_HOST_DEVICE T aligned_reinterpret_cast(U u)
{
  return reinterpret_cast<T>(reinterpret_cast<void*>(u));
}

THRUST_HOST_DEVICE inline _THRUST_STD::size_t aligned_storage_size(_THRUST_STD::size_t n, _THRUST_STD::size_t align)
{
#if _THRUST_HAS_DEVICE_SYSTEM_STD
  return _THRUST_LIBCXX::ceil_div(n, align) * align;
#else
  return ::rocprim::detail::ceiling_div(n, align) * align;
#endif
}
} // end namespace detail
THRUST_NAMESPACE_END
