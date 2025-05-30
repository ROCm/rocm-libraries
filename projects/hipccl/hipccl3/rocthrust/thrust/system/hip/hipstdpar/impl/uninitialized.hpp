// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/*
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

/*! \file thrust/system/hip/hipstdpar/impl/uninitialized.hpp
 *  \brief <tt>Operations on unitialized memory</tt> implementation detail header for HIPSTDPAR.
 */

#pragma once

#if defined(__HIPSTDPAR__)

#  include <thrust/execution_policy.h>
#  include <thrust/for_each.h>
#  include <thrust/uninitialized_copy.h>
#  include <thrust/uninitialized_fill.h>

#  include <algorithm>
#  include <execution>
#  include <utility>

#  include "hipstd.hpp"

namespace std
{
// BEGIN UNINITIALIZED_COPY
template <typename I, typename O, enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
{
  return ::thrust::uninitialized_copy(::thrust::device, fi, li, fo);
}

template <typename I, typename O, enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_copy(execution::parallel_unsequenced_policy, I fi, I li, O fo)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category,
                                          typename iterator_traits<O>::iterator_category>();

  return ::std::uninitialized_copy(::std::execution::par, fi, li, fo);
}
// END UNINITIALIZED_COPY

// BEGIN UNINITIALIZED_COPY_N
template <typename I, typename N, typename O, enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_copy_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
{
  return ::thrust::uninitialized_copy_n(::thrust::device, fi, n, fo);
}

template <typename I, typename N, typename O, enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_copy_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category,
                                          typename iterator_traits<O>::iterator_category>();

  return ::std::uninitialized_copy_n(::std::execution::par, fi, n, fo);
}
// END UNINITIALIZED_COPY_N

// BEGIN UNINITIALIZED_FILL
template <typename I, typename T, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_fill(execution::parallel_unsequenced_policy, I f, I l, const T& x)
{
  return ::thrust::uninitialized_fill(::thrust::device, f, l, x);
}

template <typename I, typename T, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_fill(execution::parallel_unsequenced_policy, I f, I l, const T& x)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::offload_category>();

  return ::std::uninitialized_fill(::std::execution::par, f, l, x);
}
// END UNINITIALIZED_FILL

// BEGIN UNINITIALIZED_FILL_N
template <typename I, typename N, typename T, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_fill(execution::parallel_unsequenced_policy, I f, N n, const T& x)
{
  return ::thrust::uninitialized_fill_n(::thrust::device, f, n, x);
}

template <typename I, typename N, typename T, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_fill(execution::parallel_unsequenced_policy, I f, N n, const T& x)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::uninitialized_fill_n(::std::execution::par, f, n, x);
}
// END UNINITIALIZED_FILL_N

// BEGIN UNINITIALIZED_MOVE
template <typename I, typename O, enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_move(execution::parallel_unsequenced_policy, I fi, I li, O fo)
{
  return ::thrust::uninitialized_copy(::thrust::device, make_move_iterator(fi), make_move_iterator(li), fo);
}

template <typename I, typename O, enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_move(execution::parallel_unsequenced_policy, I fi, I li, O fo)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category,
                                          typename iterator_traits<O>::iterator_category>();

  return ::std::uninitialized_move(::std::execution::par, fi, li, fo);
}
// END UNINITIALIZED_MOVE

// BEGIN UNINITIALIZED_MOVE_N
template <typename I, typename N, typename O, enable_if_t<::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_move_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
{
  return ::thrust::uninitialized_copy_n(::thrust::device, make_move_iterator(fi), n, fo);
}

template <typename I, typename N, typename O, enable_if_t<!::hipstd::is_offloadable_iterator<I, O>()>* = nullptr>
inline O uninitialized_move_n(execution::parallel_unsequenced_policy, I fi, N n, O fo)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category,
                                          typename iterator_traits<O>::iterator_category>();

  return ::std::uninitialized_move_n(::std::execution::par, fi, n, fo);
}
// END UNINITIALIZED_MOVE_N

// BEGIN UNINITIALIZED_DEFAULT_CONSTRUCT
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_default_construct(execution::parallel_unsequenced_policy, I f, I l)
{
  ::thrust::for_each(::thrust::device, f, l, [](auto& x) {
    auto p = const_cast<void*>(static_cast<const volatile void*>((addressof(x))));
    ::new (p) typename iterator_traits<I>::value_type;
  });
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_default_construct(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::uninitialized_default_construct(::std::execution::par, f, l);
}
// END UNINITIALIZED_DEFAULT_CONSTRUCT

// BEGIN UNINITIALIZED_DEFAULT_CONSTRUCT_N
template <typename I, typename N, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_default_construct_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::thrust::for_each_n(::thrust::device, f, n, [](auto& x) {
    auto p = const_cast<void*>(static_cast<const volatile void*>((addressof(x))));
    ::new (p) typename iterator_traits<I>::value_type;
  });
}

template <typename I, typename N, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_default_construct_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::uninitialized_default_construct_n(::std::execution::par, f, n);
}
// END UNINITIALIZED_DEFAULT_CONSTRUCT_N

// BEGIN UNINITIALIZED_VALUE_CONSTRUCT
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_value_construct(execution::parallel_unsequenced_policy, I f, I l)
{
  ::thrust::for_each(::thrust::device, f, l, [](auto& x) {
    auto p = const_cast<void*>(static_cast<const volatile void*>((addressof(x))));
    ::new (p) typename iterator_traits<I>::value_type{};
  });
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_value_construct(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::uninitialized_value_construct(::std::execution::par, f, l);
}
// END UNINITIALIZED_VALUE_CONSTRUCT

// BEGIN UNINITIALIZED_VALUE_CONSTRUCT_N
template <typename I, typename N, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_value_construct_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::thrust::for_each_n(::thrust::device, f, n, [](auto& x) {
    auto p = const_cast<void*>(static_cast<const volatile void*>((addressof(x))));
    ::new (p) typename iterator_traits<I>::value_type{};
  });
}

template <typename I, typename N, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void uninitialized_value_construct_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::uninitialized_value_construct_n(::std::execution::par, f, n);
}
// END UNINITIALIZED_VALUE_CONSTRUCT_N

// BEGIN DESTROY
template <typename I, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void destroy(execution::parallel_unsequenced_policy, I f, I l)
{
  ::thrust::for_each(f, l, [](auto& x) {
    destroy_at(addressof(x));
  });
}

template <typename I, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void destroy(execution::parallel_unsequenced_policy, I f, I l)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::destroy(::std::execution::par, f, l);
}
// END DESTROY

// BEGIN DESTROY_N
template <typename I, typename N, enable_if_t<::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void destroy_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::thrust::for_each_n(f, n, [](auto& x) {
    destroy_at(addressof(x));
  });
}

template <typename I, typename N, enable_if_t<!::hipstd::is_offloadable_iterator<I>()>* = nullptr>
inline void destroy_n(execution::parallel_unsequenced_policy, I f, N n)
{
  ::hipstd::unsupported_iterator_category<typename iterator_traits<I>::iterator_category>();

  return ::std::destroy_n(::std::execution::par, f, n);
}
// END DESTROY_N
} // namespace std
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__
