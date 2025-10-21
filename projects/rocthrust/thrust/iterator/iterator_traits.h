/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/iterator/iterator_traits.h
 *  \brief Traits and metafunctions for reasoning about the traits of iterators
 */

/*
 * (C) Copyright David Abrahams 2003.
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying NOTICE file for the complete license)
 *
 * For more information, see http://www.boost.org
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/detail/device_system_tag.h>
#include <thrust/iterator/detail/iterator_category_to_system.h>
#include <thrust/iterator/detail/iterator_category_to_traversal.h>
#include <thrust/iterator/iterator_categories.h>

#if _THRUST_HAS_DEVICE_SYSTEM_STD
// clang-format off
#  include _THRUST_STD_INCLUDE(__type_traits/void_t.h)
// clang-format on
#else
#  include <type_traits>
#endif

#if THRUST_COMPILER(NVRTC)
#  include _THRUST_STD_INCLUDE(iterator)
#else // THRUST_COMPILER(NVRTC)
#  include <iterator>
#endif // THRUST_COMPILER(NVRTC)

THRUST_NAMESPACE_BEGIN

//! \p iterator_traits is a type trait class that provides a uniform interface for querying the properties of iterators
//! at compile-time.
template <typename T>
struct iterator_traits
    :
#if THRUST_COMPILER(NVRTC)
    ::cuda
#endif // THRUST_COMPILER(NVRTC)
    ::std::iterator_traits<T>
{};

// value

template <typename Iterator>
struct iterator_value
{
#ifndef THRUST_DOXYGEN_INVOKED
  using type = typename iterator_traits<Iterator>::value_type;
#endif
};

template <typename Iterator>
using iterator_value_t = typename iterator_value<Iterator>::type;

// pointer

template <typename Iterator>
struct iterator_pointer
{
#ifndef THRUST_DOXYGEN_INVOKED
  using type = typename iterator_traits<Iterator>::pointer;
#endif
};
template <typename Iterator>
using iterator_pointer_t = typename iterator_pointer<Iterator>::type;

// reference

template <typename Iterator>
struct iterator_reference
{
#ifndef THRUST_DOXYGEN_INVOKED
  using type = typename iterator_traits<Iterator>::reference;
#endif
};

template <typename Iterator>
using iterator_reference_t = typename iterator_reference<Iterator>::type;

// difference

template <typename Iterator>
struct iterator_difference
{
#ifndef THRUST_DOXYGEN_INVOKED
  using type = typename iterator_traits<Iterator>::difference_type;
#endif
};

template <typename Iterator>
using iterator_difference_t = typename iterator_difference<Iterator>::type;

// traversal

template <typename Iterator>
struct iterator_traversal
    : detail::iterator_category_to_traversal<typename iterator_traits<Iterator>::iterator_category>
{};

#ifndef THRUST_DOXYGEN_INVOKED
template <typename Iterator>
using iterator_traversal_t = typename iterator_traversal<Iterator>::type;
#endif

// system

namespace detail
{
template <typename Iterator, typename = void>
struct iterator_system_impl
{};

template <typename Iterator>
struct iterator_system_impl<Iterator, _THRUST_STD::void_t<typename iterator_traits<Iterator>::iterator_category>>
    : iterator_category_to_system<typename iterator_traits<Iterator>::iterator_category>
{};
} // namespace detail

template <typename Iterator>
struct iterator_system : detail::iterator_system_impl<Iterator>
{};

// specialize iterator_system for void *, which has no category
template <>
struct iterator_system<void*> : iterator_system<int*>
{};

template <>
struct iterator_system<const void*> : iterator_system<const int*>
{};

#ifndef THRUST_DOXYGEN_INVOKED
template <typename Iterator>
using iterator_system_t = typename iterator_system<Iterator>::type;
#endif

THRUST_NAMESPACE_END

#include <thrust/iterator/detail/iterator_traversal_tags.h>
