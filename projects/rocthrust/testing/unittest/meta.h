/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/*! \file meta.h
 *  \brief Defines template classes
 *         for metaprogramming in the
 *         unit tests.
 */

#pragma once

// This header `<thrust/detail/config/diagnostic.h>` is added only for suppressing the
// warning generated by deprecated API `thrust::device_malloc_allocator`. Once the
// API is removed, this header should also be removed.
#include <thrust/detail/config/diagnostic.h>

namespace unittest
{

// mark the absence of a type
struct null_type
{};

// this type encapsulates a list of
// types
template <typename... Ts>
struct type_list
{};

// this type provides a way of indexing
// into a type_list
template <typename List, unsigned int i>
struct get_type
{
  using type = null_type;
};

template <typename T, typename... Ts>
struct get_type<type_list<T, Ts...>, 0>
{
  using type = T;
};

template <typename T, typename... Ts, unsigned int i>
struct get_type<type_list<T, Ts...>, i>
{
  using type = typename get_type<type_list<Ts...>, i - 1>::type;
};

template <typename T, unsigned int i>
using get_type_t = typename get_type<T, i>::type;

// this type and its specialization provides a way to
// iterate over a type_list, and
// applying a unary function to each type
template <typename TypeList, template <typename> class Function, typename T, unsigned int i = 0>
struct for_each_type
{
  template <typename U>
  void operator()(U n)
  {
    // run the function on type T
    Function<T> f;
    f(n);

    // get the next type
    using next_type = typename get_type<TypeList, i + 1>::type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop(n);
  }

  void operator()(void)
  {
    // run the function on type T
    Function<T> f;
    f();

    // get the next type
    using next_type = typename get_type<TypeList, i + 1>::type;

    // recurse to i + 1
    for_each_type<TypeList, Function, next_type, i + 1> loop;
    loop();
  }
};

// terminal case: do nothing when encountering null_type
template <typename TypeList, template <typename> class Function, unsigned int i>
struct for_each_type<TypeList, Function, null_type, i>
{
  template <typename U>
  void operator()(U)
  {
    // no-op
  }

  void operator()(void)
  {
    // no-op
  }
};

// this type and its specialization instantiates
// a template by applying T to Template.
// if T == null_type, then its result is also null_type
template <template <typename> class Template, typename T>
struct ApplyTemplate1
{
  // The `thrust::device_malloc_allocator` is deprecated, considered removed and is only
  // here for compatibility, once it is removed, the `THRUST_SUPPRESS_DEPRECATED_PUSH`
  // and `THRUST_SUPPRESS_DEPRECATED_POP` should also be removed.
  THRUST_SUPPRESS_DEPRECATED_PUSH
  using type = Template<T>;
  THRUST_SUPPRESS_DEPRECATED_POP
};

template <template <typename> class Template>
struct ApplyTemplate1<Template, null_type>
{
  using type = null_type;
};

// this type and its specializations instantiates
// a template by applying T1 & T2 to Template.
// if either T1 or T2 == null_type, then its result
// is also null_type
template <template <typename, typename> class Template, typename T1, typename T2>
struct ApplyTemplate2
{
  using type = Template<T1, T2>;
};

template <template <typename, typename> class Template, typename T>
struct ApplyTemplate2<Template, T, null_type>
{
  using type = null_type;
};

template <template <typename, typename> class Template, typename T>
struct ApplyTemplate2<Template, null_type, T>
{
  using type = null_type;
};

template <template <typename, typename> class Template>
struct ApplyTemplate2<Template, null_type, null_type>
{
  using type = null_type;
};

// this type creates a new type_list by applying a Template to each of
// the Type_list's types
template <typename TypeList, template <typename> class Template>
struct transform1;

template <typename... Ts, template <typename> class Template>
struct transform1<type_list<Ts...>, Template>
{
  using type = type_list<typename ApplyTemplate1<Template, Ts>::type...>;
};

template <typename TypeList1, typename TypeList2, template <typename, typename> class Template>
struct transform2;

template <typename... T1s, typename... T2s, template <typename, typename> class Template>
struct transform2<type_list<T1s...>, type_list<T2s...>, Template>
{
  using type = type_list<typename ApplyTemplate2<Template, T1s, T2s>::type...>;
};

} // namespace unittest
