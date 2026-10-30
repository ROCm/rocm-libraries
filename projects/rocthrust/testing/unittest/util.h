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

#pragma once

#include <thrust/detail/type_traits.h>
#include <thrust/extrema.h>
#include <thrust/limits.h>

#include <iostream>
#include <string>
#include <typeinfo>
#if !_THRUST_HAS_DEVICE_SYSTEM_STD
// Use rocprim::numeric_limits if thrust/detail/type_traits.h uses rocprim::arithmetic
#  include <limits>
#  include <type_traits>
#endif

#include <unittest/system.h>

namespace unittest
{

template <typename T>
std::string type_name()
{
  return demangle(typeid(T).name());
} // end type_name()

// Use this with counting_iterator to avoid generating a range larger than we
// can represent.
template <typename T>
typename THRUST_NS_QUALIFIER::detail::disable_if<_THRUST_STD::is_floating_point<T>::value, T>::type
truncate_to_max_representable(std::size_t n)
{
  // Use rocprim::numeric_limits if thrust/detail/type_traits.h uses rocprim::arithmetic
  return static_cast<T>(
    THRUST_NS_QUALIFIER::min<std::size_t>(n, static_cast<std::size_t>(_THRUST_STD::numeric_limits<T>::max())));
}

// TODO: This probably won't work for `half`.
template <typename T>
typename _THRUST_STD::enable_if_t<_THRUST_STD::is_floating_point<T>::value, T>
truncate_to_max_representable(std::size_t n)
{
  // Use rocprim::numeric_limits if thrust/detail/type_traits.h uses rocprim::arithmetic
  return THRUST_NS_QUALIFIER::min<T>(static_cast<T>(n), _THRUST_STD::numeric_limits<T>::max());
}

} // namespace unittest

template <typename Iterator>
void PRINT(Iterator first, Iterator last)
{
  size_t n = 0;
  for (Iterator i = first; i != last; i++, n++)
  {
    std::cout << ">>> [" << n << "] = " << *i << std::endl;
  }
}

template <typename Container>
void PRINT(const Container& c)
{
  PRINT(c.begin(), c.end());
}

template <size_t N>
void PRINT(const char (&c)[N])
{
  std::cout << std::string(c, c + N) << std::endl;
}
