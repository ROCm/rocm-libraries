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

#pragma once

#include <thrust/detail/config.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

// a type which may be assigned any other type
struct any_assign
{
  inline THRUST_HOST_DEVICE any_assign() {}

  template <typename T>
  inline THRUST_HOST_DEVICE any_assign(T)
  {}

  template <typename T>
  inline THRUST_HOST_DEVICE any_assign& operator=(T)
  {
    if (0)
    {
      // trick the compiler into silencing "warning: this expression has no effect"
      int* x = 0;
      *x     = 13;
    } // end if

    return *this;
  }
};

} // namespace detail
THRUST_NAMESPACE_END
