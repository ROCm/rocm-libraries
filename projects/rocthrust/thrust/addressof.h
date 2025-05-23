// Copyright (c) 2018 NVIDIA Corporation
// Author: Bryce Adelstein Lelbach <brycelelbach@gmail.com>
//
// Distributed under the Boost Software License v1.0 (boost.org/LICENSE_1_0.txt)

#pragma once

#include <thrust/detail/config.h>

#include <thrust/detail/memory_wrapper.h>

THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

/*! Obtains the actual address of the object or function arg, even in presence of overloaded operator&.
 */
template <typename T>
THRUST_HOST_DEVICE T* addressof(T& arg)
{
  return reinterpret_cast<T*>(&const_cast<char&>(reinterpret_cast<const volatile char&>(arg)));
}

///////////////////////////////////////////////////////////////////////////////

THRUST_NAMESPACE_END
