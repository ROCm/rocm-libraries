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

#include <cstdint>

// functions to handle memory alignment

THRUST_NAMESPACE_BEGIN
namespace detail
{
namespace util
{

template <typename T>
THRUST_HOST_DEVICE T* align_up(T* ptr, std::uintptr_t bytes)
{
  return (T*) (bytes * (((std::uintptr_t) ptr + (bytes - 1)) / bytes));
}

template <typename T>
THRUST_HOST_DEVICE T* align_down(T* ptr, std::uintptr_t bytes)
{
  return (T*) (bytes * (std::uintptr_t(ptr) / bytes));
}

template <typename T>
THRUST_HOST_DEVICE bool is_aligned(T* ptr, std::uintptr_t bytes = sizeof(T))
{
  return std::uintptr_t(ptr) % bytes == 0;
}

} // end namespace util
} // end namespace detail
THRUST_NAMESPACE_END
