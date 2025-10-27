/*
 *  Copyright 2025 NVIDIA Corporation
 *  Modifications Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/type_traits/pointer_traits.h>
#include <thrust/system/detail/bad_alloc.h>
#include <thrust/system/hip/detail/par.h>

#if !_THRUST_HAS_DEVICE_SYSTEM_STD
#  include <cstddef>
#endif

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

// If par_nosync does not have a user provided allocator attached, these
// overloads should be selected.

template <typename T>
THRUST_HOST pair<T*, _THRUST_STD::ptrdiff_t> get_temporary_buffer(par_nosync_t&, _THRUST_STD::ptrdiff_t n)
{
  void* ptr;
  hipError_t status = hipMallocAsync(&ptr, sizeof(T) * n, nullptr);

  if (status != hipSuccess)
  {
    (void) hipGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with hipMalloc.
    status = hipMalloc(&ptr, sizeof(T) * n);

    if (status != hipSuccess)
    {
      throw system::detail::bad_alloc(hip_category().message(status).c_str());
    }
  }

  return make_pair(reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer>
THRUST_HOST void return_temporary_buffer(par_nosync_t&, Pointer ptr, _THRUST_STD::ptrdiff_t)
{
  void* void_ptr = raw_pointer_cast(ptr);

  hipError_t status = hipFreeAsync(void_ptr, nullptr);

  if (status != hipSuccess)
  {
    (void) hipGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with hipFree.
    status = hipFree(void_ptr);

    if (status != hipSuccess)
    {
      throw system::detail::bad_alloc(hip_category().message(status).c_str());
    }
  }
}

template <typename T>
THRUST_HOST pair<T*, _THRUST_STD::ptrdiff_t>
get_temporary_buffer(execute_on_stream_nosync& system, _THRUST_STD::ptrdiff_t n)
{
  void* ptr;
  hipError_t status = hipMallocAsync(&ptr, sizeof(T) * n, get_stream(system));

  if (status != hipSuccess)
  {
    (void) hipGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with hipFree.
    status = hipMalloc(&ptr, sizeof(T) * n);

    if (status != hipSuccess)
    {
      throw system::detail::bad_alloc(hip_category().message(status).c_str());
    }
  }

  return make_pair(reinterpret_pointer_cast<T*>(ptr), n);
}

template <typename Pointer>
THRUST_HOST void return_temporary_buffer(execute_on_stream_nosync& system, Pointer ptr, _THRUST_STD::ptrdiff_t)
{
  void* void_ptr = raw_pointer_cast(ptr);

  hipError_t status = hipFreeAsync(void_ptr, get_stream(system));

  if (status != hipSuccess)
  {
    (void) hipGetLastError(); // Clear the CUDA global error state.

    // That didn't work. We could be somewhere where async allocation isn't
    // supported like Windows, so try again with hipMalloc.
    status = hipFree(void_ptr);

    if (status != hipSuccess)
    {
      throw system::detail::bad_alloc(hip_category().message(status).c_str());
    }
  }
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
