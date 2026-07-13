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

/*! \file thrust/system/hip/hipstdpar/impl/hipstd.hpp
 *  \brief hipstd utilities implementation detail header for HIPSTDPAR.
 */

#ifndef THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP
#define THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP

#pragma once

#if defined(__HIPSTDPAR__)

#  include <cstddef>
#  include <cstdio>
#  include <cstdlib>
#  include <iterator>
#  include <mutex>
#  include <new>
#  include <type_traits>
#  include <utility>

#  include <hip/hip_runtime_api.h>
#  include <thrust/system/hip/detail/util.h>

namespace hipstd
{
// Emits a one-time stderr warning when host-allocated memory may not be directly
// accessible on the device and -hipstdpar-interpose-alloc was not used.
//
// On Linux: warns when the current GPU is XNACK-capable but XNACK is disabled at
//   runtime (gcnArchName contains "xnack-").  No warning if XNACK is on ("xnack+")
//   or the GPU does not support XNACK at all.
// On Windows: XNACK is never available, so warns unconditionally — without
//   interpose-alloc, host memory is never device-accessible on this platform.
//
// Suppressed on both platforms when __HIPSTDPAR_INTERPOSE_ALLOC__ or
// __HIPSTDPAR_INTERPOSE_ALLOC_V1__ is defined, since interpose-alloc replaces
// allocations with hipMallocManaged, making XNACK unnecessary.
inline void warn_if_no_xnack()
{
#  if !defined(__HIPSTDPAR_INTERPOSE_ALLOC__) && !defined(__HIPSTDPAR_INTERPOSE_ALLOC_V1__)
  static ::std::once_flag xnack_flag_;
  ::std::call_once(xnack_flag_, [] {
#    if defined(__linux__)
    int device = 0;
    if (::hipGetDevice(&device) != hipSuccess)
    {
      return;
    }
    ::hipDeviceProp_t prop{};
    if (::hipGetDeviceProperties(&prop, device) != hipSuccess)
    {
      return;
    }
    // gcnArchName contains "xnack+" when XNACK is enabled and "xnack-" when the
    // GPU supports XNACK but it is currently disabled.  Warn only in the latter case.
    const char* needle = "xnack-";
    for (const char* arch = prop.gcnArchName; *arch != '\0'; ++arch)
    {
      const char* a = arch;
      const char* n = needle;
      while (*n != '\0' && *a == *n)
      {
        ++a;
        ++n;
      }
      if (*n == '\0')
      {
        ::std::fputs(
          "hipstdpar warning: the current GPU is XNACK-capable but XNACK is disabled. "
          "Host-allocated memory may not be directly accessible on the device. "
          "Enable XNACK (e.g. set HSA_XNACK=1, or add amdgpu.xnack=1 to the kernel "
          "command line) or recompile with -hipstdpar-interpose-alloc to suppress "
          "this warning.\n",
          stderr);
        return;
      }
    }
#    else // Windows
    ::std::fputs(
      "hipstdpar warning: XNACK is not supported on this platform. "
      "Host-allocated memory is not directly accessible on the device. "
      "Recompile with -hipstdpar-interpose-alloc to suppress this warning.\n",
      stderr);
#    endif // Windows
  });
#  endif // !__HIPSTDPAR_INTERPOSE_ALLOC__ && !__HIPSTDPAR_INTERPOSE_ALLOC_V1__
}

template <typename... Cs>
inline constexpr bool is_offloadable_callable() noexcept
{
  return std::conjunction_v<std::negation<std::is_pointer<Cs>>..., std::negation<std::is_member_function_pointer<Cs>>...>;
}

template <typename I, typename = void>
struct Is_offloadable_iterator : std::false_type
{};
template <typename I>
struct Is_offloadable_iterator<I,
                               std::void_t<decltype(std::declval<I>() < std::declval<I>()),
                                           decltype(std::declval<I&>() += std::declval<std::ptrdiff_t>()),
                                           decltype(std::declval<I>() + std::declval<std::ptrdiff_t>()),
                                           decltype(std::declval<I>()[std::declval<std::ptrdiff_t>()]),
                                           decltype(*std::declval<I>())>> : std::true_type
{};

template <typename... Is>
inline constexpr bool is_offloadable_iterator() noexcept
{
#  if defined(__cpp_lib_concepts)
  return (... && std::random_access_iterator<Is>);
#  else
  return std::conjunction_v<Is_offloadable_iterator<Is>...>;
#  endif
}

template <typename... Cs>
inline constexpr __attribute__((diagnose_if(
  true,
  "HIP Standard Parallelism does not support passing pointers to "
  "function as callable arguments, execution will not be "
  "offloaded.",
  "warning"))) void
unsupported_callable_type() noexcept
{}

template <typename... Is>
inline constexpr
  __attribute__((diagnose_if(true,
                             "HIP Standard Parallelism requires random access iterators, "
                             "execution will not be offloaded.",
                             "warning"))) void
  unsupported_iterator_category() noexcept
{}

namespace detail
{

// Ensures a non-trivially-destructible callable outlives any asynchronous
// kernel that references it.  The callable is placement-new'd into GPU-
// accessible memory whose lifetime is controlled by this guard.
//
//   * When --hipstdpar-interpose-alloc is active (no XNACK / no transparent
//     paging) hipMallocManaged is used so that both host and device can
//     access the callable.
//
//   * Otherwise hipMalloc is used, which is faster; XNACK guarantees that
//     the host can still access device memory directly.
template <typename Fn>
class device_callable_guard
{
public:
  explicit device_callable_guard(Fn&& fn)
  {
#  if defined(__HIPSTDPAR_INTERPOSE_ALLOC__) || defined(__HIPSTDPAR_INTERPOSE_ALLOC_V1__)
    hipError_t status = ::hipMallocManaged(reinterpret_cast<void**>(&fn_ptr_), sizeof(Fn));
#  else
    hipError_t status = ::hipMalloc(reinterpret_cast<void**>(&fn_ptr_), sizeof(Fn));
#  endif
    ::thrust::hip_rocprim::throw_on_error(status, "hipstdpar: failed to allocate device callable");
    ::new (static_cast<void*>(fn_ptr_)) Fn(::std::move(fn));
  }

  device_callable_guard(const device_callable_guard&)            = delete;
  device_callable_guard& operator=(const device_callable_guard&) = delete;

  ~device_callable_guard()
  {
    if (fn_ptr_ != nullptr)
    {
      fn_ptr_->~Fn();
      (void) ::hipFree(fn_ptr_);
    }
  }

  Fn* get() const noexcept
  {
    return fn_ptr_;
  }

  void destroy_and_free()
  {
    if (fn_ptr_ == nullptr)
    {
      return;
    }
    fn_ptr_->~Fn();
    hipError_t status = ::hipFree(fn_ptr_);
    fn_ptr_           = nullptr;
    ::thrust::hip_rocprim::throw_on_error(status, "hipstdpar: failed to free device callable");
  }

private:
  Fn* fn_ptr_ = nullptr;
};

template <typename Fn>
struct callable_proxy
{
  Fn* fn_ptr;

  template <typename... Args>
  THRUST_HIP_FUNCTION void operator()(Args&&... args) const
  {
    (*fn_ptr)(::std::forward<Args>(args)...);
  }
};

} // namespace detail
} // namespace hipstd
#else // __HIPSTDPAR__
#  error "__HIPSTDPAR__ should be defined. Please use the '--hipstdpar' compile option."
#endif // __HIPSTDPAR__

#endif // THRUST_SYSTEM_HIP_HIPSTDPAR_HIPSTD_HPP
