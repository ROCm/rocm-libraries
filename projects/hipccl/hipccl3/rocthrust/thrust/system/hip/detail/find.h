/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2026, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_HAS_HIP_COMPILER()
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/functional.h>
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/detail/general/temp_storage.h>
#  include <thrust/system/hip/detail/get_value.h>
#  include <thrust/system/hip/detail/util.h>
#  include <thrust/type_traits/unwrap_contiguous_iterator.h>

#  include _THRUST_LIBCXX_INCLUDE(__iterator/transform_iterator.h)
#  include _THRUST_STD_INCLUDE(__functional/not_fn.h)
#  include _THRUST_STD_INCLUDE(__iterator/distance.h)
#  include _THRUST_STD_INCLUDE(cstdint)

// rocprim include
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

// XXX forward declare to circumvent circular dependency
template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class T>
InputIt THRUST_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value);

}; // namespace hip_rocprim
THRUST_NAMESPACE_END

#  include <thrust/find.h>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace detail
{
template <typename Derived, typename InputIt, typename Size, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION Size
find_if_n_impl(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  using namespace thrust::system::hip_rocprim::temp_storage;

  hipStream_t stream = hip_rocprim::stream(policy);
  bool debug_sync    = THRUST_HIP_DEBUG_SYNC_FLAG;

  // Bind the predicate to the input iterator so that rocPRIM only loads the boolean results and
  // never materializes the input values. rocPRIM's find_first_of exits early as soon as a block
  // finds a match, so a single key that compares equal to `true` gives a single-pass find_if.
  const auto input = _THRUST_LIBCXX::make_transform_iterator(thrust::try_unwrap_contiguous_iterator(first), predicate);
  const auto keys  = rocprim::constant_iterator<bool>(true);

  size_t tmp_size = 0;
  hip_rocprim::throw_on_error(
    rocprim::find_first_of(
      nullptr,
      tmp_size,
      input,
      keys,
      static_cast<size_t*>(nullptr),
      static_cast<size_t>(num_items),
      1,
      rocprim::equal_to<bool>(),
      stream,
      debug_sync),
    "find_if: failed to get temp storage size");

  size_t storage_size = 0;
  void* ptr           = nullptr;
  void* tmp_ptr       = nullptr;
  size_t* result_ptr;

  auto l_part = make_linear_partition(make_partition(&tmp_ptr, tmp_size), ptr_aligned_array(&result_ptr, 1));

  // Calculate storage_size including alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Allocate temporary storage for both the algorithm and the result.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  ptr = static_cast<void*>(tmp.data().get());

  // Create pointers with alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Run find_if.
  hip_rocprim::throw_on_error(
    rocprim::find_first_of(
      tmp_ptr,
      tmp_size,
      input,
      keys,
      result_ptr,
      static_cast<size_t>(num_items),
      1,
      rocprim::equal_to<bool>(),
      stream,
      debug_sync),
    "find_if: failed to run algorithm");

  return static_cast<Size>(hip_rocprim::get_value(policy, result_ptr));
}
} // namespace detail

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class Size, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_n(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  if (num_items == 0)
  {
    return first;
  }

  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Size par(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
    {
      return hip_rocprim::detail::find_if_n_impl(policy, first, num_items, predicate);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static Size seq(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
    {
      return thrust::find_if(cvt_to_seq(derived_cast(policy)), first, first + num_items, predicate) - first;
    }
#  endif
  };

#  if !defined(__HIP_DEVICE_COMPILE__)
  Size result_idx = workaround::par(policy, first, num_items, predicate);
#  else
  Size result_idx = workaround::seq(policy, first, num_items, predicate);
#  endif

  return first + result_idx;
}

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return hip_rocprim::find_if_n(policy, first, _THRUST_STD::distance(first, last), predicate);
}

template <class Derived, class InputIt, class Predicate>
InputIt THRUST_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return hip_rocprim::find_if(policy, first, last, _THRUST_STD::not_fn(predicate));
}

template <class Derived, class InputIt, class T>
InputIt THRUST_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value)
{
  using thrust::placeholders::_1;

  return hip_rocprim::find_if(policy, first, last, _1 == value);
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
