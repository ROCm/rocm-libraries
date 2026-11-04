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
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/detail/get_value.h>
#  include <thrust/system/hip/detail/util.h>

#  include _THRUST_STD_INCLUDE(__functional/not_fn.h)
#  include _THRUST_STD_INCLUDE(__iterator/distance.h)

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
THRUST_HOST_DEVICE Size
find_if_n_impl(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  hipStream_t stream = hip_rocprim::stream(policy);

  auto call_with_adjusted_size_type = [&](auto num_items_fixed) -> Size {
    // we use the same offset type that rocPRIM uses internally for writing the result. avoids an extra kernel
    using adjusted_size_type = std::size_t;

    std::size_t tmp_size = 0;
    auto status          = ::rocprim::find_if(
      nullptr, tmp_size, first, static_cast<adjusted_size_type*>(nullptr), num_items_fixed, predicate, stream);
    hip_rocprim::throw_on_error(status, "find_if: failed to get temp storage size");

    // Allocate temporary storage for both the algorithm and the result.
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, sizeof(adjusted_size_type) + tmp_size);

    // Run find_if.
    adjusted_size_type* result_ptr = thrust::detail::aligned_reinterpret_cast<adjusted_size_type*>(tmp.data().get());
    void* tmp_ptr                  = static_cast<void*>((tmp.data() + sizeof(adjusted_size_type)).get());

    status = ::rocprim::find_if(tmp_ptr, tmp_size, first, result_ptr, num_items_fixed, predicate, stream);
    hip_rocprim::throw_on_error(status, "find_if: failed to run algorithm");

    return static_cast<Size>(hip_rocprim::get_value(policy, result_ptr));
  };

  Size result;
  THRUST_INDEX_TYPE_DISPATCH(result, call_with_adjusted_size_type, num_items, (num_items_fixed));
  return result;
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

  Size result_idx = num_items;

  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Size par(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
    {
      return thrust::hip_rocprim::detail::find_if_n_impl(policy, first, num_items, predicate);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static Size seq(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
    {
      return thrust::find_if(cvt_to_seq(derived_cast(policy)), first, first + num_items, predicate) - first;
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  result_idx = workaround::par(policy, first, num_items, predicate);
#  else
  result_idx = workaround::seq(policy, first, num_items, predicate);
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
