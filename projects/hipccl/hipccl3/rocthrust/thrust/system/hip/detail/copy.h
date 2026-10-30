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

#include <thrust/system/hip/config.h>

#include <thrust/advance.h>
#include <thrust/system/hip/detail/cross_system.h>
#include <thrust/system/hip/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename InputIt, typename OutputIt>
THRUST_HOST_DEVICE OutputIt
copy(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIt first, InputIt last, OutputIt result);

template <class DerivedPolicy, class InputIt, class Size, class OutputIt>
THRUST_HOST_DEVICE OutputIt
copy_n(const thrust::detail::execution_policy_base<DerivedPolicy>& exec, InputIt first, Size n, OutputIt result);

namespace hip_rocprim
{

// D->D copy requires HIP compiler
template <class System, class InputIterator, class OutputIterator>
OutputIterator THRUST_HOST_DEVICE
copy(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result);

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator THRUST_HOST
copy(cross_system<System1, System2> systems, InputIterator first, InputIterator last, OutputIterator result);

template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator THRUST_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result);

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator THRUST_HOST
copy_n(cross_system<System1, System2> systems, InputIterator first, Size n, OutputIterator result);

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/temporary_array.h>
#include <thrust/system/hip/detail/par_to_seq.h>
#include <thrust/system/hip/detail/uninitialized_copy.h>
#include <thrust/system/hip/detail/util.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#if THRUST_HAS_HIP_COMPILER()
#  include <thrust/system/hip/detail/transform.h>
#endif

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
// Forward declare to work around a cyclic include, since "hip/detail/transform.h" includes this header
template <class Derived, class InputIt, class OutputIt, class TransformOp>
OutputIt THRUST_HIP_FUNCTION
transform(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, TransformOp transform_op);

namespace __copy
{
template <class H, class D, class T, class Size>
THRUST_HIP_HOST_FUNCTION void trivial_cross_system_copy_n(
  cpp::execution_policy<H>&, hip_rocprim::execution_policy<D>& device_s, T* dst, const T* src, Size n)
{
  const auto status = hip_rocprim::trivial_copy_to_device(dst, src, n, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy::trivial_device_copy H->D: failed");
}

template <class H, class D, class T, class Size>
THRUST_HIP_HOST_FUNCTION void trivial_cross_system_copy_n(
  hip_rocprim::execution_policy<D>& device_s, cpp::execution_policy<H>&, T* dst, const T* src, Size n)
{
  const auto status = hip_rocprim::trivial_copy_from_device(dst, src, n, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "trivial_device_copy D->H failed");
}

template <class H, class D, class InputIt, class Size, class OutputIt>
OutputIt THRUST_HOST non_trivial_cross_system_copy_n(
  cpp::execution_policy<H>& host_s,
  hip_rocprim::execution_policy<D>& device_s,
  InputIt first,
  Size num_items,
  OutputIt result)
{
  // copy input data into uninitialized host temp storage
  using InputTy = thrust::detail::it_value_t<InputIt>;
  thrust::detail::temporary_array<InputTy, H> temp_host(host_s, num_items);
  // FIXME(bgruber): this fails to compile until #5490 is fixed
  //_THRUST_STD::uninitialized_copy_n(first, num_items, temp_host.begin());
  for (Size idx = 0; idx != num_items; idx++)
  {
    ::new (static_cast<void*>(temp_host.data().get() + idx)) InputTy(*first);
    ++first;
  }

  // trivially copy data from host to device temp storage
  thrust::detail::temporary_array<InputTy, D> temp_device(device_s, num_items);
  const hipError_t status = hip_rocprim::trivial_copy_to_device(
    temp_device.data().get(), temp_host.data().get(), num_items, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy:: H->D: failed");

  // device->device copy
  OutputIt ret = hip_rocprim::copy_n(device_s, temp_device.data(), num_items, result);
  return ret;
}

#if THRUST_HAS_HIP_COMPILER()
// non-trivial copy D->H, only supported with HIP compiler
// because copy ctor must have  __device__ annotations
template <class D, class H, class InputIt, class Size, class OutputIt>
OutputIt THRUST_HOST non_trivial_cross_system_copy_n(
  hip_rocprim::execution_policy<D>& device_s,
  cpp::execution_policy<H>& host_s,
  InputIt first,
  Size num_items,
  OutputIt result)
{
  // copy input data into uninitialized device temp storage
  using InputTy = thrust::detail::it_value_t<InputIt>;
  thrust::detail::temporary_array<InputTy, D> temp_device(device_s, num_items);
  hip_rocprim::uninitialized_copy_n(device_s, first, num_items, temp_device.data());

  // trivially copy data from device to host temp storage
  thrust::detail::temporary_array<InputTy, H> temp_host(host_s, num_items);
  const auto status = hip_rocprim::trivial_copy_from_device(
    temp_host.data().get(), temp_device.data().get(), num_items, hip_rocprim::stream(device_s));
  hip_rocprim::throw_on_error(status, "__copy:: D->H: failed");

  // host->host copy
  OutputIt ret = thrust::copy_n(host_s, temp_host.data(), num_items, result);
  return ret;
}
#endif // THRUST_HAS_HIP_COMPILER()

template <class System1, class System2, class InputIt, class Size, class OutputIt>
OutputIt THRUST_HOST cross_system_copy_n(cross_system<System1, System2> systems, InputIt begin, Size n, OutputIt result)
{
  if (n == 0)
  {
    return result;
  }

  // FIXME(bgruber): I think this is a pessimization. We should only check if the iterator is contiguous and the value
  // types are the same, and not whether value_t<InputIt> is trivially copyable, since we memcpy the content
  // regardless in the non-trivial path, but pay for a temporary storage allocation.
  // Also, trivial relocation is probably the wrong trait here, because we usually want to copy, not relocate. This
  // matters for types like unique_ptr, which are trivially relocatable, but not trivially copyable. But then we would
  // need a cross system move algorithm ...
  if constexpr (is_indirectly_trivially_relocate_to_v<InputIt, OutputIt>)
  {
    using InputTy = thrust::detail::it_value_t<InputIt>;
    auto* dst     = reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result));
    auto* src     = reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*begin));
    trivial_cross_system_copy_n(derived_cast(systems.sys1), derived_cast(systems.sys2), dst, src, n);
    return result + n;
  }
  else
  {
    return non_trivial_cross_system_copy_n(derived_cast(systems.sys1), derived_cast(systems.sys2), begin, n, result);
  }
}

#if THRUST_HAS_HIP_COMPILER()
template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_RUNTIME_FUNCTION
device_to_device(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  // FIXME(bgruber): We should not check is_trivially_relocatable, since we do not semantically relocate, but copy (the
  // source remains valid). This is relevant for types like `unique_ptr`, which are trivially relocatable, but not
  // trivially copyable.
  if constexpr (is_indirectly_trivially_relocatable_to<InputIt, OutputIt>::value)
  {
    using InputTy = thrust::detail::it_value_t<InputIt>;
    const auto n  = _THRUST_STD::distance(first, last);
    if (n > 0)
    {
      const hipError_t status = trivial_copy_device_to_device(
        policy,
        reinterpret_cast<InputTy*>(thrust::raw_pointer_cast(&*result)),
        reinterpret_cast<InputTy const*>(thrust::raw_pointer_cast(&*first)),
        n);
      hip_rocprim::throw_on_error(status, "__copy:: D->D: failed");
    }

    return result + n;
  }
  else
  {
    return hip_rocprim::transform(policy, first, last, result, _THRUST_STD::identity{});
  }
}
#endif // THRUST_HAS_HIP_COMPILER()
} // namespace __copy

#if THRUST_HAS_HIP_COMPILER()
// D->D copy requires HIP compiler

THRUST_EXEC_CHECK_DISABLE
template <class System, class InputIterator, class OutputIterator>
OutputIterator THRUST_HOST_DEVICE
copy(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIterator
    par(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result)
    {
      return result = __copy::device_to_device(system, first, last, result);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIterator
    seq(execution_policy<System>& system, InputIterator first, InputIterator last, OutputIterator result)
    {
      return result = thrust::copy(cvt_to_seq(derived_cast(system)), first, last, result);
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(system, first, last, result);
#  else
  return workaround::seq(system, first, last, result);
#  endif
} // end copy()

THRUST_EXEC_CHECK_DISABLE
template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator THRUST_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIterator
    par(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result)
    {
      return result = __copy::device_to_device(system, first, _THRUST_STD::next(first, n), result);
    }
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static OutputIterator
    seq(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result)
    {
      return result = thrust::copy_n(cvt_to_seq(derived_cast(system)), first, n, result);
    }
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
  return workaround::par(system, first, n, result);
#  else
  return workaround::seq(system, first, n, result);
#  endif
} // end copy_n()
#endif

template <class System1, class System2, class InputIterator, class OutputIterator>
OutputIterator THRUST_HOST
copy(cross_system<System1, System2> systems, InputIterator first, InputIterator last, OutputIterator result)
{
  return __copy::cross_system_copy_n(systems, first, _THRUST_STD::distance(first, last), result);
} // end copy()

template <class System1, class System2, class InputIterator, class Size, class OutputIterator>
OutputIterator THRUST_HOST
copy_n(cross_system<System1, System2> systems, InputIterator first, Size n, OutputIterator result)
{
  return __copy::cross_system_copy_n(systems, first, n, result);
} // end copy_n()

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#include <thrust/detail/temporary_array.h>
#include <thrust/memory.h>
