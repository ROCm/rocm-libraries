/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
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

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP
#  include <thrust/system/hip/config.h>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/system/hip/detail/general/temp_storage.h>
#  include <thrust/system/hip/detail/par_to_seq.h>
#  include <thrust/system/hip/detail/util.h>

// rocPRIM includes
#  include <rocprim/rocprim.hpp>

#  include <cstdint>

THRUST_NAMESPACE_BEGIN

// XXX declare generic copy_if interface
// to avoid circular dependency from thrust/copy.h
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
OutputIterator THRUST_HOST_DEVICE
copy_if(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
        InputIterator first,
        InputIterator last,
        OutputIterator result,
        Predicate pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
OutputIterator THRUST_HOST_DEVICE copy_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred);

namespace hip_rocprim
{
namespace __copy_if
{
template <unsigned int ItemsPerThread, typename InputIt, typename BoolIt, typename IntIt, typename OutputIt>
ROCPRIM_KERNEL void copy_if_kernel(InputIt first, BoolIt flagsFirst, IntIt posFirst, const size_t size, OutputIt output)
{
  const size_t baseIdx = (blockIdx.x * blockDim.x + threadIdx.x) * ItemsPerThread;

  for (size_t i = 0; i < ItemsPerThread; ++i)
  {
    const size_t index = baseIdx + i;
    if (index < size)
    {
      if (flagsFirst[index])
      {
        output[posFirst[index] - 1] = first[index];
      }
    }
  }
}

template <typename Derived, typename InputIt, typename OutputIt, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION auto
copy_if(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, Predicate predicate)
  -> std::enable_if_t<sizeof(typename std::iterator_traits<InputIt>::value_type) < 512, OutputIt>
{
  using namespace thrust::system::hip_rocprim::temp_storage;
  using size_type = typename iterator_traits<InputIt>::difference_type;

  size_type num_items       = thrust::distance(first, last);
  size_t temp_storage_bytes = 0;
  hipStream_t stream        = hip_rocprim::stream(policy);
  bool debug_sync           = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items == 0)
  {
    return output;
  }

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::select(
      nullptr,
      temp_storage_bytes,
      first,
      output,
      static_cast<size_type*>(nullptr),
      num_items,
      predicate,
      stream,
      debug_sync),
    "copy_if failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;
  size_type* d_num_selected_out;

  auto l_part =
    make_linear_partition(make_partition(&temp_stor, temp_storage_bytes), ptr_aligned_array(&d_num_selected_out, 1));

  // Calculate storage_size including alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  ptr = static_cast<void*>(tmp.data().get());

  // Create pointers with alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  hip_rocprim::throw_on_error(
    rocprim::select(
      ptr, temp_storage_bytes, first, output, d_num_selected_out, num_items, predicate, stream, debug_sync),
    "copy_if failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  return output + num_selected;
}

template <typename Derived, typename InputIt, typename OutputIt, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION auto
copy_if(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, Predicate predicate)
  -> std::enable_if_t<!(sizeof(typename std::iterator_traits<InputIt>::value_type) < 512), OutputIt>
{
  using namespace thrust::system::hip_rocprim::temp_storage;
  using size_type = typename iterator_traits<InputIt>::difference_type;

  size_type num_items = thrust::distance(first, last);
  hipStream_t stream  = hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  thrust::detail::temporary_array<std::uint8_t, Derived> flags(policy, num_items);

  hip_rocprim::throw_on_error(
    rocprim::transform(
      first,
      flags.begin(),
      num_items,
      [predicate] __host__ __device__(auto const& val) {
        return predicate(val) ? 1 : 0;
      },
      stream,
      debug_sync),
    "copy_if failed on transform");

  thrust::detail::temporary_array<std::uint32_t, Derived> pos(policy, num_items);

  thrust::inclusive_scan(policy, flags.begin(), flags.end(), pos.begin());

  constexpr static size_t items_per_thread  = 16;
  constexpr static size_t threads_per_block = 256;
  const size_t block_size                   = std::ceil(static_cast<float>(num_items) / 16 / threads_per_block);

  copy_if_kernel<items_per_thread>
    <<<block_size, threads_per_block>>>(first, flags.begin(), pos.begin(), num_items, output);

  return output + get_value(policy, &pos[num_items - 1]);
}

template <typename Derived, typename InputIt, typename StencilIt, typename OutputIt, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION OutputIt copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt output,
  Predicate predicate)
{
  using namespace thrust::system::hip_rocprim::temp_storage;
  using size_type = typename iterator_traits<InputIt>::difference_type;

  size_type num_items       = thrust::distance(first, last);
  size_t temp_storage_bytes = 0;
  hipStream_t stream        = hip_rocprim::stream(policy);
  bool debug_sync           = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items == 0)
  {
    return output;
  }

  auto flags = thrust::make_transform_iterator(stencil, predicate);

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::select(
      nullptr, temp_storage_bytes, first, flags, output, static_cast<size_type*>(nullptr), num_items, stream, debug_sync),
    "copy_if failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;
  size_type* d_num_selected_out;

  auto l_part =
    make_linear_partition(make_partition(&temp_stor, temp_storage_bytes), ptr_aligned_array(&d_num_selected_out, 1));

  // Calculate storage_size including alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  ptr = static_cast<void*>(tmp.data().get());

  // Create pointers with alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  hip_rocprim::throw_on_error(
    rocprim::select(ptr, temp_storage_bytes, first, flags, output, d_num_selected_out, num_items, stream, debug_sync),
    "copy_if failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  return output + num_selected;
}

} // namespace __copy_if

//-------------------------
// Thrust API entry points
//-------------------------

template <class Derived, class InputIterator, class OutputIterator, class Predicate>
OutputIterator THRUST_HIP_FUNCTION copy_if(
  execution_policy<Derived>& policy, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIterator par(
      execution_policy<Derived>& policy, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
    {
      return __copy_if::copy_if(policy, first, last, result, pred);
    }
    THRUST_DEVICE static OutputIterator seq(
      execution_policy<Derived>& policy, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
    {
      return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, result, pred);
    }
  };

#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, result, pred);
#  else
  return workaround::seq(policy, first, last, result, pred);
#  endif

} // func copy_if

template <class Derived, class InputIterator, class StencilIterator, class OutputIterator, class Predicate>
OutputIterator THRUST_HIP_FUNCTION copy_if(
  execution_policy<Derived>& policy,
  InputIterator first,
  InputIterator last,
  StencilIterator stencil,
  OutputIterator result,
  Predicate pred)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIterator
    par(execution_policy<Derived>& policy,
        InputIterator first,
        InputIterator last,
        StencilIterator stencil,
        OutputIterator result,
        Predicate pred)
    {
      return __copy_if::copy_if(policy, first, last, stencil, result, pred);
    }
    THRUST_DEVICE static OutputIterator
    seq(execution_policy<Derived>& policy,
        InputIterator first,
        InputIterator last,
        StencilIterator stencil,
        OutputIterator result,
        Predicate pred)
    {
      return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, stencil, result, pred);
    }
  };

#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, result, pred);
#  else
  return workaround::seq(policy, first, last, stencil, result, pred);
#  endif
} // func copy_if

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#  include <thrust/copy.h>
#endif
