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

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if THRUST_DEVICE_COMPILER == THRUST_DEVICE_COMPILER_HIP

#  include <thrust/system/hip/config.h>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/pair.h>
#  include <thrust/partition.h>
#  include <thrust/system/hip/detail/find.h>
#  include <thrust/system/hip/detail/general/temp_storage.h>
#  include <thrust/system/hip/detail/par_to_seq.h>
#  include <thrust/system/hip/detail/reverse.h>
#  include <thrust/system/hip/detail/uninitialized_copy.h>
#  include <thrust/system/hip/detail/util.h>

#  include <cstdint>

// rocprim include
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{

namespace detail
{

template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
THRUST_HIP_RUNTIME_FUNCTION pair<SelectedOutIt, RejectedOutIt> partition(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  using size_type  = typename iterator_traits<InputIt>::difference_type;
  using value_type = typename iterator_traits<InputIt>::value_type;
  using namespace thrust::system::hip_rocprim::temp_storage;

  size_t temp_storage_bytes     = 0;
  value_type* d_partition_out   = nullptr;
  size_type* d_num_selected_out = nullptr;
  size_type num_items           = static_cast<size_type>(thrust::distance(first, last));
  hipStream_t stream            = hip_rocprim::stream(policy);
  bool debug_sync               = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items <= 0)
  {
    return thrust::make_pair(selected_result, rejected_result);
  }

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::partition(
      nullptr, temp_storage_bytes, first, d_partition_out, d_num_selected_out, num_items, predicate, stream, debug_sync),
    "partition failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;

  auto l_part = make_linear_partition(
    make_partition(&temp_stor, temp_storage_bytes),
    ptr_aligned_array(&d_num_selected_out, 1),
    ptr_aligned_array(&d_partition_out, num_items));

  // Calculate storage_size including alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  ptr = static_cast<void*>(tmp.data().get());

  // Create pointers with alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  hip_rocprim::throw_on_error(
    rocprim::partition(
      ptr, temp_storage_bytes, first, d_partition_out, d_num_selected_out, num_items, predicate, stream, debug_sync),
    "partition failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  thrust::copy_n(policy, d_partition_out, num_items, selected_result);

  return thrust::make_pair(selected_result + num_selected, rejected_result);
}

template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HIP_RUNTIME_FUNCTION partition(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  using size_type  = typename iterator_traits<InputIt>::difference_type;
  using value_type = typename iterator_traits<InputIt>::value_type;
  using namespace thrust::system::hip_rocprim::temp_storage;

  size_t temp_storage_bytes = 0;
  thrust::transform_iterator<Predicate, StencilIt> flags{stencil, predicate};
  value_type* d_partition_out   = nullptr;
  size_type* d_num_selected_out = nullptr;
  size_type num_items           = static_cast<size_type>(thrust::distance(first, last));
  hipStream_t stream            = hip_rocprim::stream(policy);
  bool debug_sync               = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items <= 0)
  {
    return thrust::make_pair(selected_result, rejected_result);
  }

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::partition(
      nullptr, temp_storage_bytes, first, flags, d_partition_out, d_num_selected_out, num_items, stream, debug_sync),
    "partition failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;

  auto l_part = make_linear_partition(
    make_partition(&temp_stor, temp_storage_bytes),
    ptr_aligned_array(&d_num_selected_out, 1),
    ptr_aligned_array(&d_partition_out, num_items));

  // Calculate storage_size including alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  ptr = static_cast<void*>(tmp.data().get());

  // Create pointers with alignment
  hip_rocprim::throw_on_error(partition(ptr, storage_size, l_part));

  hip_rocprim::throw_on_error(
    rocprim::partition(
      ptr, temp_storage_bytes, first, flags, d_partition_out, d_num_selected_out, num_items, stream, debug_sync),
    "partition failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  thrust::copy_n(policy, d_partition_out, num_items, selected_result);

  return thrust::make_pair(selected_result + num_selected, rejected_result);
}

template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
THRUST_HIP_RUNTIME_FUNCTION pair<SelectedOutIt, RejectedOutIt> partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  using size_type = typename iterator_traits<InputIt>::difference_type;
  using namespace thrust::system::hip_rocprim::temp_storage;

  size_t temp_storage_bytes     = 0;
  size_type* d_num_selected_out = nullptr;
  size_type num_items           = static_cast<size_type>(thrust::distance(first, last));
  hipStream_t stream            = hip_rocprim::stream(policy);
  bool debug_sync               = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items <= 0)
  {
    return thrust::make_pair(selected_result, rejected_result);
  }

  hip_rocprim::throw_on_error(
    rocprim::partition_two_way(
      nullptr,
      temp_storage_bytes,
      first,
      selected_result,
      rejected_result,
      d_num_selected_out,
      num_items,
      predicate,
      stream,
      debug_sync),
    "partition failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;

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
    rocprim::partition_two_way(
      ptr,
      temp_storage_bytes,
      first,
      selected_result,
      rejected_result,
      d_num_selected_out,
      num_items,
      predicate,
      stream,
      debug_sync),
    "partition failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  return thrust::make_pair(selected_result + num_selected, rejected_result + num_items - num_selected);
}

template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
THRUST_HIP_RUNTIME_FUNCTION pair<SelectedOutIt, RejectedOutIt> partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  using size_type = typename iterator_traits<InputIt>::difference_type;
  using namespace thrust::system::hip_rocprim::temp_storage;

  size_t temp_storage_bytes = 0;
  thrust::transform_iterator<Predicate, StencilIt> flags{stencil, predicate};
  size_type* d_num_selected_out = nullptr;
  size_type num_items           = static_cast<size_type>(thrust::distance(first, last));
  hipStream_t stream            = hip_rocprim::stream(policy);
  bool debug_sync               = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items <= 0)
  {
    return thrust::make_pair(selected_result, rejected_result);
  }

  hip_rocprim::throw_on_error(
    rocprim::partition_two_way(
      nullptr,
      temp_storage_bytes,
      first,
      flags,
      selected_result,
      rejected_result,
      d_num_selected_out,
      num_items,
      stream,
      debug_sync),
    "partition failed on 1st step");

  size_t storage_size;
  void* ptr       = nullptr;
  void* temp_stor = nullptr;

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
    rocprim::partition_two_way(
      ptr,
      temp_storage_bytes,
      first,
      flags,
      selected_result,
      rejected_result,
      d_num_selected_out,
      num_items,
      stream,
      debug_sync),
    "partition failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  return thrust::make_pair(selected_result + num_selected, rejected_result + num_items - num_selected);
}

template <typename Derived, typename InputIt, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION InputIt
inplace_partition(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  if (thrust::distance(first, last) <= 0)
  {
    return first;
  }

  // Element type of the input iterator
  using value_t         = typename iterator_traits<InputIt>::value_type;
  std::size_t num_items = static_cast<std::size_t>(thrust::distance(first, last));

  // Allocate temporary storage, which will serve as the input to the partition
  thrust::detail::temporary_array<value_t, Derived> tmp(policy, num_items);
  hip_rocprim::uninitialized_copy(policy, first, last, tmp.begin());

  // Partition input from temporary storage to the user-provided range [`first`, `last`)
  pair<InputIt, InputIt> result =
    partition(policy, tmp.data().get(), tmp.data().get() + num_items, first, first, predicate);

  std::size_t num_selected = result.first - first;

  return first + num_selected;
}

template <typename Derived, typename InputIt, typename StencilIt, typename Predicate>
THRUST_HIP_RUNTIME_FUNCTION InputIt inplace_partition(
  execution_policy<Derived>& policy, InputIt first, InputIt last, StencilIt stencil, Predicate predicate)
{
  if (thrust::distance(first, last) <= 0)
  {
    return first;
  }

  // Element type of the input iterator
  using value_t         = typename iterator_traits<InputIt>::value_type;
  std::size_t num_items = static_cast<std::size_t>(thrust::distance(first, last));

  // Allocate temporary storage, which will serve as the input to the partition
  thrust::detail::temporary_array<value_t, Derived> tmp(policy, num_items);
  hip_rocprim::uninitialized_copy(policy, first, last, tmp.begin());

  // Partition input from temporary storage to the user-provided range [`first`, `last`)
  pair<InputIt, InputIt> result =
    partition(policy, tmp.data().get(), tmp.data().get() + num_items, stencil, first, first, predicate);

  std::size_t num_selected = result.first - first;

  return first + num_selected;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HOST_DEVICE partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static pair<SelectedOutIt, RejectedOutIt>
    par(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        StencilIt stencil,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return detail::partition_copy(policy, first, last, stencil, selected_result, rejected_result, predicate);
    }

    THRUST_DEVICE static pair<SelectedOutIt, RejectedOutIt>
    seq(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        StencilIt stencil,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return thrust::partition_copy(
        cvt_to_seq(derived_cast(policy)), first, last, stencil, selected_result, rejected_result, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, selected_result, rejected_result, predicate);
#  else
  return workaround::seq(policy, first, last, stencil, selected_result, rejected_result, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HOST_DEVICE partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static pair<SelectedOutIt, RejectedOutIt>
    par(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return detail::partition_copy(policy, first, last, selected_result, rejected_result, predicate);
    }

    THRUST_DEVICE static pair<SelectedOutIt, RejectedOutIt>
    seq(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return thrust::partition_copy(
        cvt_to_seq(derived_cast(policy)), first, last, selected_result, rejected_result, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, selected_result, rejected_result, predicate);
#  else
  return workaround::seq(policy, first, last, selected_result, rejected_result, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class StencilIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HOST_DEVICE stable_partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static pair<SelectedOutIt, RejectedOutIt>
    par(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        StencilIt stencil,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return detail::partition_copy(policy, first, last, stencil, selected_result, rejected_result, predicate);
    }
    THRUST_DEVICE static pair<SelectedOutIt, RejectedOutIt>
    seq(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        StencilIt stencil,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return thrust::stable_partition_copy(
        cvt_to_seq(derived_cast(policy)), first, last, stencil, selected_result, rejected_result, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, selected_result, rejected_result, predicate);
#  else
  return workaround::seq(policy, first, last, stencil, selected_result, rejected_result, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class SelectedOutIt, class RejectedOutIt, class Predicate>
pair<SelectedOutIt, RejectedOutIt> THRUST_HOST_DEVICE stable_partition_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  SelectedOutIt selected_result,
  RejectedOutIt rejected_result,
  Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static pair<SelectedOutIt, RejectedOutIt>
    par(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return detail::partition_copy(policy, first, last, selected_result, rejected_result, predicate);
    }

    THRUST_DEVICE static pair<SelectedOutIt, RejectedOutIt>
    seq(execution_policy<Derived>& policy,
        InputIt first,
        InputIt last,
        SelectedOutIt selected_result,
        RejectedOutIt rejected_result,
        Predicate predicate)
    {
      return thrust::stable_partition_copy(
        cvt_to_seq(derived_cast(policy)), first, last, selected_result, rejected_result, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, selected_result, rejected_result, predicate);
#  else
  return workaround::seq(policy, first, last, selected_result, rejected_result, predicate);
#  endif
}

/// inplace

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class StencilIt, class Predicate>
Iterator THRUST_HOST_DEVICE
partition(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Iterator
    par(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
    {
      return last = detail::inplace_partition(policy, first, last, stencil, predicate);
    }
    THRUST_DEVICE static Iterator
    seq(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
    {
      return last = thrust::partition(cvt_to_seq(derived_cast(policy)), first, last, stencil, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, predicate);
#  else
  return workaround::seq(policy, first, last, stencil, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class Predicate>
Iterator THRUST_HOST_DEVICE
partition(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Iterator
    par(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
    {
      return last = detail::inplace_partition(policy, first, last, predicate);
    }
    THRUST_DEVICE static Iterator
    seq(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
    {
      return last = thrust::partition(cvt_to_seq(derived_cast(policy)), first, last, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, predicate);
#  else
  return workaround::seq(policy, first, last, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class StencilIt, class Predicate>
Iterator THRUST_HOST_DEVICE stable_partition(
  execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Iterator
    par(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
    {
      Iterator ret = detail::inplace_partition(policy, first, last, stencil, predicate);

      /* partition returns rejected values in reverse order
        so reverse the rejected elements to make it stable */
      hip_rocprim::reverse(policy, ret, last);
      return ret;
    }

    THRUST_DEVICE static Iterator
    seq(execution_policy<Derived>& policy, Iterator first, Iterator last, StencilIt stencil, Predicate predicate)
    {
      return thrust::stable_partition(cvt_to_seq(derived_cast(policy)), first, last, stencil, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, stencil, predicate);
#  else
  return workaround::seq(policy, first, last, stencil, predicate);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class Predicate>
Iterator THRUST_HOST_DEVICE
stable_partition(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static Iterator
    par(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
    {
      Iterator ret = detail::inplace_partition(policy, first, last, predicate);

      /* partition returns rejected values in reverse order
        so reverse the rejected elements to make it stable */
      hip_rocprim::reverse(policy, ret, last);
      return ret;
    }

    THRUST_DEVICE static Iterator
    seq(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate predicate)
    {
      return thrust::stable_partition(cvt_to_seq(derived_cast(policy)), first, last, predicate);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, predicate);
#  else
  return workaround::seq(policy, first, last, predicate);
#  endif
}

template <class Derived, class ItemsIt, class Predicate>
bool THRUST_HOST_DEVICE
is_partitioned(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, Predicate predicate)
{
  ItemsIt boundary = hip_rocprim::find_if_not(policy, first, last, predicate);
  ItemsIt end      = hip_rocprim::find_if(policy, boundary, last, predicate);
  return end == last;
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
