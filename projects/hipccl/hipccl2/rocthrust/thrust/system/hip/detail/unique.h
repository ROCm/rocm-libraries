/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *  Modifications Copyright© 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#  include <thrust/detail/minmax.h>
#  include <thrust/detail/mpl/math.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/functional.h>
#  include <thrust/system/hip/detail/general/temp_storage.h>
#  include <thrust/system/hip/detail/get_value.h>
#  include <thrust/system/hip/detail/par_to_seq.h>
#  include <thrust/system/hip/detail/util.h>

#  include <cstdint>

// rocPRIM includes
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename ForwardIterator, typename BinaryPredicate>
ForwardIterator THRUST_HOST_DEVICE
unique(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
       ForwardIterator first,
       ForwardIterator last,
       BinaryPredicate binary_pred);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryPredicate>
THRUST_HOST_DEVICE OutputIterator unique_copy(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryPredicate binary_pred);

namespace hip_rocprim
{
namespace __unique
{
template <typename Derived, typename ItemsInputIt, typename ItemsOutputIt, typename BinaryPred>
THRUST_HIP_RUNTIME_FUNCTION ItemsOutputIt unique(
  execution_policy<Derived>& policy,
  ItemsInputIt items_first,
  ItemsInputIt items_last,
  ItemsOutputIt items_result,
  BinaryPred binary_pred)
{
  using namespace thrust::system::hip_rocprim::temp_storage;
  using size_type = size_t;

  size_type num_items       = static_cast<size_type>(thrust::distance(items_first, items_last));
  size_t temp_storage_bytes = 0;
  hipStream_t stream        = hip_rocprim::stream(policy);
  bool debug_sync           = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items == 0)
  {
    return items_result;
  }

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::unique(
      nullptr,
      temp_storage_bytes,
      items_first,
      items_result,
      static_cast<size_type*>(nullptr),
      num_items,
      binary_pred,
      stream,
      debug_sync),
    "unique failed on 1st step");

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
    rocprim::unique(
      ptr, temp_storage_bytes, items_first, items_result, d_num_selected_out, num_items, binary_pred, stream, debug_sync),
    "unique failed on 2nd step");

  size_type num_selected = get_value(policy, d_num_selected_out);

  return items_result + num_selected;
}
} // namespace __unique

//-------------------------
// Thrust API entry points
//-------------------------

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class OutputIt, class BinaryPred>
OutputIt THRUST_HIP_FUNCTION
unique_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryPred binary_pred)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static OutputIt
    par(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryPred binary_pred)
    {
      return __unique::unique(policy, first, last, result, binary_pred);
    }
    THRUST_DEVICE static OutputIt
    seq(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryPred binary_pred)
    {
      return thrust::unique_copy(cvt_to_seq(derived_cast(policy)), first, last, result, binary_pred);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, result, binary_pred);
#  else
  return workaround::seq(policy, first, last, result, binary_pred);
#  endif
}

template <class Derived, class InputIt, class OutputIt>
OutputIt THRUST_HIP_FUNCTION unique_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  using input_type = typename iterator_traits<InputIt>::value_type;
  return hip_rocprim::unique_copy(policy, first, last, result, equal_to<input_type>());
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class BinaryPred>
InputIt THRUST_HIP_FUNCTION
unique(execution_policy<Derived>& policy, InputIt first, InputIt last, BinaryPred binary_pred)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static InputIt
    par(execution_policy<Derived>& policy, InputIt first, InputIt last, BinaryPred binary_pred)
    {
      return hip_rocprim::unique_copy(policy, first, last, first, binary_pred);
    }
    THRUST_DEVICE static InputIt
    seq(execution_policy<Derived>& policy, InputIt first, InputIt last, BinaryPred binary_pred)
    {
      return thrust::unique(cvt_to_seq(derived_cast(policy)), first, last, binary_pred);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, first, last, binary_pred);
#  else
  return workaround::seq(policy, first, last, binary_pred);
#  endif
}

template <class Derived, class InputIt>
InputIt THRUST_HIP_FUNCTION unique(execution_policy<Derived>& policy, InputIt first, InputIt last)
{
  using input_type = typename iterator_traits<InputIt>::value_type;
  return hip_rocprim::unique(policy, first, last, equal_to<input_type>());
}

template <typename BinaryPred>
struct zip_adj_not_predicate
{
  template <typename TupleType>
  bool THRUST_HOST_DEVICE operator()(TupleType&& tuple)
  {
    return !binary_pred(thrust::get<0>(tuple), thrust::get<1>(tuple));
  }

  BinaryPred binary_pred;
};

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class ForwardIt, class BinaryPred>
typename thrust::iterator_traits<ForwardIt>::difference_type THRUST_HIP_FUNCTION
unique_count(execution_policy<Derived>& policy, ForwardIt first, ForwardIt last, BinaryPred binary_pred)
{
  if (first == last)
  {
    return 0;
  }
  auto size = thrust::distance(first, last);
  auto it   = thrust::make_zip_iterator(thrust::make_tuple(first, thrust::next(first)));
  return 1 + thrust::count_if(policy, it, thrust::next(it, size - 1), zip_adj_not_predicate<BinaryPred>{binary_pred});
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/unique.h>
#endif
