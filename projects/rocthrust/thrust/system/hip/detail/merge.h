/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/pair.h>
#  include <thrust/system/hip/detail/dispatch.h>
#  include <thrust/system/hip/detail/util.h>

#  include <cstdint>

// rocPRIM includes
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace __merge
{
template <class KeyType, class ValueType, class Predicate>
struct predicate_wrapper
{
  Predicate predicate;
  using pair_type = rocprim::tuple<KeyType, ValueType>;

  THRUST_HIP_FUNCTION
  predicate_wrapper(Predicate p)
      : predicate(p)
  {}

  bool THRUST_HIP_DEVICE_FUNCTION operator()(pair_type const& lhs, pair_type const& rhs) const
  {
    return predicate(rocprim::get<0>(lhs), rocprim::get<0>(rhs));
  }
}; // struct predicate_wrapper

template <class Derived, class KeysIt1, class KeysIt2, class ResultIt, class CompareOp>
ResultIt THRUST_HIP_RUNTIME_FUNCTION merge(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_begin,
  KeysIt1 keys1_end,
  KeysIt2 keys2_begin,
  KeysIt2 keys2_end,
  ResultIt result_begin,
  CompareOp compare_op)

{
  using size_type = size_t;

  size_type input1_size = static_cast<size_type>(thrust::distance(keys1_begin, keys1_end));
  size_type input2_size = static_cast<size_type>(thrust::distance(keys2_begin, keys2_end));

  if (input1_size == 0 && input2_size == 0)
  {
    return result_begin;
  }

  size_t storage_size = 0;
  hipStream_t stream  = hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::merge(
      nullptr,
      storage_size,
      keys1_begin,
      keys2_begin,
      result_begin,
      input1_size,
      input2_size,
      compare_op,
      stream,
      debug_sync),
    "merge failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  hip_rocprim::throw_on_error(
    rocprim::merge(
      ptr, storage_size, keys1_begin, keys2_begin, result_begin, input1_size, input2_size, compare_op, stream, debug_sync),
    "merge failed on 2nd step");
  hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(policy), "merge: failed to synchronize");

  ResultIt result_end = result_begin + input1_size + input2_size;
  return result_end;
}

template <typename Derived,
          typename KeysIt1,
          typename KeysIt2,
          typename ItemsIt1,
          typename ItemsIt2,
          typename KeysOutputIt,
          typename ItemsOutputIt,
          typename CompareOp>
THRUST_HIP_RUNTIME_FUNCTION pair<KeysOutputIt, ItemsOutputIt> merge(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_begin,
  KeysIt1 keys1_end,
  KeysIt2 keys2_begin,
  KeysIt2 keys2_end,
  ItemsIt1 items1_begin,
  ItemsIt2 items2_begin,
  KeysOutputIt keys_out_begin,
  ItemsOutputIt items_out_begin,
  CompareOp compare_op)
{
  using size_type = size_t;

  using KeyType   = typename iterator_traits<KeysIt1>::value_type;
  using ValueType = typename iterator_traits<ItemsIt1>::value_type;

  predicate_wrapper<KeyType, ValueType, CompareOp> wrapped_binary_pred(compare_op);

  size_type input1_size = static_cast<size_type>(thrust::distance(keys1_begin, keys1_end));
  size_type input2_size = static_cast<size_type>(thrust::distance(keys2_begin, keys2_end));

  if (input1_size == 0 && input2_size == 0)
  {
    return thrust::make_pair(keys_out_begin, items_out_begin);
  };

  size_t storage_size = 0;
  hipStream_t stream  = hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  // Determine temporary device storage requirements.
  hip_rocprim::throw_on_error(
    rocprim::merge(
      nullptr,
      storage_size,
      rocprim::make_zip_iterator(rocprim::make_tuple(keys1_begin, items1_begin)),
      rocprim::make_zip_iterator(rocprim::make_tuple(keys2_begin, items2_begin)),
      rocprim::make_zip_iterator(rocprim::make_tuple(keys_out_begin, items_out_begin)),
      input1_size,
      input2_size,
      wrapped_binary_pred,
      stream,
      debug_sync),
    "merge_by_key failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);
  void* ptr = static_cast<void*>(tmp.data().get());

  hip_rocprim::throw_on_error(
    rocprim::merge(
      ptr,
      storage_size,
      rocprim::make_zip_iterator(rocprim::make_tuple(keys1_begin, items1_begin)),
      rocprim::make_zip_iterator(rocprim::make_tuple(keys2_begin, items2_begin)),
      rocprim::make_zip_iterator(rocprim::make_tuple(keys_out_begin, items_out_begin)),
      input1_size,
      input2_size,
      wrapped_binary_pred,
      stream,
      debug_sync),
    "merge_by_key failed on 2nd step");
  hip_rocprim::throw_on_error(hip_rocprim::synchronize_optional(policy), "merge: failed to synchronize");

  size_t count = input1_size + input2_size;
  return thrust::make_pair(keys_out_begin + count, items_out_begin + count);
}

} // namespace __merge

//-------------------------
// Thrust API entry points
//-------------------------
THRUST_EXEC_CHECK_DISABLE
template <class Derived, class KeysIt1, class KeysIt2, class ResultIt, class CompareOp = less<>>
ResultIt THRUST_HOST_DEVICE
merge(execution_policy<Derived>& policy,
      KeysIt1 keys1_begin,
      KeysIt1 keys1_end,
      KeysIt2 keys2_begin,
      KeysIt2 keys2_end,
      ResultIt result_begin,
      CompareOp compare_op = {})
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static ResultIt
    par(execution_policy<Derived>& policy,
        KeysIt1 keys1_begin,
        KeysIt1 keys1_end,
        KeysIt2 keys2_begin,
        KeysIt2 keys2_end,
        ResultIt result_begin,
        CompareOp compare_op)
    {
      return __merge::merge(policy, keys1_begin, keys1_end, keys2_begin, keys2_end, result_begin, compare_op);
    }
    THRUST_DEVICE static ResultIt
    seq(execution_policy<Derived>& policy,
        KeysIt1 keys1_begin,
        KeysIt1 keys1_end,
        KeysIt2 keys2_begin,
        KeysIt2 keys2_end,
        ResultIt result_begin,
        CompareOp compare_op)
    {
      return thrust::merge(
        cvt_to_seq(derived_cast(policy)), keys1_begin, keys1_end, keys2_begin, keys2_end, result_begin, compare_op);
    }
  };
#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, keys1_begin, keys1_end, keys2_begin, keys2_end, result_begin, compare_op);
#  else
  return workaround::seq(policy, keys1_begin, keys1_end, keys2_begin, keys2_end, result_begin, compare_op);
#  endif
}

THRUST_EXEC_CHECK_DISABLE
template <class Derived,
          class KeysIt1,
          class KeysIt2,
          class ItemsIt1,
          class ItemsIt2,
          class KeysOutputIt,
          class ItemsOutputIt,
          class CompareOp = less<>>
pair<KeysOutputIt, ItemsOutputIt> THRUST_HOST_DEVICE merge_by_key(
  execution_policy<Derived>& policy,
  KeysIt1 keys1_begin,
  KeysIt1 keys1_end,
  KeysIt2 keys2_begin,
  KeysIt2 keys2_end,
  ItemsIt1 items1_begin,
  ItemsIt2 items2_begin,
  KeysOutputIt keys_out_begin,
  ItemsOutputIt items_out_begin,
  CompareOp compare_op = {})
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static pair<KeysOutputIt, ItemsOutputIt> par(
      execution_policy<Derived>& policy,
      KeysIt1 keys1_begin,
      KeysIt1 keys1_end,
      KeysIt2 keys2_begin,
      KeysIt2 keys2_end,
      ItemsIt1 items1_begin,
      ItemsIt2 items2_begin,
      KeysOutputIt keys_out_begin,
      ItemsOutputIt items_out_begin,
      CompareOp compare_op)
    {
      return __merge::merge(
        policy,
        keys1_begin,
        keys1_end,
        keys2_begin,
        keys2_end,
        items1_begin,
        items2_begin,
        keys_out_begin,
        items_out_begin,
        compare_op);
    }
    THRUST_DEVICE static pair<KeysOutputIt, ItemsOutputIt> seq(
      execution_policy<Derived>& policy,
      KeysIt1 keys1_begin,
      KeysIt1 keys1_end,
      KeysIt2 keys2_begin,
      KeysIt2 keys2_end,
      ItemsIt1 items1_begin,
      ItemsIt2 items2_begin,
      KeysOutputIt keys_out_begin,
      ItemsOutputIt items_out_begin,
      CompareOp compare_op)
    {
      return thrust::merge_by_key(
        cvt_to_seq(derived_cast(policy)),
        keys1_begin,
        keys1_end,
        keys2_begin,
        keys2_end,
        items1_begin,
        items2_begin,
        keys_out_begin,
        items_out_begin,
        compare_op);
    }
  };

#  if __THRUST_HAS_HIPRT__
  return workaround::par(
    policy,
    keys1_begin,
    keys1_end,
    keys2_begin,
    keys2_end,
    items1_begin,
    items2_begin,
    keys_out_begin,
    items_out_begin,
    compare_op);
#  else
  return workaround::seq(
    policy,
    keys1_begin,
    keys1_end,
    keys2_begin,
    keys2_end,
    items1_begin,
    items2_begin,
    keys_out_begin,
    items_out_begin,
    compare_op);
#  endif
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END
#endif
