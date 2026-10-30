// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-FileCopyrightText: Modifications Copyright (c) 2019-2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: BSD-3
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

#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/hip/detail/dispatch.h>
#  include <thrust/system/hip/detail/execution_policy.h>
#  include <thrust/system/hip/detail/util.h>

#  include _THRUST_STD_INCLUDE(cstdint)
#  include _THRUST_STD_INCLUDE(__utility/pair.h)

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

  size_type input1_size = static_cast<size_type>(_THRUST_STD::distance(keys1_begin, keys1_end));
  size_type input2_size = static_cast<size_type>(_THRUST_STD::distance(keys2_begin, keys2_end));

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
THRUST_HIP_RUNTIME_FUNCTION _THRUST_STD::pair<KeysOutputIt, ItemsOutputIt> merge(
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

  using KeyType   = thrust::detail::it_value_t<KeysIt1>;
  using ValueType = thrust::detail::it_value_t<ItemsIt1>;

  predicate_wrapper<KeyType, ValueType, CompareOp> wrapped_binary_pred(compare_op);

  size_type input1_size = static_cast<size_type>(_THRUST_STD::distance(keys1_begin, keys1_end));
  size_type input2_size = static_cast<size_type>(_THRUST_STD::distance(keys2_begin, keys2_end));

  if (input1_size == 0 && input2_size == 0)
  {
    return _THRUST_STD::make_pair(keys_out_begin, items_out_begin);
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
  return _THRUST_STD::make_pair(keys_out_begin + count, items_out_begin + count);
}

} // namespace __merge

//-------------------------
// Thrust API entry points
//-------------------------
THRUST_EXEC_CHECK_DISABLE
template <class Derived, class KeysIt1, class KeysIt2, class ResultIt, class CompareOp = ::cuda::std::less<>>
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
#  if defined(__HIP_DEVICE_COMPILE__)
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
#  endif
  };
#  if !defined(__HIP_DEVICE_COMPILE__)
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
          class CompareOp = ::cuda::std::less<>>
_THRUST_STD::pair<KeysOutputIt, ItemsOutputIt> THRUST_HOST_DEVICE merge_by_key(
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
    THRUST_HOST static _THRUST_STD::pair<KeysOutputIt, ItemsOutputIt> par(
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
#  if defined(__HIP_DEVICE_COMPILE__)
    THRUST_DEVICE static _THRUST_STD::pair<KeysOutputIt, ItemsOutputIt> seq(
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
#  endif
  };

#  if !defined(__HIP_DEVICE_COMPILE__)
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
