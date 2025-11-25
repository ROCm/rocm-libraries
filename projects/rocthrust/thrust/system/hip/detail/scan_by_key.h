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

#  include <thrust/detail/minmax.h>
#  include <thrust/detail/mpl/math.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/system/hip/detail/par_to_seq.h>
#  include <thrust/system/hip/detail/util.h>
#  include <thrust/system/hip/execution_policy.h>

#  include <cstdint>

// rocprim include
#  include <rocprim/rocprim.hpp>

THRUST_NAMESPACE_BEGIN
namespace hip_rocprim
{
namespace __scan_by_key
{

template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename ValuesOutputIt,
          typename BinaryFunction,
          typename KeyCompareFunction>
THRUST_HOST_DEVICE auto invoke_inclusive_scan_by_key(
  execution_policy<Derived>& policy,
  void* temporary_storage,
  size_t& storage_size,
  const KeysInputIt keys_input,
  const ValuesInputIt values_input,
  const ValuesOutputIt values_output,
  const size_t size,
  const BinaryFunction scan_op,
  const KeyCompareFunction key_compare_op,
  const hipStream_t stream,
  bool debug_sync) -> std::enable_if_t<decltype(nondeterministic(policy))::value, hipError_t>
{
  return rocprim::inclusive_scan_by_key(
    temporary_storage,
    storage_size,
    keys_input,
    values_input,
    values_output,
    size,
    scan_op,
    key_compare_op,
    stream,
    debug_sync);
}

template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename ValuesOutputIt,
          typename BinaryFunction,
          typename KeyCompareFunction>
THRUST_HOST_DEVICE auto invoke_inclusive_scan_by_key(
  execution_policy<Derived>& policy,
  void* temporary_storage,
  size_t& storage_size,
  const KeysInputIt keys_input,
  const ValuesInputIt values_input,
  const ValuesOutputIt values_output,
  const size_t size,
  const BinaryFunction scan_op,
  const KeyCompareFunction key_compare_op,
  const hipStream_t stream,
  bool debug_sync) -> std::enable_if_t<!decltype(nondeterministic(policy))::value, hipError_t>
{
  return rocprim::deterministic_inclusive_scan_by_key(
    temporary_storage,
    storage_size,
    keys_input,
    values_input,
    values_output,
    size,
    scan_op,
    key_compare_op,
    stream,
    debug_sync);
}

THRUST_EXEC_CHECK_DISABLE
template <
  typename Derived,
  typename KeysInputIterator,
  typename ValuesInputIterator,
  typename ValuesOutputIterator,
  typename KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
  typename BinaryFunction     = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>>
THRUST_HOST_DEVICE ValuesOutputIterator inclusive_scan_by_key(
  thrust::hip_rocprim::execution_policy<Derived>& policy,
  KeysInputIterator key_first,
  KeysInputIterator key_last,
  ValuesInputIterator value_first,
  ValuesOutputIterator value_result,
  KeyCompareFunction key_compare_op,
  BinaryFunction scan_op)
{
  size_t num_items    = static_cast<size_t>(thrust::distance(key_first, key_last));
  hipStream_t stream  = hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;

  if (num_items == 0)
  {
    return value_result;
  }

  // Determine temporary storage requirements:
  std::size_t storage_size = 0;
  {
    hip_rocprim::throw_on_error(
      invoke_inclusive_scan_by_key(
        policy,
        nullptr,
        storage_size,
        key_first,
        value_first,
        value_result,
        num_items,
        scan_op,
        key_compare_op,
        stream,
        debug_sync),
      "scan_by_key failed on 1st step");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, storage_size};
    void* ptr = static_cast<void*>(tmp.data().get());

    hip_rocprim::throw_on_error(
      invoke_inclusive_scan_by_key(
        policy,
        ptr,
        storage_size,
        key_first,
        value_first,
        value_result,
        num_items,
        scan_op,
        key_compare_op,
        stream,
        debug_sync),
      "scan_by_key failed on 2nd step");

    thrust::hip_rocprim::throw_on_error(
      thrust::hip_rocprim::synchronize_optional(policy), "inclusive_scan_by_key failed to synchronize");
  }

  return value_result + num_items;
}

template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename ValuesOutputIt,
          typename InitialValueType,
          typename BinaryFunction,
          typename KeyCompareFunction>
THRUST_HOST_DEVICE auto invoke_exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  void* temporary_storage,
  size_t& storage_size,
  const KeysInputIt keys_input,
  const ValuesInputIt values_input,
  const ValuesOutputIt values_output,
  const InitialValueType initial_value,
  const size_t size,
  const BinaryFunction scan_op,
  const KeyCompareFunction key_compare_op,
  const hipStream_t stream,
  bool debug_sync) -> std::enable_if_t<decltype(nondeterministic(policy))::value, hipError_t>
{
  return rocprim::exclusive_scan_by_key(
    temporary_storage,
    storage_size,
    keys_input,
    values_input,
    values_output,
    initial_value,
    size,
    scan_op,
    key_compare_op,
    stream,
    debug_sync);
}

template <typename Derived,
          typename KeysInputIt,
          typename ValuesInputIt,
          typename ValuesOutputIt,
          typename InitialValueType,
          typename BinaryFunction,
          typename KeyCompareFunction>
THRUST_HOST_DEVICE auto invoke_exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  void* temporary_storage,
  size_t& storage_size,
  const KeysInputIt keys_input,
  const ValuesInputIt values_input,
  const ValuesOutputIt values_output,
  const InitialValueType initial_value,
  const size_t size,
  const BinaryFunction scan_op,
  const KeyCompareFunction key_compare_op,
  const hipStream_t stream,
  bool debug_sync) -> std::enable_if_t<!decltype(nondeterministic(policy))::value, hipError_t>
{
  return rocprim::deterministic_exclusive_scan_by_key(
    temporary_storage,
    storage_size,
    keys_input,
    values_input,
    values_output,
    initial_value,
    size,
    scan_op,
    key_compare_op,
    stream,
    debug_sync);
}

THRUST_EXEC_CHECK_DISABLE
template <
  typename Derived,
  typename KeysInputIterator,
  typename ValuesInputIterator,
  typename ValuesOutputIterator,
  typename InitialValueType,
  typename KeyCompareFunction = ::rocprim::equal_to<typename std::iterator_traits<KeysInputIterator>::value_type>,
  typename BinaryFunction     = ::rocprim::plus<typename std::iterator_traits<ValuesInputIterator>::value_type>>
THRUST_HOST_DEVICE ValuesOutputIterator exclusive_scan_by_key(
  thrust::hip_rocprim::execution_policy<Derived>& policy,
  KeysInputIterator key_first,
  KeysInputIterator key_last,
  ValuesInputIterator value_first,
  ValuesOutputIterator value_result,
  InitialValueType init,
  KeyCompareFunction key_compare_op,
  BinaryFunction scan_op)
{
  size_t num_items    = static_cast<size_t>(thrust::distance(key_first, key_last));
  hipStream_t stream  = hip_rocprim::stream(policy);
  bool debug_sync     = THRUST_HIP_DEBUG_SYNC_FLAG;
  if (num_items == 0)
  {
    return value_result;
  }

  // Determine temporary storage requirements:
  std::size_t storage_size = 0;
  {
    hip_rocprim::throw_on_error(
      invoke_exclusive_scan_by_key(
        policy,
        nullptr,
        storage_size,
        key_first,
        value_first,
        value_result,
        init,
        num_items,
        scan_op,
        key_compare_op,
        stream,
        debug_sync),
      "scan_by_key failed on 1st step");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, storage_size};
    void* ptr = static_cast<void*>(tmp.data().get());

    hip_rocprim::throw_on_error(
      invoke_exclusive_scan_by_key(
        policy,
        ptr,
        storage_size,
        key_first,
        value_first,
        value_result,
        init,
        num_items,
        scan_op,
        key_compare_op,
        stream,
        debug_sync),
      "scan_by_key failed on 2nd step");

    thrust::hip_rocprim::throw_on_error(
      thrust::hip_rocprim::synchronize_optional(policy), "exclusive_scan_by_key failed to synchronize");
  }

  return value_result + num_items;
}

} // namespace __scan_by_key

//-------------------------
// Thrust API entry points
//-------------------------

//---------------------------
//   Inclusive scan
//---------------------------

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class BinaryPred, class ScanOp>
ValOutputIt THRUST_HOST_DEVICE inclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result,
  BinaryPred binary_pred,
  ScanOp scan_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static ValOutputIt
    par(execution_policy<Derived>& policy,
        KeyInputIt key_first,
        KeyInputIt key_last,
        ValInputIt value_first,
        ValOutputIt value_result,
        BinaryPred binary_pred,
        ScanOp scan_op)
    {
      return thrust::hip_rocprim::__scan_by_key::inclusive_scan_by_key(
        policy, key_first, key_last, value_first, value_result, binary_pred, scan_op);
    }

    THRUST_DEVICE static ValOutputIt
    seq(execution_policy<Derived>& policy,
        KeyInputIt key_first,
        KeyInputIt key_last,
        ValInputIt value_first,
        ValOutputIt value_result,
        BinaryPred binary_pred,
        ScanOp scan_op)
    {
      return thrust::inclusive_scan_by_key(
        cvt_to_seq(derived_cast(policy)), key_first, key_last, value_first, value_result, binary_pred, scan_op);
    }
  };

#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, key_first, key_last, value_first, value_result, binary_pred, scan_op);
#  else
  return workaround::seq(policy, key_first, key_last, value_first, value_result, binary_pred, scan_op);
#  endif
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class BinaryPred>
ValOutputIt THRUST_HOST_DEVICE inclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result,
  BinaryPred binary_pred)
{
  return hip_rocprim::inclusive_scan_by_key(
    policy, key_first, key_last, value_first, value_result, binary_pred, thrust::plus<>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt>
ValOutputIt THRUST_HOST_DEVICE inclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result)
{
  return hip_rocprim::inclusive_scan_by_key(
    policy, key_first, key_last, value_first, value_result, thrust::equal_to<>());
}

//---------------------------
//   Exclusive scan
//---------------------------

THRUST_EXEC_CHECK_DISABLE
template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class Init, class BinaryPred, class ScanOp>
ValOutputIt THRUST_HOST_DEVICE exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result,
  Init init,
  BinaryPred binary_pred,
  ScanOp scan_op)
{
  // struct workaround is required for HIP-clang
  struct workaround
  {
    THRUST_HOST static ValOutputIt
    par(execution_policy<Derived>& policy,
        KeyInputIt key_first,
        KeyInputIt key_last,
        ValInputIt value_first,
        ValOutputIt value_result,
        Init init,
        BinaryPred binary_pred,
        ScanOp scan_op)
    {
      return thrust::hip_rocprim::__scan_by_key::exclusive_scan_by_key(
        policy, key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
    }

    THRUST_DEVICE static ValOutputIt
    seq(execution_policy<Derived>& policy,
        KeyInputIt key_first,
        KeyInputIt key_last,
        ValInputIt value_first,
        ValOutputIt value_result,
        Init init,
        BinaryPred binary_pred,
        ScanOp scan_op)
    {
      return thrust::exclusive_scan_by_key(
        cvt_to_seq(derived_cast(policy)), key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
    }
  };

#  if __THRUST_HAS_HIPRT__
  return workaround::par(policy, key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
#  else
  return workaround::seq(policy, key_first, key_last, value_first, value_result, init, binary_pred, scan_op);
#  endif
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class Init, class BinaryPred>
ValOutputIt THRUST_HOST_DEVICE exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result,
  Init init,
  BinaryPred binary_pred)
{
  return hip_rocprim::exclusive_scan_by_key(
    policy, key_first, key_last, value_first, value_result, init, binary_pred, thrust::plus<>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt, class Init>
ValOutputIt THRUST_HOST_DEVICE exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result,
  Init init)
{
  return hip_rocprim::exclusive_scan_by_key(
    policy, key_first, key_last, value_first, value_result, init, thrust::equal_to<>());
}

template <class Derived, class KeyInputIt, class ValInputIt, class ValOutputIt>
ValOutputIt THRUST_HOST_DEVICE exclusive_scan_by_key(
  execution_policy<Derived>& policy,
  KeyInputIt key_first,
  KeyInputIt key_last,
  ValInputIt value_first,
  ValOutputIt value_result)
{
  using value_type = typename thrust::iterator_traits<ValInputIt>::value_type;
  return hip_rocprim::exclusive_scan_by_key(policy, key_first, key_last, value_first, value_result, value_type{});
}

} // namespace hip_rocprim
THRUST_NAMESPACE_END

#  include <thrust/scan.h>

#endif
