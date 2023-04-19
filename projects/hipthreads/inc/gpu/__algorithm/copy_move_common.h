//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_COPY_MOVE_COMMON_H
#define __GPU___ALGORITHM_COPY_MOVE_COMMON_H

#include "gpu/__config"
#include "gpu/__algorithm/iterator_operations.h"
#include "gpu/__utility/pair.h"
#include "gpu/__type_traits/is_always_bitcastable.h"
#include "gpu/__algorithm/unwrap_iter.h"
#include "gpu/__algorithm/unwrap_range.h"
#include <type_traits>

namespace gpu {

// Type traits.

template <class _From, class _To>
struct __can_lower_copy_assignment_to_memmove {
  static const bool value =
    // If the types are always bitcastable, it's valid to do a bitwise copy between them.
    gpu::__is_always_bitcastable<_From, _To>::value &&
    // Reject conversions that wouldn't be performed by the regular built-in assignment (e.g. between arrays).
    std::is_trivially_assignable<_To&, const _From&>::value &&
    // `memmove` doesn't accept `volatile` pointers, make sure the optimization SFINAEs away in that case.
    !std::is_volatile<_From>::value &&
    !std::is_volatile<_To>::value;
};

template <class _From, class _To>
struct __can_lower_move_assignment_to_memmove {
  static const bool value =
    gpu::__is_always_bitcastable<_From, _To>::value &&
    std::is_trivially_assignable<_To&, _From&&>::value &&
    !std::is_volatile<_From>::value &&
    !std::is_volatile<_To>::value;
};

// `memmove` algorithms implementation.

template <class _In, class _Out>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_In*, _Out*>
__copy_trivial_impl(_In* __first, _In* __last, _Out* __result) {
  const std::size_t __n = static_cast<std::size_t>(__last - __first);
  ::__builtin_memmove(__result, __first, __n * sizeof(_Out));

  return gpu::make_pair(__last, __result + __n);
}

template <class _In, class _Out>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 gpu::pair<_In*, _Out*>
__copy_backward_trivial_impl(_In* __first, _In* __last, _Out* __result) {
  const std::size_t __n = static_cast<std::size_t>(__last - __first);
  __result -= __n;

  ::__builtin_memmove(__result, __first, __n * sizeof(_Out));

  return gpu::make_pair(__last, __result);
}

// Iterator unwrapping and dispatching to the correct overload.

template <class _F1, class _F2>
struct __overload : _F1, _F2 {
  using _F1::operator();
  using _F2::operator();
};

template <class _InIter, class _Sent, class _OutIter, class = void>
struct __can_rewrap : std::false_type {};

template <class _InIter, class _Sent, class _OutIter>
struct __can_rewrap<_InIter,
                    _Sent,
                    _OutIter,
                    // Note that sentinels are always copy-constructible.
                    std::enable_if_t< std::is_copy_constructible<_InIter>::value &&
                                   std::is_copy_constructible<_OutIter>::value > > : std::true_type {};

template <class _Algorithm,
          class _InIter,
          class _Sent,
          class _OutIter,
          std::enable_if_t<__can_rewrap<_InIter, _Sent, _OutIter>::value, int> = 0>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17 gpu::pair<_InIter, _OutIter>
__unwrap_and_dispatch(_InIter __first, _Sent __last, _OutIter __out_first) {
  auto __range  = gpu::__unwrap_range(__first, std::move(__last));
  auto __result = _Algorithm()(std::move(__range.first), std::move(__range.second), gpu::__unwrap_iter(__out_first));
  return gpu::make_pair(gpu::__rewrap_range<_Sent>(std::move(__first), std::move(__result.first)),
                                 gpu::__rewrap_iter(std::move(__out_first), std::move(__result.second)));
}

template <class _Algorithm,
          class _InIter,
          class _Sent,
          class _OutIter,
          std::enable_if_t<!__can_rewrap<_InIter, _Sent, _OutIter>::value, int> = 0>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17 gpu::pair<_InIter, _OutIter>
__unwrap_and_dispatch(_InIter __first, _Sent __last, _OutIter __out_first) {
  return _Algorithm()(std::move(__first), std::move(__last), std::move(__out_first));
}

template <class _IterOps, class _InValue, class _OutIter, class = void>
struct __can_copy_without_conversion : std::false_type {};

template <class _IterOps, class _InValue, class _OutIter>
struct __can_copy_without_conversion<
    _IterOps,
    _InValue,
    _OutIter,
    std::enable_if_t<std::is_same<_InValue, typename _IterOps::template __value_type<_OutIter> >::value> > : std::true_type {};

template <class _AlgPolicy,
          class _NaiveAlgorithm,
          class _OptimizedAlgorithm,
          class _InIter,
          class _Sent,
          class _OutIter>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX17 gpu::pair<_InIter, _OutIter>
__dispatch_copy_or_move(_InIter __first, _Sent __last, _OutIter __out_first) {
#ifdef _LIBGPU_COMPILER_GCC
  // GCC doesn't support `__builtin_memmove` during constant evaluation.
  if (__builtin_is_constant_evaluated()) {
    return gpu::__unwrap_and_dispatch<_NaiveAlgorithm>(std::move(__first), std::move(__last), std::move(__out_first));
  }
#else
  // In Clang, `__builtin_memmove` only supports fully trivially copyable types (just having trivial copy assignment is
  // insufficient). Also, conversions are not supported.
  if (__builtin_is_constant_evaluated()) {
    using _InValue = typename _IterOps<_AlgPolicy>::template __value_type<_InIter>;
    if (!std::is_trivially_copyable<_InValue>::value ||
        !__can_copy_without_conversion<_IterOps<_AlgPolicy>, _InValue, _OutIter>::value) {
      return gpu::__unwrap_and_dispatch<_NaiveAlgorithm>(std::move(__first), std::move(__last), std::move(__out_first));
    }
  }
#endif // _LIBGPU_COMPILER_GCC

  using _Algorithm = __overload<_NaiveAlgorithm, _OptimizedAlgorithm>;
  return gpu::__unwrap_and_dispatch<_Algorithm>(std::move(__first), std::move(__last), std::move(__out_first));
}

} // namespace gpu

#endif // __GPU___ALGORITHM_COPY_MOVE_COMMON_H
