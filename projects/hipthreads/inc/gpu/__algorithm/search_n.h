// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SEARCH_N_H
#define __GPU___ALGORITHM_SEARCH_N_H

#include "gpu/__config"

namespace gpu {

template <class _AlgPolicy, class _Pred, class _Iter, class _Sent, class _SizeT, class _Type, class _Proj>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Iter, _Iter> __search_n_forward_impl(_Iter __first, _Sent __last,
                                           _SizeT __count,
                                           const _Type& __value,
                                           _Pred& __pred,
                                           _Proj& __proj) {
  if (__count <= 0)
    return gpu::make_pair(__first, __first);
  while (true) {
    // Find first element in sequence that matchs __value, with a mininum of loop checks
    while (true) {
      if (__first == __last) { // return __last if no element matches __value
        _IterOps<_AlgPolicy>::__advance_to(__first, __last);
        return gpu::make_pair(__first, __first);
      }
      if (std::__invoke(__pred, std::__invoke(__proj, *__first), __value))
        break;
      ++__first;
    }
    // *__first matches __value, now match elements after here
    _Iter __m = __first;
    _SizeT __c(0);
    while (true) {
      if (++__c == __count) // If pattern exhausted, __first is the answer (works for 1 element pattern)
        return gpu::make_pair(__first, ++__m);
      if (++__m == __last) { // Otherwise if source exhaused, pattern not found
        _IterOps<_AlgPolicy>::__advance_to(__first, __last);
        return gpu::make_pair(__first, __first);
      }

      // if there is a mismatch, restart with a new __first
      if (!std::__invoke(__pred, std::__invoke(__proj, *__m), __value))
      {
        __first = __m;
        ++__first;
        break;
      } // else there is a match, check next elements
    }
  }
}

template <class _AlgPolicy, class _Pred, class _Iter, class _Sent, class _SizeT, class _Type, class _Proj, class _DiffT>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Iter, _Iter> __search_n_random_access_impl(_Iter __first, _Sent __last,
                                                      _SizeT __count,
                                                      const _Type& __value,
                                                      _Pred& __pred,
                                                      _Proj& __proj,
                                                      _DiffT __size1) {
  using difference_type = typename std::iterator_traits<_Iter>::difference_type;
  if (__count == 0)
    return gpu::make_pair(__first, __first);
  if (__size1 < static_cast<_DiffT>(__count)) {
    _IterOps<_AlgPolicy>::__advance_to(__first, __last);
    return gpu::make_pair(__first, __first);
  }

  const auto __s = __first + __size1 - difference_type(__count - 1); // Start of pattern match can't go beyond here
  while (true) {
    // Find first element in sequence that matchs __value, with a mininum of loop checks
    while (true) {
      if (__first >= __s) { // return __last if no element matches __value
        _IterOps<_AlgPolicy>::__advance_to(__first, __last);
        return gpu::make_pair(__first, __first);
      }
      if (std::__invoke(__pred, std::__invoke(__proj, *__first), __value))
        break;
      ++__first;
    }
    // *__first matches __value_, now match elements after here
    auto __m = __first;
    _SizeT __c(0);
    while (true) {
      if (++__c == __count) // If pattern exhausted, __first is the answer (works for 1 element pattern)
        return gpu::make_pair(__first, __first + _DiffT(__count));
      ++__m; // no need to check range on __m because __s guarantees we have enough source

      // if there is a mismatch, restart with a new __first
      if (!std::__invoke(__pred, std::__invoke(__proj, *__m), __value))
      {
        __first = __m;
        ++__first;
        break;
      } // else there is a match, check next elements
    }
  }
}

template <class _Iter, class _Sent,
          class _DiffT,
          class _Type,
          class _Pred,
          class _Proj>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Iter, _Iter> __search_n_impl(_Iter __first, _Sent __last,
                                   _DiffT __count,
                                   const _Type& __value,
                                   _Pred& __pred,
                                   _Proj& __proj,
                                   std::enable_if_t<__is_cpp17_random_access_iterator<_Iter>::value>* = nullptr) {
  return std::__search_n_random_access_impl<_ClassicAlgPolicy>(__first, __last,
                                                               __count,
                                                               __value,
                                                               __pred,
                                                               __proj,
                                                               __last - __first);
}

template <class _Iter1, class _Sent1,
          class _DiffT,
          class _Type,
          class _Pred,
          class _Proj>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
gpu::pair<_Iter1, _Iter1> __search_n_impl(_Iter1 __first, _Sent1 __last,
                                     _DiffT __count,
                                     const _Type& __value,
                                     _Pred& __pred,
                                     _Proj& __proj,
                                     std::enable_if_t<__is_cpp17_forward_iterator<_Iter1>::value
                                               && !__is_cpp17_random_access_iterator<_Iter1>::value>* = nullptr) {
  return std::__search_n_forward_impl<_ClassicAlgPolicy>(__first, __last,
                                                         __count,
                                                         __value,
                                                         __pred,
                                                         __proj);
}

template <class _ForwardIterator, class _Size, class _Tp, class _BinaryPredicate>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_ForwardIterator search_n(_ForwardIterator __first, _ForwardIterator __last,
                          _Size __count,
                          const _Tp& __value,
                          _BinaryPredicate __pred) {
  static_assert(__is_callable<_BinaryPredicate, decltype(*__first), const _Tp&>::value,
                "BinaryPredicate has to be callable");
  auto __proj = __identity();
  return std::__search_n_impl(__first, __last, std::__convert_to_integral(__count), __value, __pred, __proj).first;
}

template <class _ForwardIterator, class _Size, class _Tp>
_LIBGPU_NODISCARD_EXT inline _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_ForwardIterator search_n(_ForwardIterator __first, _ForwardIterator __last, _Size __count, const _Tp& __value) {
  return gpu::search_n(__first, __last, std::__convert_to_integral(__count), __value, __equal_to());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SEARCH_N_H
