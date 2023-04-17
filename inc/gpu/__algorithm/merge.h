//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MERGE_H
#define __GPU___ALGORITHM_MERGE_H

#include "gpu/__config"

namespace gpu {

template <class _Compare, class _InputIterator1, class _InputIterator2, class _OutputIterator>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
__merge(_InputIterator1 __first1, _InputIterator1 __last1,
        _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    for (; __first1 != __last1; ++__result)
    {
        if (__first2 == __last2)
            return gpu::copy(__first1, __last1, __result);
        if (__comp(*__first2, *__first1))
        {
            *__result = *__first2;
            ++__first2;
        }
        else
        {
            *__result = *__first1;
            ++__first1;
        }
    }
    return gpu::copy(__first2, __last2, __result);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator, class _Compare>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
merge(_InputIterator1 __first1, _InputIterator1 __last1,
      _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result, _Compare __comp)
{
    return gpu::__merge<__comp_ref_type<_Compare> >(__first1, __last1, __first2, __last2, __result, __comp);
}

template <class _InputIterator1, class _InputIterator2, class _OutputIterator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
merge(_InputIterator1 __first1, _InputIterator1 __last1,
      _InputIterator2 __first2, _InputIterator2 __last2, _OutputIterator __result)
{
    typedef typename std::iterator_traits<_InputIterator1>::value_type __v1;
    typedef typename std::iterator_traits<_InputIterator2>::value_type __v2;
    return gpu::merge(__first1, __last1, __first2, __last2, __result, __less<__v1, __v2>());
}

} // namespace gpu

#endif // __GPU___ALGORITHM_MERGE_H
