//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_COPY_N_H
#define __GPU___ALGORITHM_COPY_N_H

#include "gpu/__config"

namespace gpu {

template<class _InputIterator, class _Size, class _OutputIterator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
typename std::enable_if
<
    __is_cpp17_input_iterator<_InputIterator>::value &&
   !__is_cpp17_random_access_iterator<_InputIterator>::value,
    _OutputIterator
>::type
copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
    typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
    _IntegralSize __n = __orig_n;
    if (__n > 0)
    {
        *__result = *__first;
        ++__result;
        for (--__n; __n > 0; --__n)
        {
            ++__first;
            *__result = *__first;
            ++__result;
        }
    }
    return __result;
}

template<class _InputIterator, class _Size, class _OutputIterator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
typename std::enable_if
<
    __is_cpp17_random_access_iterator<_InputIterator>::value,
    _OutputIterator
>::type
copy_n(_InputIterator __first, _Size __orig_n, _OutputIterator __result)
{
    typedef typename std::iterator_traits<_InputIterator>::difference_type difference_type;
    typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
    _IntegralSize __n = __orig_n;
    return gpu::copy(__first, __first + difference_type(__n), __result);
}

} // namespace gpu

#endif // __GPU___ALGORITHM_COPY_N_H
