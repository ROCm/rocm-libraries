//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_GENERATE_N_H
#define __GPU___ALGORITHM_GENERATE_N_H

#include "gpu/__config"

namespace gpu {

template <class _OutputIterator, class _Size, class _Generator>
inline _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX20
_OutputIterator
generate_n(_OutputIterator __first, _Size __orig_n, _Generator __gen)
{
    typedef decltype(std::__convert_to_integral(__orig_n)) _IntegralSize;
    _IntegralSize __n = __orig_n;
    for (; __n > 0; ++__first, (void) --__n)
        *__first = __gen();
    return __first;
}

} // namespace gpu

#endif // __GPU___ALGORITHM_GENERATE_N_H
