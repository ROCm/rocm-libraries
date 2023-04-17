//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_PARTITION_COPY_H
#define __GPU___ALGORITHM_PARTITION_COPY_H

#include "gpu/__config"

namespace gpu {

template <class _InputIterator, class _OutputIterator1,
          class _OutputIterator2, class _Predicate>
_LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 gpu::pair<_OutputIterator1, _OutputIterator2>
partition_copy(_InputIterator __first, _InputIterator __last,
               _OutputIterator1 __out_true, _OutputIterator2 __out_false,
               _Predicate __pred)
{
    for (; __first != __last; ++__first)
    {
        if (__pred(*__first))
        {
            *__out_true = *__first;
            ++__out_true;
        }
        else
        {
            *__out_false = *__first;
            ++__out_false;
        }
    }
    return gpu::pair<_OutputIterator1, _OutputIterator2>(__out_true, __out_false);
}

} // namespace gpu

#endif // __GPU___ALGORITHM_PARTITION_COPY_H
