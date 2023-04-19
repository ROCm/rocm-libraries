//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_MIN_H
#define __GPU___ALGORITHM_MIN_H

#include "gpu/__config"
#include "gpu/__algorithm/comp.h"
#include "gpu/__algorithm/comp_ref_type.h"
#include "gpu/__algorithm/min_element.h"

namespace gpu {

template <class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
const _Tp&
min(_LIBGPU_LIFETIMEBOUND const _Tp& __a, _LIBGPU_LIFETIMEBOUND const _Tp& __b, _Compare __comp)
{
    return __comp(__b, __a) ? __b : __a;
}

template <class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
const _Tp&
min(_LIBGPU_LIFETIMEBOUND const _Tp& __a, _LIBGPU_LIFETIMEBOUND const _Tp& __b)
{
    return gpu::min(__a, __b, __less<_Tp>());
}

#ifndef _LIBGPU_CXX03_LANG

template<class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
_Tp
min(std::initializer_list<_Tp> __t, _Compare __comp)
{
    return *gpu::__min_element<__comp_ref_type<_Compare> >(__t.begin(), __t.end(), __comp);
}

template<class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR_SINCE_CXX14
_Tp
min(std::initializer_list<_Tp> __t)
{
    return *gpu::min_element(__t.begin(), __t.end(), __less<_Tp>());
}

#endif // _LIBGPU_CXX03_LANG

} // namespace gpu

#endif // __GPU___ALGORITHM_MIN_H
