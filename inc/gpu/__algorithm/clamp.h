//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_CLAMP_H
#define __GPU___ALGORITHM_CLAMP_H

#include "gpu/__config"

namespace gpu {

#if _LIBGPU_STD_VER >= 17
template<class _Tp, class _Compare>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY constexpr
const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi, _Compare __comp)
{
    _LIBGPU_ASSERT(!__comp(__hi, __lo), "Bad bounds passed to gpu::clamp");
    return __comp(__v, __lo) ? __lo : __comp(__hi, __v) ? __hi : __v;

}

template<class _Tp>
_LIBGPU_NODISCARD_EXT __device__ inline
_LIBGPU_INLINE_VISIBILITY constexpr
const _Tp&
clamp(const _Tp& __v, const _Tp& __lo, const _Tp& __hi)
{
    return gpu::clamp(__v, __lo, __hi, __less<_Tp>());
}
#endif

} // namespace gpu

#endif // __GPU___ALGORITHM_CLAMP_H
