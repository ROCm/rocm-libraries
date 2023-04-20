//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___UTILITY_DECLVAL_H__
#define __GPU___UTILITY_DECLVAL_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::declval
//====================================================================================================================//

// Suppress deprecation notice for volatile-qualified return type resulting
// from volatile-qualified types _Tp.
_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Tp>
__host__ __device__ _Tp&& __declval(int);
template <class _Tp>
__host__ __device__ _Tp __declval(long);
_LIBGPU_SUPPRESS_DEPRECATED_POP

template <class _Tp>
__host__ __device__ decltype(std::__declval<_Tp>(0)) declval() _NOEXCEPT;

} // namespace gpu

#endif // __GPU___UTILITY_DECLVAL_H__
