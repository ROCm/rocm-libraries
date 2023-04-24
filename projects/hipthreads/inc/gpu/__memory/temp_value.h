//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___MEMORY_TEMP_VALUE_H__
#define __GPU___MEMORY_TEMP_VALUE_H__

#include "gpu/__config"

#include <type_traits>

#include "gpu/__memory/addressof.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __memory/temp_value.h
//====================================================================================================================//

// TODO: re-add allocator template parameter
template <class _Tp>
struct __temp_value {
#ifdef _LIBGPU_CXX03_LANG
    typename std::aligned_storage<sizeof(_Tp), _LIBGPU_ALIGNOF(_Tp)>::type __v;
#else
    union { _Tp __v; };
#endif

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp *__addr() {
#ifdef _LIBGPU_CXX03_LANG
        return reinterpret_cast<_Tp*>(gpu::addressof(__v));
#else
        return gpu::addressof(__v);
#endif
    }

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 _Tp &   get() { return *__addr(); }

    template<class... _Args>
    __device__ _LIBGPU_NO_CFI
    _LIBGPU_CONSTEXPR_SINCE_CXX20 __temp_value(_Args&& ... __args) {
        ::new (static_cast<void*>(__addr())) _Tp(std::forward<_Args>(__args)...);
    }

    __device__ _LIBGPU_CONSTEXPR_SINCE_CXX20 ~__temp_value() { __addr()->~_Tp(); }
};


} // namespace gpu

#endif // __GPU___MEMORY_TEMP_VALUE_H__
