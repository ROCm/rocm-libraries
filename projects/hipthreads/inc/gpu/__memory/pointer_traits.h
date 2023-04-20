// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___MEMORY_POINTER_TRAITS_H__
#define __GPU___MEMORY_POINTER_TRAITS_H__

#include "gpu/__config"

#include <type_traits>

#include "gpu/__utility/declval.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::to_address
//====================================================================================================================//

template <class _Pointer, class = void>
struct __to_address_helper;

template <class _Tp>
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
_Tp* __to_address(_Tp* __p) _NOEXCEPT {
    static_assert(!std::is_function<_Tp>::value, "_Tp is a function type");
    return __p;
}

// TODO: Uncomment this when we add pointer_traits
// template <class _Pointer, class = void>
// struct _HasToAddress : std::false_type {};
//
// template <class _Pointer>
// struct _HasToAddress<_Pointer,
//     decltype((void)pointer_traits<_Pointer>::to_address(gpu::declval<const _Pointer&>()))
// > : std::true_type {};

template <class _Pointer, class = void>
struct _HasArrow : std::false_type {};

template <class _Pointer>
struct _HasArrow<_Pointer,
    decltype((void)gpu::declval<const _Pointer&>().operator->())
> : std::true_type {};

template <class _Pointer>
struct _IsFancyPointer {
  // TODO: Uncomment this when we add pointer_traits
  static const bool value = _HasArrow<_Pointer>::value /* || _HasToAddress<_Pointer>::value */;
};

// std::enable_if is needed here to avoid instantiating checks for fancy pointers on raw pointers
template <class _Pointer, class = std::enable_if_t<
    std::conjunction<std::is_class<_Pointer>, _IsFancyPointer<_Pointer> >::value
> >
__device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
std::decay_t<decltype(__to_address_helper<_Pointer>::__call(gpu::declval<const _Pointer&>()))>
__to_address(const _Pointer& __p) _NOEXCEPT {
    return __to_address_helper<_Pointer>::__call(__p);
}

template <class _Pointer, class>
struct __to_address_helper {
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
    static decltype(_VSTD::__to_address(gpu::declval<const _Pointer&>().operator->()))
    __call(const _Pointer& __p) _NOEXCEPT {
        return _VSTD::__to_address(__p.operator->());
    }
};

// TODO: Uncomment this when we add pointer_traits
// template <class _Pointer>
// struct __to_address_helper<_Pointer, decltype((void)pointer_traits<_Pointer>::to_address(gpu::declval<const _Pointer&>()))> {
//     __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR
//     static decltype(pointer_traits<_Pointer>::to_address(gpu::declval<const _Pointer&>()))
//     __call(const _Pointer& __p) _NOEXCEPT {
//         return pointer_traits<_Pointer>::to_address(__p);
//     }
// };

#if _LIBGPU_STD_VER >= 20
template <class _Tp>
__device__ inline _LIBGPU_INLINE_VISIBILITY constexpr
auto to_address(_Tp *__p) noexcept {
    return _VSTD::__to_address(__p);
}

template <class _Pointer>
__device__ inline _LIBGPU_INLINE_VISIBILITY constexpr
auto to_address(const _Pointer& __p) noexcept -> decltype(std::__to_address(__p)) {
    return _VSTD::__to_address(__p);
}
#endif

} // namespace gpu

#endif // __GPU___MEMORY_POINTER_TRAITS_H__
