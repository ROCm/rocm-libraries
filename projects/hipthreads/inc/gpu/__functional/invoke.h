// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___FUNCTIONAL_INVOKE_H__
#define __GPU___FUNCTIONAL_INVOKE_H__

#include "gpu/__config"

#include <type_traits>
#include "gpu/__type_traits/is_reference_wrapper.h"
#include "gpu/__type_traits/nat.h"
#include "gpu/__utility/declval.h"

namespace gpu {

struct __any
{
    __any(...);
};

template <class _DecayedFp>
struct __member_pointer_class_type {};

template <class _Ret, class _ClassType>
struct __member_pointer_class_type<_Ret _ClassType::*> {
  typedef _ClassType type;
};

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0>,
         class _ClassT = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet1 = typename std::enable_if
    <
        std::is_member_function_pointer<_DecayFp>::value
        && std::is_base_of<_ClassT, _DecayA0>::value
    >::type;

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0> >
using __enable_if_bullet2 = typename std::enable_if
    <
        std::is_member_function_pointer<_DecayFp>::value
        && __is_reference_wrapper<_DecayA0>::value
    >::type;

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0>,
         class _ClassT = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet3 = typename std::enable_if
    <
        std::is_member_function_pointer<_DecayFp>::value
        && !std::is_base_of<_ClassT, _DecayA0>::value
        && !__is_reference_wrapper<_DecayA0>::value
    >::type;

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0>,
         class _ClassT = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet4 = typename std::enable_if
    <
        std::is_member_object_pointer<_DecayFp>::value
        && std::is_base_of<_ClassT, _DecayA0>::value
    >::type;

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0> >
using __enable_if_bullet5 = typename std::enable_if
    <
        std::is_member_object_pointer<_DecayFp>::value
        && __is_reference_wrapper<_DecayA0>::value
    >::type;

template <class _Fp, class _A0,
         class _DecayFp = std::decay_t<_Fp>,
         class _DecayA0 = std::decay_t<_A0>,
         class _ClassT = typename __member_pointer_class_type<_DecayFp>::type>
using __enable_if_bullet6 = typename std::enable_if
    <
        std::is_member_object_pointer<_DecayFp>::value
        && !std::is_base_of<_ClassT, _DecayA0>::value
        && !__is_reference_wrapper<_DecayA0>::value
    >::type;

// __invoke forward declarations

// fall back - none of the bullets

template <class ..._Args>
__nat __invoke(__any, _Args&& ...__args);

// bullets 1, 2 and 3

template <class _Fp, class _A0, class ..._Args,
          class = __enable_if_bullet1<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype((gpu::declval<_A0>().*gpu::declval<_Fp>())(gpu::declval<_Args>()...))
__invoke(_Fp&& __f, _A0&& __a0, _Args&& ...__args)
    _NOEXCEPT_(noexcept((static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...)))
    { return           (static_cast<_A0&&>(__a0).*__f)(static_cast<_Args&&>(__args)...); }

template <class _Fp, class _A0, class ..._Args,
          class = __enable_if_bullet2<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype((gpu::declval<_A0>().get().*gpu::declval<_Fp>())(gpu::declval<_Args>()...))
__invoke(_Fp&& __f, _A0&& __a0, _Args&& ...__args)
    _NOEXCEPT_(noexcept((__a0.get().*__f)(static_cast<_Args&&>(__args)...)))
    { return          (__a0.get().*__f)(static_cast<_Args&&>(__args)...); }

template <class _Fp, class _A0, class ..._Args,
          class = __enable_if_bullet3<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype(((*gpu::declval<_A0>()).*gpu::declval<_Fp>())(gpu::declval<_Args>()...))
__invoke(_Fp&& __f, _A0&& __a0, _Args&& ...__args)
    _NOEXCEPT_(noexcept(((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...)))
    { return          ((*static_cast<_A0&&>(__a0)).*__f)(static_cast<_Args&&>(__args)...); }

// bullets 4, 5 and 6

template <class _Fp, class _A0,
          class = __enable_if_bullet4<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype(gpu::declval<_A0>().*gpu::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0)
    _NOEXCEPT_(noexcept(static_cast<_A0&&>(__a0).*__f))
    { return          static_cast<_A0&&>(__a0).*__f; }

template <class _Fp, class _A0,
          class = __enable_if_bullet5<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype(gpu::declval<_A0>().get().*gpu::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0)
    _NOEXCEPT_(noexcept(__a0.get().*__f))
    { return          __a0.get().*__f; }

template <class _Fp, class _A0,
          class = __enable_if_bullet6<_Fp, _A0> >
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype((*gpu::declval<_A0>()).*gpu::declval<_Fp>())
__invoke(_Fp&& __f, _A0&& __a0)
    _NOEXCEPT_(noexcept((*static_cast<_A0&&>(__a0)).*__f))
    { return          (*static_cast<_A0&&>(__a0)).*__f; }

// bullet 7

template <class _Fp, class ..._Args>
__device__ inline _LIBGPU_INLINE_VISIBILITY
_LIBGPU_CONSTEXPR decltype(gpu::declval<_Fp>()(gpu::declval<_Args>()...))
__invoke(_Fp&& __f, _Args&& ...__args)
    _NOEXCEPT_(noexcept(static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...)))
    { return          static_cast<_Fp&&>(__f)(static_cast<_Args&&>(__args)...); }


template <class _Ret, bool = std::is_void<_Ret>::value>
struct __invoke_void_return_wrapper
{
    template <class ..._Args>
    __device__ _LIBGPU_HIDE_FROM_ABI static _Ret __call(_Args&&... __args) {
        return std::__invoke(std::forward<_Args>(__args)...);
    }
};

template <class _Ret>
struct __invoke_void_return_wrapper<_Ret, true>
{
    template <class ..._Args>
    __device__ _LIBGPU_HIDE_FROM_ABI static void __call(_Args&&... __args) {
        std::__invoke(std::forward<_Args>(__args)...);
    }
};

#if _LIBGPU_STD_VER >= 17

template <class _Fn, class ..._Args>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 std::invoke_result_t<_Fn, _Args...>
invoke(_Fn&& __f, _Args&&... __args)
    noexcept(std::is_nothrow_invocable_v<_Fn, _Args...>)
{
    return gpu::__invoke(std::forward<_Fn>(__f), std::forward<_Args>(__args)...);
}

#endif // _LIBGPU_STD_VER >= 17

#if _LIBGPU_STD_VER >= 23
template <class _Result, class _Fn, class... _Args>
  requires std::is_invocable_r_v<_Result, _Fn, _Args...>
__device__ _LIBGPU_HIDE_FROM_ABI constexpr _Result
invoke_r(_Fn&& __f, _Args&&... __args) noexcept(std::is_nothrow_invocable_r_v<_Result, _Fn, _Args...>) {
    if constexpr (std::is_void_v<_Result>) {
        static_cast<void>(gpu::invoke(std::forward<_Fn>(__f), std::forward<_Args>(__args)...));
    } else {
        // TODO: Use std::reference_converts_from_temporary_v once implemented
        // using _ImplicitInvokeResult = std::invoke_result_t<_Fn, _Args...>;
        // static_assert(!std::reference_converts_from_temporary_v<_Result, _ImplicitInvokeResult>,
        static_assert(true,
            "Returning from invoke_r would bind a temporary object to the reference return type, "
            "which would result in a dangling reference.");
        return gpu::invoke(std::forward<_Fn>(__f), std::forward<_Args>(__args)...);
    }
}
#endif

} // namespace gpu

#endif // __GPU___FUNCTIONAL_INVOKE_H__
