// -*- C++ -*-

// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef __GPU___FUNCTIONAL_OPERATIONS_H__
#define __GPU___FUNCTIONAL_OPERATIONS_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __functional/operations.h
//====================================================================================================================//

#if _LIBGPU_STD_VER >= 14
template <class _Tp = void>
#else
template <class _Tp>
#endif
struct _LIBGPU_TEMPLATE_VIS less
{
    typedef bool __result_type;  // used by valarray
    __host__ __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14 _LIBGPU_INLINE_VISIBILITY
    bool operator()(const _Tp& __x, const _Tp& __y) const
        {return __x < __y;}
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(less);

#if _LIBGPU_STD_VER >= 14
template <>
struct _LIBGPU_TEMPLATE_VIS less<void>
{
    template <class _T1, class _T2>
    __host__ __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14 _LIBGPU_INLINE_VISIBILITY
    auto operator()(_T1&& __t, _T2&& __u) const
        noexcept(noexcept(std::forward<_T1>(__t) < std::forward<_T2>(__u)))
        -> decltype(      std::forward<_T1>(__t) < std::forward<_T2>(__u))
        { return          std::forward<_T1>(__t) < std::forward<_T2>(__u); }
    typedef void is_transparent;
};
#endif


}

#endif // __GPU___FUNCTIONAL_OPERATIONS_H__
