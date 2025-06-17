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

#ifndef __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__
#define __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::type_identity
//====================================================================================================================//

template <class _Tp>
struct __type_identity { typedef _Tp type; };

template <class _Tp>
using __type_identity_t _LIBGPU_NODEBUG = typename __type_identity<_Tp>::type;

#if _LIBGPU_STD_VER >= 20
template<class _Tp> struct type_identity { typedef _Tp type; };
template<class _Tp> using type_identity_t = typename type_identity<_Tp>::type;
#endif

}

#endif // __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__
