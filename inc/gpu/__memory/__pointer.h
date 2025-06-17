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

#ifndef __GPU___MEMORY___POINTER_H__
#define __GPU___MEMORY___POINTER_H__

#include <type_traits>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __pointer
//====================================================================================================================//

#define _LIBGPU_ALLOCATOR_TRAITS_HAS_XXX(NAME, PROPERTY)                \
    template <class _Tp, class = void> struct NAME : std::false_type { };    \
    template <class _Tp>               struct NAME<_Tp, std::void_t<typename _Tp:: PROPERTY > > : std::true_type { }

// __pointer
_LIBGPU_ALLOCATOR_TRAITS_HAS_XXX(__has_pointer, pointer);
template <class _Tp, class _Alloc,
          class _RawAlloc = std::remove_reference_t<_Alloc>,
          bool = __has_pointer<_RawAlloc>::value>
struct __pointer {
    using type = typename _RawAlloc::pointer;
};
template <class _Tp, class _Alloc, class _RawAlloc>
struct __pointer<_Tp, _Alloc, _RawAlloc, false> {
    using type = _Tp*;
};

}

#endif // __GPU___MEMORY___POINTER_H__
