// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

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

#ifndef __GPU___ITERATOR_OSTREAMBUF_ITERATOR_H
#define __GPU___ITERATOR_OSTREAMBUF_ITERATOR_H

#include "gpu/__config"
#include <iterator>

namespace gpu {

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _CharT, class _Traits>
class _LIBGPU_TEMPLATE_VIS ostreambuf_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public std::iterator<std::output_iterator_tag, void, void, void, void>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
public:
    typedef std::output_iterator_tag                 iterator_category;
    typedef void                                value_type;
#if _LIBGPU_STD_VER >= 20
    typedef std::ptrdiff_t                           difference_type;
#else
    typedef void                                difference_type;
#endif
    typedef void                                pointer;
    typedef void                                reference;
    typedef _CharT                              char_type;
    typedef _Traits                             traits_type;
    typedef basic_streambuf<_CharT, _Traits>    streambuf_type;
    typedef basic_ostream<_CharT, _Traits>      ostream_type;

private:
    streambuf_type* __sbuf_;
public:
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator(ostream_type& __s) _NOEXCEPT
        : __sbuf_(__s.rdbuf()) {}
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator(streambuf_type* __s) _NOEXCEPT
        : __sbuf_(__s) {}
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator& operator=(_CharT __c)
        {
            if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sputc(__c), traits_type::eof()))
                __sbuf_ = nullptr;
            return *this;
        }
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator& operator*()     {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator& operator++()    {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY ostreambuf_iterator& operator++(int) {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY bool failed() const _NOEXCEPT {return __sbuf_ == nullptr;}

    template <class _Ch, class _Tr>
    __device__ friend
    _LIBGPU_HIDE_FROM_ABI
    ostreambuf_iterator<_Ch, _Tr>
    __pad_and_output(ostreambuf_iterator<_Ch, _Tr> __s,
                     const _Ch* __ob, const _Ch* __op, const _Ch* __oe,
                     ios_base& __iob, _Ch __fl);
};

} // namespace gpu

#endif // __GPU___ITERATOR_OSTREAMBUF_ITERATOR_H
