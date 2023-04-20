// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_OSTREAM_ITERATOR_H
#define __GPU___ITERATOR_OSTREAM_ITERATOR_H

#include "gpu/__config"
#include <iterator>

namespace gpu {

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char, class _Traits = char_traits<_CharT> >
class _LIBGPU_TEMPLATE_VIS ostream_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public std::iterator<std::output_iterator_tag, void, void, void, void>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
public:
    typedef std::output_iterator_tag             iterator_category;
    typedef void                            value_type;
#if _LIBGPU_STD_VER >= 20
    typedef std::ptrdiff_t                       difference_type;
#else
    typedef void                            difference_type;
#endif
    typedef void                            pointer;
    typedef void                            reference;
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef basic_ostream<_CharT, _Traits>  ostream_type;

private:
    ostream_type* __out_stream_;
    const char_type* __delim_;
public:
    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator(ostream_type& __s) _NOEXCEPT
        : __out_stream_(gpu::addressof(__s)), __delim_(nullptr) {}
    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator(ostream_type& __s, const _CharT* __delimiter) _NOEXCEPT
        : __out_stream_(gpu::addressof(__s)), __delim_(__delimiter) {}
    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator& operator=(const _Tp& __value)
        {
            *__out_stream_ << __value;
            if (__delim_)
                *__out_stream_ << __delim_;
            return *this;
        }

    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator& operator*()     {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator& operator++()    {return *this;}
    __device__ _LIBGPU_INLINE_VISIBILITY ostream_iterator& operator++(int) {return *this;}
};

} // namespace gpu

#endif // __GPU___ITERATOR_OSTREAM_ITERATOR_H
