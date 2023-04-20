// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_ISTREAMBUF_ITERATOR_H
#define __GPU___ITERATOR_ISTREAMBUF_ITERATOR_H

#include "gpu/__config"
#include <iterator>

namespace gpu {

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template<class _CharT, class _Traits>
class _LIBGPU_TEMPLATE_VIS istreambuf_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public std::iterator<std::input_iterator_tag, _CharT,
                      typename _Traits::off_type, _CharT*,
                      _CharT>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
public:
    typedef std::input_iterator_tag              iterator_category;
    typedef _CharT                          value_type;
    typedef typename _Traits::off_type      difference_type;
    typedef _CharT*                         pointer;
    typedef _CharT                          reference;
    typedef _CharT                          char_type;
    typedef _Traits                         traits_type;
    typedef typename _Traits::int_type      int_type;
    typedef basic_streambuf<_CharT,_Traits> streambuf_type;
    typedef basic_istream<_CharT,_Traits>   istream_type;
private:
    mutable streambuf_type* __sbuf_;

    class __proxy
    {
        char_type __keep_;
        streambuf_type* __sbuf_;
        __device__ _LIBGPU_INLINE_VISIBILITY
        explicit __proxy(char_type __c, streambuf_type* __s)
            : __keep_(__c), __sbuf_(__s) {}
        friend class istreambuf_iterator;
    public:
        __device__ _LIBGPU_INLINE_VISIBILITY char_type operator*() const {return __keep_;}
    };

    __device__ _LIBGPU_INLINE_VISIBILITY
    bool __test_for_eof() const
    {
        if (__sbuf_ && traits_type::eq_int_type(__sbuf_->sgetc(), traits_type::eof()))
            __sbuf_ = nullptr;
        return __sbuf_ == nullptr;
    }
public:
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR istreambuf_iterator() _NOEXCEPT : __sbuf_(nullptr) {}
#if _LIBGPU_STD_VER >= 20
    __device__ _LIBGPU_INLINE_VISIBILITY constexpr istreambuf_iterator(default_sentinel_t) noexcept
        : istreambuf_iterator() {}
#endif // _LIBGPU_STD_VER >= 20
    __device__ _LIBGPU_INLINE_VISIBILITY istreambuf_iterator(istream_type& __s) _NOEXCEPT
        : __sbuf_(__s.rdbuf()) {}
    __device__ _LIBGPU_INLINE_VISIBILITY istreambuf_iterator(streambuf_type* __s) _NOEXCEPT
        : __sbuf_(__s) {}
    __device__ _LIBGPU_INLINE_VISIBILITY istreambuf_iterator(const __proxy& __p) _NOEXCEPT
        : __sbuf_(__p.__sbuf_) {}

    __device__ _LIBGPU_INLINE_VISIBILITY char_type  operator*() const
        {return static_cast<char_type>(__sbuf_->sgetc());}
    __device__ _LIBGPU_INLINE_VISIBILITY istreambuf_iterator& operator++()
        {
            __sbuf_->sbumpc();
            return *this;
        }
    __device__ _LIBGPU_INLINE_VISIBILITY __proxy              operator++(int)
        {
            return __proxy(__sbuf_->sbumpc(), __sbuf_);
        }

    __device__ _LIBGPU_INLINE_VISIBILITY bool equal(const istreambuf_iterator& __b) const
        {return __test_for_eof() == __b.__test_for_eof();}

#if _LIBGPU_STD_VER >= 20
    __device__ friend _LIBGPU_HIDE_FROM_ABI bool operator==(const istreambuf_iterator& __i, default_sentinel_t) {
      return __i.__test_for_eof();
    }
#endif // _LIBGPU_STD_VER >= 20
};

template <class _CharT, class _Traits>
__device__ inline _LIBGPU_INLINE_VISIBILITY
bool operator==(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return __a.equal(__b);}

#if _LIBGPU_STD_VER <= 17
template <class _CharT, class _Traits>
__device__ inline _LIBGPU_INLINE_VISIBILITY
bool operator!=(const istreambuf_iterator<_CharT,_Traits>& __a,
                const istreambuf_iterator<_CharT,_Traits>& __b)
                {return !__a.equal(__b);}
#endif // _LIBGPU_STD_VER <= 17

} // namespace gpu

#endif // __GPU___ITERATOR_ISTREAMBUF_ITERATOR_H
