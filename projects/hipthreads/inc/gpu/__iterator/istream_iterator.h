// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ITERATOR_ISTREAM_ITERATOR_H
#define __GPU___ITERATOR_ISTREAM_ITERATOR_H

#include "gpu/__config"

namespace gpu {

_LIBGPU_SUPPRESS_DEPRECATED_PUSH
template <class _Tp, class _CharT = char,
          class _Traits = char_traits<_CharT>, class _Distance = std::ptrdiff_t>
class _LIBGPU_TEMPLATE_VIS istream_iterator
#if _LIBGPU_STD_VER <= 14 || !defined(_LIBGPU_ABI_NO_ITERATOR_BASES)
    : public std::iterator<std::input_iterator_tag, _Tp, _Distance, const _Tp*, const _Tp&>
#endif
{
_LIBGPU_SUPPRESS_DEPRECATED_POP
public:
    typedef std::input_iterator_tag iterator_category;
    typedef _Tp value_type;
    typedef _Distance difference_type;
    typedef const _Tp* pointer;
    typedef const _Tp& reference;
    typedef _CharT char_type;
    typedef _Traits traits_type;
    typedef basic_istream<_CharT,_Traits> istream_type;
private:
    istream_type* __in_stream_;
    _Tp __value_;
public:
    __device__ _LIBGPU_INLINE_VISIBILITY _LIBGPU_CONSTEXPR istream_iterator() : __in_stream_(nullptr), __value_() {}
#if _LIBGPU_STD_VER >= 20
    __device__ _LIBGPU_HIDE_FROM_ABI constexpr istream_iterator(default_sentinel_t) : istream_iterator() {}
#endif // _LIBGPU_STD_VER >= 20
    __device__ _LIBGPU_INLINE_VISIBILITY istream_iterator(istream_type& __s) : __in_stream_(gpu::addressof(__s))
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = nullptr;
        }

    __device__ _LIBGPU_INLINE_VISIBILITY const _Tp& operator*() const {return __value_;}
    __device__ _LIBGPU_INLINE_VISIBILITY const _Tp* operator->() const {return gpu::addressof((operator*()));}
    __device__ _LIBGPU_INLINE_VISIBILITY istream_iterator& operator++()
        {
            if (!(*__in_stream_ >> __value_))
                __in_stream_ = nullptr;
            return *this;
        }
    __device__ _LIBGPU_INLINE_VISIBILITY istream_iterator  operator++(int)
        {istream_iterator __t(*this); ++(*this); return __t;}

    template <class _Up, class _CharU, class _TraitsU, class _DistanceU>
    __device__ friend _LIBGPU_INLINE_VISIBILITY
    bool
    operator==(const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __x,
               const istream_iterator<_Up, _CharU, _TraitsU, _DistanceU>& __y);

#if _LIBGPU_STD_VER >= 20
    __device__ friend _LIBGPU_HIDE_FROM_ABI bool operator==(const istream_iterator& __i, default_sentinel_t) {
      return __i.__in_stream_ == nullptr;
    }
#endif // _LIBGPU_STD_VER >= 20
};

template <class _Tp, class _CharT, class _Traits, class _Distance>
__device__ inline _LIBGPU_INLINE_VISIBILITY
bool
operator==(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return __x.__in_stream_ == __y.__in_stream_;
}

#if _LIBGPU_STD_VER <= 17
template <class _Tp, class _CharT, class _Traits, class _Distance>
__device__ inline _LIBGPU_INLINE_VISIBILITY
bool
operator!=(const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __x,
           const istream_iterator<_Tp, _CharT, _Traits, _Distance>& __y)
{
    return !(__x == __y);
}
#endif // _LIBGPU_STD_VER <= 17

} // namespace gpu

#endif // __GPU___ITERATOR_ISTREAM_ITERATOR_H
