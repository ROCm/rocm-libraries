// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___THREAD_ID_H
#define __GPU___THREAD_ID_H

#include <iostream>

#include "gpu/__config"

namespace gpu {

class _LIBGPU_EXPORTED_FROM_ABI __thread_id;

namespace this_thread {

__device__ _LIBGPU_HIDE_FROM_ABI __thread_id get_id() _NOEXCEPT;

} // namespace this_thread

namespace internal {
  class thread;
  struct WorkNode_Header;
  struct ThreadData;
}

} // namespace gpu

template <class _CharT, class _Traits>
_LIBGPU_HIDE_FROM_ABI std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, gpu::__thread_id __id);

namespace gpu {

class _LIBGPU_TEMPLATE_VIS __thread_id {
  using underlying_type = uint32_t;
  underlying_type __id_;

  __host__ __device__ static _LIBGPU_HIDE_FROM_ABI bool
  __lt_impl(__thread_id __x, __thread_id __y) _NOEXCEPT { // id==0 is always less than any other thread_id
    if (__x.__id_ == 0)
      return __y.__id_ != 0;
    if (__y.__id_ == 0)
      return false;
    return __x.__id_ < __y.__id_;
  }

public:
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI __thread_id() _NOEXCEPT : __id_(0) {}

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI void __reset() { __id_ = 0; }

  __host__ __device__ friend _LIBGPU_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  if _LIBGPU_STD_VER <= 17
  __host__ __device__ friend _LIBGPU_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  else  // _LIBGPU_STD_VER <= 17
  __host__ __device__ friend _LIBGPU_HIDE_FROM_ABI std::strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept;
#  endif // _LIBGPU_STD_VER <= 17

  template <class _CharT, class _Traits>
  friend _LIBGPU_HIDE_FROM_ABI std::basic_ostream<_CharT, _Traits>&
  ::operator<<(std::basic_ostream<_CharT, _Traits>& __os, __thread_id __id);

private:
  __host__ __device__ _LIBGPU_HIDE_FROM_ABI __thread_id(underlying_type __id) : __id_(__id) {}

  __host__ __device__ _LIBGPU_HIDE_FROM_ABI friend underlying_type __get_underlying_id(const __thread_id __id) { return __id.__id_; }

  friend __device__ __thread_id this_thread::get_id() _NOEXCEPT;
  friend class internal::thread;
  friend struct internal::ThreadData;
};

__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT {
  // Don't pass id==0 to underlying routines
  if (__x.__id_ == 0)
    return __y.__id_ == 0;
  if (__y.__id_ == 0)
    return false;
  return __x.__id_ == __y.__id_;
}

#  if _LIBGPU_STD_VER <= 17

__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator!=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x == __y); }

__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT {
  return __thread_id::__lt_impl(__x, __y);
}

__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator<=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__y < __x); }
__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator>(__thread_id __x, __thread_id __y) _NOEXCEPT { return __y < __x; }
__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI bool operator>=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x < __y); }

#  else // _LIBGPU_STD_VER <= 17

__host__ __device__ inline _LIBGPU_HIDE_FROM_ABI std::strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept {
  if (__x == __y)
    return std::strong_ordering::equal;
  if (__thread_id::__lt_impl(__x, __y))
    return std::strong_ordering::less;
  return std::strong_ordering::greater;
}

#  endif // _LIBGPU_STD_VER <= 17

} // namespace gpu

template <class _CharT, class _Traits>
_LIBGPU_HIDE_FROM_ABI std::basic_ostream<_CharT, _Traits>&
operator<<(std::basic_ostream<_CharT, _Traits>& __os, gpu::__thread_id __id) {
  // [thread.thread.id]/9
  //   Effects: Inserts the text representation for charT of id into out.
  //
  // [thread.thread.id]/2
  //   The text representation for the character type charT of an
  //   object of type thread::id is an unspecified sequence of charT
  //   such that, for two objects of type thread::id x and y, if
  //   x == y is true, the thread::id objects have the same text
  //   representation, and if x != y is true, the thread::id objects
  //   have distinct text representations.
  //
  // Since various flags in the output stream can affect how the
  // thread id is represented (e.g. numpunct or showbase), we
  // use a temporary stream instead and just output the thread
  // id representation as a string.

  std::basic_ostringstream<_CharT, _Traits> __sstr;
  __sstr.imbue(std::locale::classic());
  __sstr << __id.__id_;
  return __os << __sstr.str();
}

#endif // __GPU___THREAD_ID_H
