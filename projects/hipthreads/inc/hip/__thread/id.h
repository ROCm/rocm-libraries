// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LIBHIPTHREADS___THREAD_ID_H
#define __LIBHIPTHREADS___THREAD_ID_H

#include <iostream>

#include "hip/thread_config"

/**
 * @file
 * @brief Lightweight GPU thread identifier type and helper I/O operator.
 * @ingroup thread
 *
 * Provides cuda::__thread_id, a trivially copyable identifier for a GPU fiber
 * produced by cuda::this_thread::get_id() and cuda::wthread::get_id(). 
 * Semantics mirror (a subset of) std::thread::id:
 *
 *  - Default constructed id compares equal only to other default ids.
 *  - Value 0 is a reserved sentinel used to impose a strict weak ordering
 *    where the sentinel is always the least element.
 *  - Comparison operators provide total ordering (C++20: three-way comparison).
 *  - Stream insertion operator writes a stable textual form (implementation detail).
 *
 * Notes:
 *  - Underlying representation is an opaque 32-bit value; no guarantees about
 *    reuse after a thread completes.
 *  - Equality of ids implies they refer (or referred) to the same logical
 *    execution context; inequality does not guarantee concurrency.
 *  - API is in flux. It may change in the future so all fibers in the same
 *    `hip::wthread` share the same id. A "sub-ID" may be introduced instead for
 *    identifying individual fibers.
 */

namespace cuda {

class _LIBHIPTHREADS_EXPORTED_FROM_ABI __thread_id;

namespace this_thread {

__device__ _LIBHIPTHREADS_HIDE_FROM_ABI __thread_id get_id() _NOEXCEPT;

} // namespace this_thread

namespace internal {
  class wthread;
  struct WorkNode_Header;
  struct ThreadData;
}

} // namespace cuda

template <class _CharT, class _Traits>
_LIBHIPTHREADS_HIDE_FROM_ABI ::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os, hip::__thread_id __id);

namespace cuda {

/**
 * @class __thread_id
 * @brief Opaque handle identifying a logical GPU thread (work node lane).
 * @ingroup thread
 *
 * Acts similarly to std::thread::id but for the hipThreads runtime. A default
 * constructed id (value 0) represents “no thread” and is always ordered
 * before any non‑default id. Instances are obtained via:
 *  - cuda::this_thread::get_id()
 *  - cuda::wthread::get_id()
 *
 * Ordering:
 *  - All non‑zero ids are ordered by underlying integral value.
 *  - Zero (sentinel) < any non‑zero.
 *
 * Lifetime:
 *  - An id may outlive the associated thread; equality remains valid for
 *    comparison but does not retain resources.
 */
class _LIBHIPTHREADS_TEMPLATE_VIS __thread_id {
  using underlying_type = uint32_t;
  underlying_type __id_;

  __host__ __device__ static _LIBHIPTHREADS_HIDE_FROM_ABI bool
  __lt_impl(__thread_id __x, __thread_id __y) _NOEXCEPT { // id==0 is always less than any other thread_id
    if (__x.__id_ == 0)
      return __y.__id_ != 0;
    if (__y.__id_ == 0)
      return false;
    return __x.__id_ < __y.__id_;
  }

public:

  /// Constructs a sentinel (non-joinable / no-thread) id (value 0).
  __host__ __device__ _LIBHIPTHREADS_HIDE_FROM_ABI __thread_id() _NOEXCEPT : __id_(0) {}

  /// Resets this id back to the sentinel (value 0).
  __host__ __device__ _LIBHIPTHREADS_HIDE_FROM_ABI void __reset() { __id_ = 0; }

  __host__ __device__ friend _LIBHIPTHREADS_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  if _LIBHIPTHREADS_STD_VER <= 17
  __host__ __device__ friend _LIBHIPTHREADS_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT;
#  else  // _LIBHIPTHREADS_STD_VER <= 17
  __host__ __device__ friend _LIBHIPTHREADS_HIDE_FROM_ABI ::std::strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept;
#  endif // _LIBHIPTHREADS_STD_VER <= 17

  template <class _CharT, class _Traits>
  friend _LIBHIPTHREADS_HIDE_FROM_ABI ::std::basic_ostream<_CharT, _Traits>&
  ::operator<<(::std::basic_ostream<_CharT, _Traits>& __os, __thread_id __id);

private:
  __host__ __device__ _LIBHIPTHREADS_HIDE_FROM_ABI __thread_id(underlying_type __id) : __id_(__id) {}

  __host__ __device__ _LIBHIPTHREADS_HIDE_FROM_ABI friend underlying_type __get_underlying_id(const __thread_id __id) { return __id.__id_; }

  friend __device__ __thread_id this_thread::get_id() _NOEXCEPT;
  friend class internal::wthread;
  friend struct internal::ThreadData;
};

__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator==(__thread_id __x, __thread_id __y) _NOEXCEPT {
  // Don't pass id==0 to underlying routines
  if (__x.__id_ == 0)
    return __y.__id_ == 0;
  if (__y.__id_ == 0)
    return false;
  return __x.__id_ == __y.__id_;
}

#  if _LIBHIPTHREADS_STD_VER <= 17

__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator!=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x == __y); }

__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator<(__thread_id __x, __thread_id __y) _NOEXCEPT {
  return __thread_id::__lt_impl(__x, __y);
}

__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator<=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__y < __x); }
__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator>(__thread_id __x, __thread_id __y) _NOEXCEPT { return __y < __x; }
__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI bool operator>=(__thread_id __x, __thread_id __y) _NOEXCEPT { return !(__x < __y); }

#  else // _LIBHIPTHREADS_STD_VER <= 17

__host__ __device__ inline _LIBHIPTHREADS_HIDE_FROM_ABI ::std::strong_ordering operator<=>(__thread_id __x, __thread_id __y) noexcept {
  if (__x == __y)
    return ::std::strong_ordering::equal;
  if (__thread_id::__lt_impl(__x, __y))
    return ::std::strong_ordering::less;
  return ::std::strong_ordering::greater;
}

#  endif // _LIBHIPTHREADS_STD_VER <= 17

} // namespace cuda

/**
 * @brief Stream insertion for cuda::__thread_id.
 * @ingroup thread
 *
 * Produces a textual representation; distinct ids yield distinct text,
 * equal ids yield identical text (except default constructed which prints 0).
 */
template <class _CharT, class _Traits>
_LIBHIPTHREADS_HIDE_FROM_ABI ::std::basic_ostream<_CharT, _Traits>&
operator<<(::std::basic_ostream<_CharT, _Traits>& __os, hip::__thread_id __id) {
  // [thread.thread.id]/9
  //   Effects: Inserts the text representation for charT of id into out.
  //
  // [thread.thread.id]/2
  //   The text representation for the character type charT of an
  //   object of type wthread::id is an unspecified sequence of charT
  //   such that, for two objects of type wthread::id x and y, if
  //   x == y is true, the wthread::id objects have the same text
  //   representation, and if x != y is true, the wthread::id objects
  //   have distinct text representations.
  //
  // Since various flags in the output stream can affect how the
  // thread id is represented (e.g. numpunct or showbase), we
  // use a temporary stream instead and just output the thread
  // id representation as a string.

  ::std::basic_ostringstream<_CharT, _Traits> __sstr;
  __sstr.imbue(::std::locale::classic());
  __sstr << __id.__id_;
  return __os << __sstr.str();
}

#endif // __LIBHIPTHREADS___THREAD_ID_H
