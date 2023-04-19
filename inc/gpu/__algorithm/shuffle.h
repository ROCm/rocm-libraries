//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __GPU___ALGORITHM_SHUFFLE_H
#define __GPU___ALGORITHM_SHUFFLE_H

#include "gpu/__config"

namespace gpu {

class _LIBGPU_TYPE_VIS __LIBGPU_debug_randomizer {
public:
  __LIBGPU_debug_randomizer() {
    __state_ = __seed();
    __inc_ = __state_ + 0xda3e39cb94b95bdbULL;
    __inc_ = (__inc_ << 1) | 1;
  }
  typedef uint_fast32_t result_type;

  static const result_type _Min = 0;
  static const result_type _Max = 0xFFFFFFFF;

  __device__ _LIBGPU_HIDE_FROM_ABI result_type operator()() {
    uint_fast64_t __oldstate = __state_;
    __state_ = __oldstate * 6364136223846793005ULL + __inc_;
    return __oldstate >> 32;
  }

  __device__ static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR result_type min() { return _Min; }
  __device__ static _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR result_type max() { return _Max; }

private:
  uint_fast64_t __state_;
  uint_fast64_t __inc_;
  __device__ _LIBGPU_HIDE_FROM_ABI static uint_fast64_t __seed() {
#ifdef _LIBGPU_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY_SEED
    return _LIBGPU_DEBUG_RANDOMIZE_UNSPECIFIED_STABILITY_SEED;
#else
    static char __x;
    return reinterpret_cast<uintptr_t>(&__x);
#endif
  }
};

#if _LIBGPU_STD_VER <= 14 || defined(_LIBGPU_ENABLE_CXX17_REMOVED_RANDOM_SHUFFLE) \
  || defined(_LIBGPU_BUILDING_LIBRARY)
class _LIBGPU_TYPE_VIS __rs_default;

_LIBGPU_FUNC_VIS __rs_default __rs_get();

class _LIBGPU_TYPE_VIS __rs_default
{
    static unsigned __c_;

    __rs_default();
public:
    typedef uint_fast32_t result_type;

    static const result_type _Min = 0;
    static const result_type _Max = 0xFFFFFFFF;

    __rs_default(const __rs_default&);
    ~__rs_default();

    result_type operator()();

    static _LIBGPU_CONSTEXPR result_type min() {return _Min;}
    static _LIBGPU_CONSTEXPR result_type max() {return _Max;}

    friend _LIBGPU_FUNC_VIS __rs_default __rs_get();
};

_LIBGPU_FUNC_VIS __rs_default __rs_get();

template <class _RandomAccessIterator>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_DEPRECATED_IN_CXX14 void
random_shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last)
{
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type difference_type;
    typedef uniform_int_distribution<std::ptrdiff_t> _Dp;
    typedef typename _Dp::param_type _Pp;
    difference_type __d = __last - __first;
    if (__d > 1)
    {
        _Dp __uid;
        __rs_default __g = __rs_get();
        for (--__last, (void) --__d; __first < __last; ++__first, (void) --__d)
        {
            difference_type __i = __uid(__g, _Pp(0, __d));
            if (__i != difference_type(0))
                swap(*__first, *(__first + __i));
        }
    }
}

template <class _RandomAccessIterator, class _RandomNumberGenerator>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_DEPRECATED_IN_CXX14 void
random_shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last,
#ifndef _LIBGPU_CXX03_LANG
               _RandomNumberGenerator&& __rand)
#else
               _RandomNumberGenerator& __rand)
#endif
{
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type difference_type;
    difference_type __d = __last - __first;
    if (__d > 1)
    {
        for (--__last; __first < __last; ++__first, (void) --__d)
        {
            difference_type __i = __rand(__d);
            if (__i != difference_type(0))
              swap(*__first, *(__first + __i));
        }
    }
}
#endif

template <class _AlgPolicy, class _RandomAccessIterator, class _Sentinel, class _UniformRandomNumberGenerator>
__device__ _LIBGPU_HIDE_FROM_ABI _RandomAccessIterator __shuffle(
    _RandomAccessIterator __first, _Sentinel __last_sentinel, _UniformRandomNumberGenerator&& __g) {
    typedef typename std::iterator_traits<_RandomAccessIterator>::difference_type difference_type;
    typedef uniform_int_distribution<std::ptrdiff_t> _Dp;
    typedef typename _Dp::param_type _Pp;

    auto __original_last = _IterOps<_AlgPolicy>::next(__first, __last_sentinel);
    auto __last = __original_last;
    difference_type __d = __last - __first;
    if (__d > 1)
    {
        _Dp __uid;
        for (--__last, (void) --__d; __first < __last; ++__first, (void) --__d)
        {
            difference_type __i = __uid(__g, _Pp(0, __d));
            if (__i != difference_type(0))
                _IterOps<_AlgPolicy>::iter_swap(__first, __first + __i);
        }
    }

    return __original_last;
}

template <class _RandomAccessIterator, class _UniformRandomNumberGenerator>
__device__ _LIBGPU_HIDE_FROM_ABI void
shuffle(_RandomAccessIterator __first, _RandomAccessIterator __last, _UniformRandomNumberGenerator&& __g) {
  (void)gpu::__shuffle<_ClassicAlgPolicy>(
      std::move(__first), std::move(__last), std::forward<_UniformRandomNumberGenerator>(__g));
}

} // namespace gpu

#endif // __GPU___ALGORITHM_SHUFFLE_H
