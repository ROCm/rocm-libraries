#ifndef __GPU___MEMORY_UNINITIALIZED_ALGORITHMS_H__
#define __GPU___MEMORY_UNINITIALIZED_ALGORITHMS_H__

#include "gpu/__config"
#include <iterator>

#include "gpu/__algorithm/copy.h"
#include "gpu/__algorithm/move.h"
#include "gpu/__memory/pointer_traits.h"
#include "gpu/__utility/exception_guard.h"
#include "gpu/__type_traits/is_trivially_relocatable.h"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __memory/uninitialized_algorithms.h
//====================================================================================================================//

// Destroy all elements in [__first, __last) from left to right using allocator destruction.
template <class _Iter, class _Sent>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 void
__allocator_destroy(_Iter __first, _Sent __last) {
  using value_type = typename std::iterator_traits<_Iter>::value_type;
  for (; __first != __last; ++__first)
    __first->~value_type();
}

template <class _Iter>
class _AllocatorDestroyRangeReverse {
public:
  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14
  _AllocatorDestroyRangeReverse(_Iter& __first, _Iter& __last)
      : __first_(__first), __last_(__last) {}

  __device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 void operator()() const {
    gpu::__allocator_destroy(std::reverse_iterator<_Iter>(__last_), std::reverse_iterator<_Iter>(__first_));
  }

private:
  _Iter& __first_;
  _Iter& __last_;
};

// Copy-construct [__first1, __last1) in [__first2, __first2 + N), where N is distance(__first1, __last1).
//
// The caller has to ensure that __first2 can hold at least N uninitialized elements. If an exception is thrown the
// already copied elements are destroyed in reverse order of their construction.
template <class _Iter1, class _Sent1, class _Iter2>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _Iter2
__uninitialized_allocator_copy(_Iter1 __first1, _Sent1 __last1, _Iter2 __first2) {
  using value_type = typename std::iterator_traits<_Iter2>::value_type;
  auto __destruct_first = __first2;
  auto __guard =
      gpu::__make_exception_guard(_AllocatorDestroyRangeReverse<_Iter2>(__destruct_first, __first2));
  while (__first1 != __last1) {
    ::new (gpu::__to_address(__first2)) value_type(*__first1);
    ++__first1;
    ++__first2;
  }
  __guard.__complete();
  return __first2;
}

template <class _Type,
          class _RawType = std::remove_const_t<_Type>,
          std::enable_if_t<
              // using _RawType because of the allocator<T const> extension
              std::is_trivially_copy_constructible<_RawType>::value && std::is_trivially_copy_assignable<_RawType>::value>* = nullptr>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX20 _Type*
__uninitialized_allocator_copy(const _Type* __first1, const _Type* __last1, _Type* __first2) {
  // TODO: Remove the const_cast once we drop support for std::allocator<T const>
  if (__builtin_is_constant_evaluated()) {
    while (__first1 != __last1) {
      ::new (gpu::__to_address(__first2)) _Type(*__first1);
      ++__first1;
      ++__first2;
    }
    return __first2;
  } else {
    return gpu::copy(__first1, __last1, const_cast<_RawType*>(__first2));
  }
}

// __uninitialized_allocator_relocate relocates the objects in [__first, __last) into __result.
// Relocation means that the objects in [__first, __last) are placed into __result as-if by move-construct and destroy,
// except that the move constructor and destructor may never be called if they are known to be equivalent to a memcpy.
//
// Preconditions:  __result doesn't contain any objects and [__first, __last) contains objects
// Postconditions: __result contains the objects from [__first, __last) and
//                 [__first, __last) doesn't contain any objects
//
// The strong exception guarantee is provided if any of the following are true:
// - is_nothrow_move_constructible<_Tp>
// - is_copy_constructible<_Tp>
// - __libcpp_is_trivially_relocatable<_Tp>
template <class _Tp>
__device__ _LIBGPU_HIDE_FROM_ABI _LIBGPU_CONSTEXPR_SINCE_CXX14 void
__uninitialized_allocator_relocate(_Tp* __first, _Tp* __last, _Tp* __result) {
  if (__builtin_is_constant_evaluated() || !__libcpp_is_trivially_relocatable<_Tp>::value) {
    auto __destruct_first = __result;
    auto __guard =
        gpu::__make_exception_guard(_AllocatorDestroyRangeReverse<_Tp*>(__destruct_first, __result));
    auto __iter = __first;
    while (__iter != __last) {
#ifndef _LIBGPU_HAS_NO_EXCEPTIONS
      ::new (__result) _Tp(std::move_if_noexcept(*__iter));
#else
      ::new (__result) _Tp(std::move(*__iter));
#endif
      ++__iter;
      ++__result;
    }
    __guard.__complete();
    gpu::__allocator_destroy(__first, __last);
  } else {
    __builtin_memcpy(const_cast<std::remove_const_t<_Tp>*>(__result), __first, sizeof(_Tp) * (__last - __first));
  }
}

} // namespace gpu

#endif // __GPU___MEMORY_UNINITIALIZED_ALGORITHMS_H__
