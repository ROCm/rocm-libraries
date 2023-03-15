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
