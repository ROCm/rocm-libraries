#ifndef __GPU___FUNCTIONAL_IDENTITY_H__
#define __GPU___FUNCTIONAL_IDENTITY_H__

#include "gpu/__config"
#include <type_traits>

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::identity
//====================================================================================================================//

template <class _Tp>
struct __is_identity : std::false_type {};

struct __identity {
  template <class _Tp>
  _LIBGPU_NODISCARD _LIBGPU_CONSTEXPR _Tp&& operator()(_Tp&& __t) const _NOEXCEPT {
    return std::forward<_Tp>(__t);
  }

  using is_transparent = void;
};

template <>
struct __is_identity<__identity> : std::true_type {};

#if _LIBGPU_STD_VER >= 20

struct identity {
    template<class _Tp>
    _LIBGPU_NODISCARD_EXT constexpr _Tp&& operator()(_Tp&& __t) const noexcept
    {
        return std::forward<_Tp>(__t);
    }

    using is_transparent = void;
};

template <>
struct __is_identity<identity> : std::true_type {};

#endif // _LIBGPU_STD_VER >= 20


} // namespace gpu

#endif // __GPU___FUNCTIONAL_IDENTITY_H__
