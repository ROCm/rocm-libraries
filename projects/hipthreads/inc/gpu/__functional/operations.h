#ifndef __GPU___FUNCTIONAL_OPERATIONS_H__
#define __GPU___FUNCTIONAL_OPERATIONS_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __functional/operations.h
//====================================================================================================================//

#if _LIBGPU_STD_VER >= 14
template <class _Tp = void>
#else
template <class _Tp>
#endif
struct _LIBGPU_TEMPLATE_VIS less
{
    typedef bool __result_type;  // used by valarray
    __host__ __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14 _LIBGPU_INLINE_VISIBILITY
    bool operator()(const _Tp& __x, const _Tp& __y) const
        {return __x < __y;}
};
_LIBGPU_CTAD_SUPPORTED_FOR_TYPE(less);

#if _LIBGPU_STD_VER >= 14
template <>
struct _LIBGPU_TEMPLATE_VIS less<void>
{
    template <class _T1, class _T2>
    __host__ __device__ _LIBGPU_CONSTEXPR_SINCE_CXX14 _LIBGPU_INLINE_VISIBILITY
    auto operator()(_T1&& __t, _T2&& __u) const
        noexcept(noexcept(std::forward<_T1>(__t) < std::forward<_T2>(__u)))
        -> decltype(      std::forward<_T1>(__t) < std::forward<_T2>(__u))
        { return          std::forward<_T1>(__t) < std::forward<_T2>(__u); }
    typedef void is_transparent;
};
#endif


}

#endif // __GPU___FUNCTIONAL_OPERATIONS_H__
