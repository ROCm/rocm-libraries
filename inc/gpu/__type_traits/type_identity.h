#ifndef __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__
#define __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::type_identity
//====================================================================================================================//

template <class _Tp>
struct __type_identity { typedef _Tp type; };

template <class _Tp>
using __type_identity_t _LIBGPU_NODEBUG = typename __type_identity<_Tp>::type;

#if _LIBGPU_STD_VER >= 20
template<class _Tp> struct type_identity { typedef _Tp type; };
template<class _Tp> using type_identity_t = typename type_identity<_Tp>::type;
#endif

}

#endif // __GPU___TYPE_TRAITS_TYPE_IDENTITY_H__
