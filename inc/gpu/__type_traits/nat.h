#ifndef __GPU___TYPE_TRAITS_NAT_H__
#define __GPU___TYPE_TRAITS_NAT_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ __type_traits/nat.h
//====================================================================================================================//

struct __nat
{
#ifndef _LIBGPU_CXX03_LANG
    __nat() = delete;
    __nat(const __nat&) = delete;
    __nat& operator=(const __nat&) = delete;
    ~__nat() = delete;
#endif
};

} // namespace gpu

#endif // __GPU___TYPE_TRAITS_NAT_H__