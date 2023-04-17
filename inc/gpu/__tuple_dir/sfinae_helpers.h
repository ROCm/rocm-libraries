#ifndef __GPU___TUPLE_DIR_SFINAE_HELPERS_H__
#define __GPU___TUPLE_DIR_SFINAE_HELPERS_H__

#include "gpu/__config"

namespace gpu {

//====================================================================================================================//
//      Adapted from libc++ std::sfinae_helpers
//====================================================================================================================//

struct _LIBGPU_TYPE_VIS __check_tuple_constructor_fail {

    static constexpr bool __enable_explicit_default() { return false; }
    static constexpr bool __enable_implicit_default() { return false; }
    template <class ...>
    static constexpr bool __enable_explicit() { return false; }
    template <class ...>
    static constexpr bool __enable_implicit() { return false; }
    template <class ...>
    static constexpr bool __enable_assign() { return false; }
};

} // namespace gpu

#endif // __GPU___TUPLE_DIR_SFINAE_HELPERS_H__