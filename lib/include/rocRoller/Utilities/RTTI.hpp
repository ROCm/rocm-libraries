#pragma once

#include <string>

#if defined(__clang__)
#if __has_feature(cxx_rtti)
#define __RTTI_ENABLED
#endif
#elif defined(__GNUG__)
#if defined(__GXX_RTTI)
#define __RTTI_ENABLED
#endif
#else
#undef __RTTI_ENABLED
#endif

namespace
{
    // TODO: It might be desirable to have a similar method implemented for
    // RR types that do not rely on RTTI. In that case, RTTI could be
    // disabled for RR code.
    template <typename T>
    inline std::string typeName()
    {
#ifdef __RTTI_ENABLED
        return typeid(T).name();
#else
        return "unknown type (RTTI is disabled)";
#endif
    }
}
