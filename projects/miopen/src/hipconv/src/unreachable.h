#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define HIPCONV_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define HIPCONV_UNREACHABLE() __assume(0)
#else
#include <cstdlib>
#define HIPCONV_UNREACHABLE() std::abort()
#endif
