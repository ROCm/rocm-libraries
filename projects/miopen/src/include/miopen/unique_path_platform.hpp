// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef UNIQUE_PATH_PLATFORM_HPP
#define UNIQUE_PATH_PLATFORM_HPP

#if defined(_WIN32) || defined(__CYGWIN__) // Windows default, including MinGW and Cygwin
#define MIOPEN_WINDOWS_API
#else
#define MIOPEN_POSIX_API

#define MIOPEN_VERSION_NUMBER(major, minor, patch) \
    ((((major) % 100) * 10000000) + (((minor) % 100) * 100000) + ((patch) % 100000))

#define MIOPEN_VERSION_NUMBER_ZERO MIOPEN_VERSION_NUMBER(0, 0, 0)

#define MIOPEN_VERSION_NUMBER_MIN MIOPEN_VERSION_NUMBER(0, 0, 1)

#define MIOPEN_VERSION_NUMBER_AVAILABLE MIOPEN_VERSION_NUMBER_MIN

#define MIOPEN_VERSION_NUMBER_NOT_AVAILABLE MIOPEN_VERSION_NUMBER_ZERO

#define MIOPEN_PREDEF_MAKE_10_VRPPPP(V) \
    MIOPEN_VERSION_NUMBER(((V) / 100000) % 10, ((V) / 10000) % 10, (V) % 10000)

#define MIOPEN_PREDEF_MAKE_10_VVRRPPP(V) \
    MIOPEN_VERSION_NUMBER(((V) / 100000) % 100, ((V) / 1000) % 100, (V) % 1000)

#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER_NOT_AVAILABLE
#define MIOPEN_OS_BSD_FREE MIOPEN_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__OpenBSD__)
#undef MIOPEN_OS_BSD_OPEN
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_0)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 0, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_1)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 1, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_2)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 2, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_3)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 3, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_4)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 4, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_5)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 5, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_6)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 6, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_7)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 7, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_8)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 8, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD2_9)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(2, 9, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_0)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 0, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_1)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 1, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_2)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 2, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_3)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 3, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_4)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 4, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_5)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 5, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_6)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 6, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_7)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 7, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_8)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 8, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD3_9)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(3, 9, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_0)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 0, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_1)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 1, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_2)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 2, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_3)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 3, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_4)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 4, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_5)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 5, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_6)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 6, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_7)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 7, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_8)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 8, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD4_9)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(4, 9, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_0)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 0, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_1)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 1, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_2)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 2, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_3)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 3, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_4)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 4, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_5)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 5, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_6)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 6, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_7)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 7, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_8)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 8, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD5_9)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(5, 9, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_0)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 0, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_1)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 1, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_2)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 2, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_3)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 3, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_4)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 4, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_5)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 5, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_6)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 6, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_7)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 7, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_8)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 8, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN) && defined(OpenBSD6_9)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER(6, 9, 0)
#endif
#if !defined(MIOPEN_OS_BSD_OPEN)
#define MIOPEN_OS_BSD_OPEN MIOPEN_VERSION_NUMBER_AVAILABLE
#endif

#elif defined(__FreeBSD__)

#undef MIOPEN_OS_BSD_FREE
#include <sys/param.h>
#if defined(__FreeBSD_version)
#if __FreeBSD_version == 491000
#define MIOPEN_OS_BSD_FREE MIOPEN_VERSION_NUMBER(4, 10, 0)
#elif __FreeBSD_version == 492000
#define MIOPEN_OS_BSD_FREE MIOPEN_VERSION_NUMBER(4, 11, 0)
#elif __FreeBSD_version < 500000
#define MIOPEN_OS_BSD_FREE MIOPEN_PREDEF_MAKE_10_VRPPPP(__FreeBSD_version)
#else
#define MIOPEN_OS_BSD_FREE MIOPEN_PREDEF_MAKE_10_VVRRPPP(__FreeBSD_version)
#endif
#else
#define MIOPEN_OS_BSD_FREE MIOPEN_VERSION_NUMBER_AVAILABLE
#endif

#endif

#define MIOPEN_LIB_C_CLOUDABI MIOPEN_VERSION_NUMBER_NOT_AVAILABLE

#if defined(__cloudlibc__)
#undef MIOPEN_LIB_C_CLOUDABI
#define MIOPEN_LIB_C_CLOUDABI MIOPEN_VERSION_NUMBER(__cloudlibc_major__, __cloudlibc_minor__, 0)
#endif

#endif // defined(_WIN32) || defined(__CYGWIN__)

#endif // UNIQUE_PATH_PLATFORM_HPP
