#ifndef UNIQUE_PATH_FEATURES_H
#define UNIQUE_PATH_FEATURES_H

#include "unique_path_platform.hpp"

#ifdef MIOPEN_WINDOWS_API

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#if defined(MIOPEN_FILESYSTEM_HAS_BCRYPT) // defined on the command line by the project
#include <bcrypt.h>
#if defined(_MSC_VER)
#pragma comment(lib, "bcrypt.lib")
#endif // defined(_MSC_VER)
#else  // defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)
#include <wincrypt.h>
#if defined(_MSC_VER)
#pragma comment(lib, "advapi32.lib")
#endif // !defined(MIOPEN_FILESYSTEM_NO_DEPRECATED) && defined(_MSC_VER)
#endif // defined(MIOPEN_FILESYSTEM_HAS_BCRYPT)

// Note: Legacy MinGW doesn't have ntstatus.h and doesn't define NTSTATUS error codes other than
// STATUS_SUCCESS.
#if !defined(NT_SUCCESS)
#define NT_SUCCESS(Status) (((NTSTATUS_)(Status)) >= 0)
#endif
#if !defined(STATUS_SUCCESS)
#define STATUS_SUCCESS ((NTSTATUS)0x00000000l)
#endif
#if !defined(STATUS_NOT_IMPLEMENTED)
#define STATUS_NOT_IMPLEMENTED ((NTSTATUS)0xC0000002l)
#endif
#if !defined(STATUS_INVALID_INFO_CLASS)
#define STATUS_INVALID_INFO_CLASS ((NTSTATUS)0xC0000003l)
#endif
#if !defined(STATUS_INVALID_HANDLE)
#define STATUS_INVALID_HANDLE ((NTSTATUS)0xC0000008l)
#endif
#if !defined(STATUS_INVALID_PARAMETER)
#define STATUS_INVALID_PARAMETER ((NTSTATUS)0xC000000Dl)
#endif
#if !defined(STATUS_NO_SUCH_DEVICE)
#define STATUS_NO_SUCH_DEVICE ((NTSTATUS)0xC000000El)
#endif
#if !defined(STATUS_NO_SUCH_FILE)
#define STATUS_NO_SUCH_FILE ((NTSTATUS)0xC000000Fl)
#endif
#if !defined(STATUS_NO_MORE_FILES)
#define STATUS_NO_MORE_FILES ((NTSTATUS)0x80000006l)
#endif
#if !defined(STATUS_BUFFER_OVERFLOW)
#define STATUS_BUFFER_OVERFLOW ((NTSTATUS)0x80000005l)
#endif
#if !defined(STATUS_NO_MEMORY)
#define STATUS_NO_MEMORY ((NTSTATUS)0xC0000017l)
#endif
#if !defined(STATUS_ACCESS_DENIED)
#define STATUS_ACCESS_DENIED ((NTSTATUS)0xC0000022l)
#endif
#if !defined(STATUS_OBJECT_NAME_NOT_FOUND)
#define STATUS_OBJECT_NAME_NOT_FOUND ((NTSTATUS)0xC0000034l)
#endif
#if !defined(STATUS_OBJECT_PATH_NOT_FOUND)
#define STATUS_OBJECT_PATH_NOT_FOUND ((NTSTATUS)0xC000003Al)
#endif
#if !defined(STATUS_SHARING_VIOLATION)
#define STATUS_SHARING_VIOLATION ((NTSTATUS)0xC0000043l)
#endif
#if !defined(STATUS_EAS_NOT_SUPPORTED)
#define STATUS_EAS_NOT_SUPPORTED ((NTSTATUS)0xC000004Fl)
#endif
#if !defined(STATUS_NOT_SUPPORTED)
#define STATUS_NOT_SUPPORTED ((NTSTATUS)0xC00000BBl)
#endif
#if !defined(STATUS_BAD_NETWORK_PATH)
#define STATUS_BAD_NETWORK_PATH ((NTSTATUS)0xC00000BEl)
#endif
#if !defined(STATUS_DEVICE_DOES_NOT_EXIST)
#define STATUS_DEVICE_DOES_NOT_EXIST ((NTSTATUS)0xC00000C0l)
#endif
#if !defined(STATUS_BAD_NETWORK_NAME)
#define STATUS_BAD_NETWORK_NAME ((NTSTATUS)0xC00000CCl)
#endif
#if !defined(STATUS_DIRECTORY_NOT_EMPTY)
#define STATUS_DIRECTORY_NOT_EMPTY ((NTSTATUS)0xC0000101l)
#endif
#if !defined(STATUS_NOT_A_DIRECTORY)
#define STATUS_NOT_A_DIRECTORY ((NTSTATUS)0xC0000103l)
#endif
#if !defined(STATUS_NOT_FOUND)
#define STATUS_NOT_FOUND ((NTSTATUS)0xC0000225l)
#endif

#else // MIOPEN_WINDOWS_API

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/utsname.h>
#include <unistd.h>

// At least Mac OS X 10.6 and older doesn't support O_CLOEXEC
#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif // O_CLOEXEC

#if !defined(MIOPEN_FILESYSTEM_DISABLE_ARC4RANDOM)
#if MIOPEN_OS_BSD_OPEN >= MIOPEN_VERSION_NUMBER(2, 1, 0) || \
    MIOPEN_OS_BSD_FREE >= MIOPEN_VERSION_NUMBER(8, 0, 0) || MIOPEN_LIB_C_CLOUDABI
#include <stdlib.h>
#define MIOPEN_FILESYSTEM_HAS_ARC4RANDOM
#endif
#endif // !defined(MIOPEN_FILESYSTEM_DISABLE_ARC4RANDOM)

#if !defined(MIOPEN_FILESYSTEM_DISABLE_GETRANDOM)
#if(defined(__linux__) || defined(__linux) || defined(linux)) && \
    (!defined(__ANDROID__) || __ANDROID_API__ >= 28)
#include <sys/syscall.h>
#if defined(SYS_getrandom)
#define MIOPEN_FILESYSTEM_HAS_GETRANDOM_SYSCALL
#endif // defined(SYS_getrandom)
#if defined(__has_include)
#if __has_include(<sys/random.h>)
#define MIOPEN_FILESYSTEM_HAS_GETRANDOM
#endif
#elif defined(__GLIBC__)
#if __GLIBC_PREREQ(2, 25)
#define MIOPEN_FILESYSTEM_HAS_GETRANDOM
#endif
#endif // MIOPEN_FILESYSTEM_HAS_GETRANDOM definition
#if defined(MIOPEN_FILESYSTEM_HAS_GETRANDOM)
#include <sys/random.h>
#endif
#endif // (defined(__linux__) || defined(__linux) || defined(linux)) && (!defined(__ANDROID__) ||
       // __ANDROID_API__ >= 28)
#endif // !defined(MIOPEN_FILESYSTEM_DISABLE_GETRANDOM)

#define MIOPEN_ATTRIBUTE_UNUSED __attribute__((__unused__))

#if defined(__has_attribute)
#if __has_attribute(__used__)
#define MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN __attribute__((__used__))
#endif
#endif

#if !defined(MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN) && defined(__GNUC__) && \
    (__GNUC__ * 100 + __GNUC_MINOR__) >= 402
#define MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN __attribute__((__used__))
#endif

#if !defined(MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN)
#define MIOPEN_FILESYSTEM_NO_ATTRIBUTE_RETAIN
#define MIOPEN_FILESYSTEM_ATTRIBUTE_RETAIN
#endif

#endif // MIOPEN_WINDOWS_API

#endif // UNIQUE_PATH_FEATURES_H
