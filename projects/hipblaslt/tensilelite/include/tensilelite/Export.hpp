// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCM_TENSILELITE_EXPORT_HPP
#define ROCM_TENSILELITE_EXPORT_HPP

// Visibility annotations; use TENSILELITE_EXPORTED for exported API.
//
// On Windows, __declspec(dllexport) and __declspec(dllimport) are only
// generated if TENSILELITE_ENABLE_WINDOWS_DLL_DECLSPEC is defined, which must
// only happen when building or consuming a DLL.

#if (defined(_WIN32) || defined(__CYGWIN__)) &&                                \
    !defined(TENSILELITE_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define TENSILELITE_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if defined(TENSILELITE_BUILDING_LIBRARY)
#define TENSILELITE_EXPORTED __declspec(dllexport)
#else
#define TENSILELITE_EXPORTED __declspec(dllimport)
#endif
#else
// Non-Windows: use visibility attributes.
#define TENSILELITE_EXPORTED __attribute__((visibility("default")))
#endif

#endif // ROCM_TENSILELITE_EXPORT_HPP
