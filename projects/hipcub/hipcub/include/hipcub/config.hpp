/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2025, Advanced Micro Devices, Inc.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#ifndef HIPCUB_CONFIG_HPP_
#define HIPCUB_CONFIG_HPP_

#include <hip/hip_runtime.h>

// Version
#include "hipcub_version.hpp" // IWYU pragma: export

#define HIPCUB_NAMESPACE hipcub

// Inline namespace (e.g. HIPCUB_300400_NS where 300400 is the hipCUB version) is used to
// eliminate issues when shared libraries are built with different versions of hipCUB so they may
// have symbols with the same name but different content.
// HIPCUB_DISABLE_INLINE_NAMESPACE can be defined to disable inline namespaces (the old behavior).
// HIPCUB_INLINE_NAMESPACE can be defined to override the standard inline namespace name.
// Additionally, all kernels have hidden visibility.
// For rocPRIM backend, see rocprim/config.hpp for similar definitions.
#if defined(DOXYGEN_SHOULD_SKIP_THIS) || defined(HIPCUB_DISABLE_INLINE_NAMESPACE)
    #define HIPCUB_INLINE_NAMESPACE
    #define BEGIN_HIPCUB_INLINE_NAMESPACE
    #define END_HIPCUB_INLINE_NAMESPACE
#else
    #define HIPCUB_CONCAT_(SEP, A, B) A##SEP##B
    #define HIPCUB_CONCAT(SEP, A, B) HIPCUB_CONCAT_(SEP, A, B)

    #ifndef HIPCUB_INLINE_NAMESPACE
        #define HIPCUB_INLINE_NAMESPACE \
            HIPCUB_CONCAT(_, HIPCUB, HIPCUB_CONCAT(_, HIPCUB_VERSION, NS))
    #endif
    #define BEGIN_HIPCUB_INLINE_NAMESPACE        \
        inline namespace HIPCUB_INLINE_NAMESPACE \
        {
    #define END_HIPCUB_INLINE_NAMESPACE } /* inline namespace */
#endif

#define BEGIN_HIPCUB_NAMESPACE \
    namespace HIPCUB_NAMESPACE \
    {                          \
    BEGIN_HIPCUB_INLINE_NAMESPACE

#define END_HIPCUB_NAMESPACE    \
    END_HIPCUB_INLINE_NAMESPACE \
    } /* namespace hipcub */

#ifdef __HIP_PLATFORM_AMD__
    #define HIPCUB_ROCPRIM_API 1
    #define HIPCUB_RUNTIME_FUNCTION __host__

    #include <rocprim/device/config_types.hpp>
    #include <rocprim/intrinsics/arch.hpp>
    #include <rocprim/intrinsics/thread.hpp>

BEGIN_HIPCUB_NAMESPACE
namespace detail
{
inline unsigned int host_warp_size_wrapper()
{
    int          device_id      = 0;
    unsigned int host_warp_size = 0;
    hipError_t   error          = hipGetDevice(&device_id);
    if(error != hipSuccess)
    {
        fprintf(stderr, "HIP error: %d line: %d: %s\n", error, __LINE__, hipGetErrorString(error));
        fflush(stderr);
    }
    if(::rocprim::host_warp_size(device_id, host_warp_size) != hipSuccess)
    {
        return 0u;
    }
    return host_warp_size;
}
} // namespace detail
END_HIPCUB_NAMESPACE
    #include <rocprim/intrinsics/arch.hpp>

    #define HIPCUB_WARP_THREADS ::rocprim::warp_size()
    // HIPCUB (and CUB) don't have a method to express min and max warp size.
    #define HIPCUB_DEVICE_WARP_THREADS ::rocprim::arch::wavefront::max_size()
    #define HIPCUB_HOST_WARP_THREADS ::hipcub::detail::host_warp_size_wrapper()
    #define HIPCUB_ARCH 1 // ignored with rocPRIM backend
#elif defined(__HIP_PLATFORM_NVIDIA__)
    #define HIPCUB_CUB_API 1
    #define HIPCUB_RUNTIME_FUNCTION CUB_RUNTIME_FUNCTION

    #include <cub/util_arch.cuh>
    #define HIPCUB_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_DEVICE_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_HOST_WARP_THREADS CUB_PTX_WARP_THREADS
    #define HIPCUB_ARCH CUB_PTX_ARCH
    BEGIN_HIPCUB_NAMESPACE
    using namespace cub;
    END_HIPCUB_NAMESPACE
#endif

/// Supported warp sizes
#define HIPCUB_WARP_SIZE_32 32u
#define HIPCUB_WARP_SIZE_64 64u
#define HIPCUB_MAX_WARP_SIZE HIPCUB_WARP_SIZE_64

#define HIPCUB_HOST __host__
#define HIPCUB_DEVICE __device__
#define HIPCUB_HOST_DEVICE __host__ __device__
#define HIPCUB_FORCEINLINE __forceinline__
#define HIPCUB_SHARED_MEMORY __shared__

// Helper macros to disable warnings in clang
#ifdef __clang__
#define HIPCUB_PRAGMA_TO_STR(x) _Pragma(#x)
#define HIPCUB_CLANG_SUPPRESS_WARNING_PUSH _Pragma("clang diagnostic push")
#define HIPCUB_CLANG_SUPPRESS_WARNING(w) HIPCUB_PRAGMA_TO_STR(clang diagnostic ignored w)
#define HIPCUB_CLANG_SUPPRESS_WARNING_POP _Pragma("clang diagnostic pop")
#define HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH(w) \
    HIPCUB_CLANG_SUPPRESS_WARNING_PUSH HIPCUB_CLANG_SUPPRESS_WARNING(w)
#else // __clang__
#define HIPCUB_CLANG_SUPPRESS_WARNING_PUSH
#define HIPCUB_CLANG_SUPPRESS_WARNING(w)
#define HIPCUB_CLANG_SUPPRESS_WARNING_POP
#define HIPCUB_CLANG_SUPPRESS_WARNING_WITH_PUSH(w)
#endif // __clang__

#define HIPCUB_CLANG_SUPPRESS_DEPRECATED_PUSH     \
    HIPCUB_CLANG_SUPPRESS_WARNING_PUSH            \
    HIPCUB_CLANG_SUPPRESS_WARNING("-Wdeprecated") \
    HIPCUB_CLANG_SUPPRESS_WARNING("-Wdeprecated-declarations")
#define HIPCUB_CLANG_SUPPRESS_DEPRECATED_POP HIPCUB_CLANG_SUPPRESS_WARNING_POP

/// hipCUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG)) && !defined(HIPCUB_STDERR)
    #define HIPCUB_STDERR
#endif

BEGIN_HIPCUB_NAMESPACE

/// \brief Don't use this function directly, but via the `HipcubDebug` macro instead.
/// If `error` is not `hipSuccess`, prints an error message containing the source filename and
/// line information to the standard error output.
/// \note This only happens if `HIPCUB_STDERR` is defined.
inline
hipError_t Debug(
    hipError_t      error,
    const char*     filename,
    int             line)
{
    (void)filename;
    (void)line;
#ifdef HIPCUB_STDERR
    if (error)
    {
        fprintf(stderr, "HIP error %d [%s, %d]: %s\n", error, filename, line, hipGetErrorString(error));
        fflush(stderr);
    }
#endif
    return error;
}

/// \brief Don't use this function directly, but via the `HipcubLog` macro instead.
/// Prints the provided message containing the source filename and
/// line information to the standard output.
inline void Log(const char* message, const char* filename, int line)
{
    printf("hipcub: %s [%s:%d]\n", message, filename, line);
}

END_HIPCUB_NAMESPACE

#ifndef HipcubDebug
    #define HipcubDebug(e) ::hipcub::Debug((hipError_t)(e), __FILE__, __LINE__)
#endif

#ifndef HipcubLog
    #define HipcubLog(msg) ::hipcub::Log(msg, __FILE__, __LINE__)
#endif

#if __cpp_if_constexpr
    #define HIPCUB_IF_CONSTEXPR constexpr
#else
    #if defined(_MSC_VER) && !defined(__clang__)
        // MSVC (and not Clang pretending to be MSVC) unconditionally exposes if constexpr (even in C++14 mode),
        // moreover it triggers warning C4127 (conditional expression is constant) when not using it. nvcc will
        // be calling cl.exe for host-side codegen.
        #define HIPCUB_IF_CONSTEXPR constexpr
    #else
        #define HIPCUB_IF_CONSTEXPR
    #endif
#endif

#ifdef DOXYGEN_SHOULD_SKIP_THIS // Documentation only

    /// \def HIPCUB_DEBUG_SYNC
    /// \brief If defined, synchronizes the stream after every kernel launch and prints the launch information
    /// to the standard output. If any of `CUB_DEBUG_SYNC`, `CUB_DEBUG_HOST_ASSERTIONS`, `CUB_DEBUG_DEVICE_ASSERTIONS`
    /// or `CUB_DEBUG_ALL` is defined, this is also defined automatically.
    #define HIPCUB_DEBUG_SYNC

#endif // DOXYGEN_SHOULD_SKIP_THIS

#if defined(HIPCUB_CUB_API) && defined(HIPCUB_DEBUG_SYNC) && !defined(CUB_DEBUG_SYNC)
    #define CUB_DEBUG_SYNC
#endif

#if !defined(HIPCUB_DEBUG_SYNC)                                       \
    && (defined(CUB_DEBUG_SYNC) || defined(CUB_DEBUG_HOST_ASSERTIONS) \
        || defined(CUB_DEBUG_DEVICE_ASSERTIONS) || defined(CUB_DEBUG_ALL))
    #define HIPCUB_DEBUG_SYNC
#endif

#ifdef HIPCUB_ROCPRIM_API
    // TODO C++17: use an inline constexpr variable
    #ifdef HIPCUB_DEBUG_SYNC
        #define HIPCUB_DETAIL_DEBUG_SYNC_VALUE true
    #else
        #define HIPCUB_DETAIL_DEBUG_SYNC_VALUE false
    #endif
#endif // HIPCUB_ROCPRIM_API

#endif // HIPCUB_CONFIG_HPP_
