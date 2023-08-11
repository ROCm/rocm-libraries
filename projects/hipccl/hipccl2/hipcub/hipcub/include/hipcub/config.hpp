/******************************************************************************
 * Copyright (c) 2010-2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 * Modifications Copyright (c) 2019-2023, Advanced Micro Devices, Inc.  All rights reserved.
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

#define HIPCUB_NAMESPACE hipcub

#define BEGIN_HIPCUB_NAMESPACE \
    namespace hipcub {

#define END_HIPCUB_NAMESPACE \
    } /* hipcub */

#ifdef __HIP_PLATFORM_AMD__
    #define HIPCUB_ROCPRIM_API 1
    #define HIPCUB_RUNTIME_FUNCTION __host__

    #include <rocprim/intrinsics/thread.hpp>
    #define HIPCUB_WARP_THREADS ::rocprim::warp_size()
    #define HIPCUB_DEVICE_WARP_THREADS ::rocprim::device_warp_size()
    #define HIPCUB_HOST_WARP_THREADS ::rocprim::host_warp_size()
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

BEGIN_HIPCUB_NAMESPACE

/// hipCUB error reporting macro (prints error messages to stderr)
#if (defined(DEBUG) || defined(_DEBUG)) && !defined(HIPCUB_STDERR)
    #define HIPCUB_STDERR
#endif

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

#ifndef HipcubDebug
    #define HipcubDebug(e) hipcub::Debug((hipError_t) (e), __FILE__, __LINE__)
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

END_HIPCUB_NAMESPACE

#endif // HIPCUB_CONFIG_HPP_
