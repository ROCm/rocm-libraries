// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/// \file unreachable.hpp
/// Provides a platform-agnostic unreachable macro.

#ifndef ROCRAND_RNG_UTILS_UNREACHABLE_HPP_
#define ROCRAND_RNG_UTILS_UNREACHABLE_HPP_

#include <cstdlib>
#include <iostream>

#ifndef __has_builtin
    #define __has_builtin(x) 0
#endif

#if __has_builtin(__builtin_unreachable) || defined(__GNUC__)
    #define ROCRAND_BUILTIN_UNREACHABLE __builtin_unreachable()
#elif defined(_MSC_VER)
    #define ROCRAND_BUILTIN_UNREACHABLE __assume(false)
#endif

#if defined(__CUDA_ARCH__)
    #define ROCRAND_BUILTIN_TRAP __trap()
#elif __has_builtin(__builtin_trap) || defined(__GNUC__)
    #define ROCRAND_BUILTIN_TRAP __builtin_trap()
#else
    #define ROCRAND_BUILTIN_TRAP std::abort();
#endif

namespace rocrand_impl
{

[[noreturn]]
#if defined(__CUDACC__) || defined(__HIP__)
__host__ __device__
#endif
    inline static void
    unreachable_internal(const char* msg, const char* file, unsigned line)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    (void)msg;
    (void)file;
    (void)line;
    ROCRAND_BUILTIN_TRAP;
#else
    if(msg)
        std::cerr << msg << '\n';
    std::cerr << "UNREACHABLE executed";
    if(file)
        std::cerr << " at " << file << ":" << line;
    std::abort();
#endif
}

} // namespace rocrand_impl

#if !defined(NDEBUG)
    #define ROCRAND_UNREACHABLE(msg) ::rocrand_impl::unreachable_internal(msg, __FILE__, __LINE__)
#elif !defined(ROCRAND_BUILTIN_UNREACHABLE)
    #define ROCRAND_UNREACHABLE(msg) ::rocrand_impl::unreachable_internal(msg, __FILE__, __LINE__)
#else
    #define ROCRAND_UNREACHABLE(msg)     \
        do                               \
        {                                \
            ROCRAND_BUILTIN_TRAP;        \
            ROCRAND_BUILTIN_UNREACHABLE; \
        }                                \
        while(false)
#endif

#endif // ROCRAND_RNG_UTILS_UNREACHABLE_HPP_
