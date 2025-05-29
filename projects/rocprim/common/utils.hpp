// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef COMMON_UTILS_HPP_
#define COMMON_UTILS_HPP_

#include <rocprim/intrinsics/thread.hpp>

#ifdef USE_GTEST
    // GoogleTest-compatible HIP_CHECK macro. FAIL is called to log the Google Test trace.
    // The lambda is invoked immediately as assertions that generate a fatal failure can
    // only be used in void-returning functions.
    #define HIP_CHECK(condition)                                                            \
        {                                                                                   \
            hipError_t error = condition;                                                   \
            if(error != hipSuccess)                                                         \
            {                                                                               \
                [error]()                                                                   \
                { FAIL() << "HIP error " << error << ": " << hipGetErrorString(error); }(); \
                exit(error);                                                                \
            }                                                                               \
        }
#else
    #define HIP_CHECK(condition)                                                                \
        {                                                                                       \
            hipError_t error = condition;                                                       \
            if(error != hipSuccess)                                                             \
            {                                                                                   \
                std::cout << "HIP error: " << hipGetErrorString(error) << " file: " << __FILE__ \
                          << " line: " << __LINE__ << std::endl;                                \
                exit(error);                                                                    \
            }                                                                                   \
        }
#endif

namespace common
{
template<unsigned int LogicalWarpSize>
__device__ constexpr bool device_test_enabled_for_warp_size_v
    = ::rocprim::arch::wavefront::max_size() >= LogicalWarpSize;

inline char* __get_env(const char* name)
{
    char* env;
#ifdef _MSC_VER
    errno_t err = _dupenv_s(&env, nullptr, name);
    if(err)
    {
        return nullptr;
    }
#else
    env = std::getenv(name);
#endif
    return env;
}

inline void clean_env(char* env)
{
#ifdef _MSC_VER
    free(env);
#endif
    (void)env;
}

inline bool use_hmm()
{

    char*      env = __get_env("ROCPRIM_USE_HMM");
    const bool hmm = (env != nullptr) && (strcmp(env, "1") == 0);
    clean_env(env);
    return hmm;
}

// Helper for HMM allocations: HMM is requested through ROCPRIM_USE_HMM=1 environment variable
template<class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if(use_hmm())
    {
        return hipMallocManaged(reinterpret_cast<void**>(devPtr), size);
    }
    else
    {
        return hipMalloc(reinterpret_cast<void**>(devPtr), size);
    }
    return hipSuccess;
}

} // namespace common

#endif // COMMON_UTILS_HPP_
