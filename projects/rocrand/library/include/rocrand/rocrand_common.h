// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_COMMON_H_
#define ROCRAND_COMMON_H_

#define ROCRAND_2POW16_INV (1.5258789e-05f)
#define ROCRAND_2POW16_INV_2PI (9.58738e-05f)
#define ROCRAND_2POW32_INV (2.3283064e-10f)
#define ROCRAND_2POW32_INV_DOUBLE (2.3283064365386963e-10)
#define ROCRAND_2POW64_INV (5.4210109e-20f)
#define ROCRAND_2POW64_INV_DOUBLE (5.4210108624275221700372640043497e-20)
#define ROCRAND_2POW32_INV_2PI (1.46291807e-09f)
#define ROCRAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define ROCRAND_PI (3.141592653f)
#define ROCRAND_PI_DOUBLE  (3.1415926535897932)
#define ROCRAND_2PI (6.2831855f)
#define ROCRAND_SQRT2 (1.4142135f)
#define ROCRAND_SQRT2_DOUBLE (1.4142135623730951)

#include <hip/hip_runtime.h>
#include <utility>

#define ROCRAND_KERNEL __global__ static

#if __HIP_DEVICE_COMPILE__            \
    && (defined(__HIP_PLATFORM_AMD__) \
        || (defined(__HIP_PLATFORM_NVCC__) && (__CUDA_ARCH__ >= 530)))
    #define ROCRAND_HALF_MATH_SUPPORTED
#endif

//  Copyright 2001 John Maddock.
//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_STRINGIZE(X)
#define ROCRAND_STRINGIZE(X) ROCRAND_DO_STRINGIZE(X)
#define ROCRAND_DO_STRINGIZE(X) #X

//  Copyright 2017 Peter Dimov.
//
//  Distributed under the Boost Software License, Version 1.0.
//
//  See http://www.boost.org/LICENSE_1_0.txt
//
//  BOOST_PRAGMA_MESSAGE("message")
//
//  Expands to the equivalent of #pragma message("message")
#if defined(__INTEL_COMPILER)
    #define ROCRAND_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCRAND_STRINGIZE(__LINE__) "): note: " x))
#elif defined(__GNUC__)
    #define ROCRAND_PRAGMA_MESSAGE(x) _Pragma(ROCRAND_STRINGIZE(message(x)))
#elif defined(_MSC_VER)
    #define ROCRAND_PRAGMA_MESSAGE(x) \
        __pragma(message(__FILE__ "(" ROCRAND_STRINGIZE(__LINE__) "): note: " x))
#else
    #define ROCRAND_PRAGMA_MESSAGE(x)
#endif

#if __cplusplus >= 201402L
    #define ROCRAND_DEPRECATED(msg) [[deprecated(msg)]]
#elif defined(_MSC_VER) && !defined(__clang__)
    #define ROCRAND_DEPRECATED(msg) __declspec(deprecated(msg))
#elif defined(__clang__) || defined(__GNUC__)
    #define ROCRAND_DEPRECATED(msg) __attribute__((deprecated(msg)))
#else
    #define ROCRAND_DEPRECATED(msg)
#endif

// This is an accessor macro for HIP vector types (eg. int2, float4, etc.).
// Prior to HIP 7.0, individual elements could be accessed through the
// data member:
//
// int2 vec;
// vec.data[0] = 1;
//
// Beginning with HIP 7.0, the data member is hidden, and individual
// elements must be accessed like this:
//
// int2 vec;
// vec[0] = 1;
//
// You can use the macro like this:
//
// int2 vec;
// ROCRAND_HIPVEC_ACCESS(vec)[0];
//
#if defined(__HIP_PLATFORM_AMD__)
    #if HIP_VERSION_MAJOR < 7
        #define ROCRAND_HIPVEC_ACCESS(x) x.data
    #else
        #define ROCRAND_HIPVEC_ACCESS(x) x
    #endif
#endif

namespace rocrand_device {
namespace detail {

__forceinline__ __device__ __host__
unsigned long long
    mad_u64_u32(const unsigned int x, const unsigned int y, const unsigned long long z)
{
    return static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y) + z;
}

__forceinline__ __device__ __host__
unsigned long long mul_u64_u32(const unsigned int x, const unsigned int y)
{
    return static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y);
}

// This helps access fields of engine's internal state which
// saves floats and doubles generated using the Box–Muller transform
template<typename Engine>
struct engine_boxmuller_helper
{
    static __forceinline__ __device__ __host__ bool has_float(const Engine* engine)
    {
        return engine->m_state.boxmuller_float_state != 0;
    }

    static __forceinline__ __device__ __host__ float get_float(Engine* engine)
    {
        engine->m_state.boxmuller_float_state = 0;
        return engine->m_state.boxmuller_float;
    }

    static __forceinline__ __device__ __host__ void save_float(Engine* engine, float f)
    {
        engine->m_state.boxmuller_float_state = 1;
        engine->m_state.boxmuller_float = f;
    }

    static __forceinline__ __device__ __host__ bool has_double(const Engine* engine)
    {
        return engine->m_state.boxmuller_double_state != 0;
    }

    static __forceinline__ __device__ __host__ float get_double(Engine* engine)
    {
        engine->m_state.boxmuller_double_state = 0;
        return engine->m_state.boxmuller_double;
    }

    static __forceinline__ __device__ __host__ void save_double(Engine* engine, double d)
    {
        engine->m_state.boxmuller_double_state = 1;
        engine->m_state.boxmuller_double = d;
    }
};

template<typename T>
__forceinline__ __device__ __host__ void split_ull(T& lo, T& hi, unsigned long long int val);

template<>
__forceinline__ __device__ __host__ void
    split_ull(unsigned int& lo, unsigned int& hi, unsigned long long int val)
{
    lo = val & 0xFFFFFFFF;
    hi = (val >> 32) & 0xFFFFFFFF;
}

template<>
__forceinline__ __device__ __host__ void
    split_ull(unsigned long long int& lo, unsigned long long int& hi, unsigned long long int val)
{
    lo = val;
    hi = 0;
}

} // end namespace detail
} // end namespace rocrand_device

#endif // ROCRAND_COMMON_H_
