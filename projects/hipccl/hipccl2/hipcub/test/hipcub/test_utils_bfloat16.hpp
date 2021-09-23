// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_TEST_UTILS_BFLOAT16_HPP_
#define HIPCUB_TEST_TEST_UTILS_BFLOAT16_HPP_

#include <type_traits>

#include <hipcub/util_type.hpp>

#include "bfloat16.hpp"

namespace test_utils
{

/// \brief Bfloat16-precision floating point type
#ifdef __HIP_PLATFORM_HCC__
using bfloat16 = ::hip_bfloat16;
#elif defined(__HIP_PLATFORM_NVIDIA__)
using bfloat16 = ::__nv_bfloat16;
#endif

#ifdef __HIP_CPU_RT__
using native_bfloat16 = bfloat16_t;
#else
using native_bfloat16 = bfloat16_t;
#endif
// hipCUB

// Support bfloat16 operators on host side
HIPCUB_HOST inline
test_utils::native_bfloat16 bfloat16_to_native(const test_utils::bfloat16& x)
{
    return *reinterpret_cast<const test_utils::native_bfloat16 *>(&x);
}

HIPCUB_HOST inline
test_utils::bfloat16 native_to_bfloat16(const test_utils::native_bfloat16& x)
{
    return *reinterpret_cast<const test_utils::bfloat16 *>(&x);
}

struct bfloat16_less
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a < b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_less_equal
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a <= b;
        #else
        return bfloat16_to_native(a) <= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a > b;
        #else
        return bfloat16_to_native(a) > bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater_equal
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a >= b;
        #else
        return bfloat16_to_native(a) >= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_equal_to
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a == b;
        #else
        return bfloat16_to_native(a) == bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_not_equal_to
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a != b;
        #else
        return bfloat16_to_native(a) != bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_plus
{
    HIPCUB_HOST_DEVICE inline
    test_utils::bfloat16 operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a + b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) + bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_minus
{
    HIPCUB_HOST_DEVICE inline
    test_utils::bfloat16 operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a - b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) - bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_multiplies
{
    HIPCUB_HOST_DEVICE inline
    test_utils::bfloat16 operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a * b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) * bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_maximum
{
    HIPCUB_HOST_DEVICE inline
    test_utils::bfloat16 operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a < b ? b : a;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? b : a;
        #endif
    }
};

struct bfloat16_minimum
{
    HIPCUB_HOST_DEVICE inline
    test_utils::bfloat16 operator()(const test_utils::bfloat16& a, const test_utils::bfloat16& b) const
    {
        #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
        return a < b ? a : b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? a : b;
        #endif
    }
};

}

#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_BFLOAT16_HPP_
