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

#ifndef HIPCUB_TEST_TEST_UTILS_HALF_HPP_
#define HIPCUB_TEST_TEST_UTILS_HALF_HPP_

#include <type_traits>
#include <hipcub/util_type.hpp>

#include "half.hpp"

namespace test_utils
{

/// \brief Half-precision floating point type
using half = ::__half;
using native_half = half_t;

HIPCUB_HOST_DEVICE
inline test_utils::half native_to_half(const test_utils::native_half& x)
{
    return *reinterpret_cast<const test_utils::half *>(&x);
}

struct half_equal_to
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __heq(a, b);
        #else
        return test_utils::native_half(a) == test_utils::native_half(b);
        #endif
    }
};

struct half_not_equal_to
{
    HIPCUB_HOST_DEVICE inline
    bool operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hne(a, b);
        #else
        return test_utils::native_half(a) != test_utils::native_half(b);
        #endif
    }
};

struct half_plus
{
    HIPCUB_HOST_DEVICE inline
    test_utils::half operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hadd(a, b);
        #else
        return native_to_half(test_utils::native_half(a) + test_utils::native_half(b));
        #endif
    }
};

struct half_minus
{
    HIPCUB_HOST_DEVICE inline
    test_utils::half operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hsub(a, b);
        #else
        return native_to_half(test_utils::native_half(a) - test_utils::native_half(b));
        #endif
    }
};

struct half_multiplies
{
    HIPCUB_HOST_DEVICE inline
    test_utils::half operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hmul(a, b);
        #else
        return native_to_half(test_utils::native_half(a) * test_utils::native_half(b));
        #endif
    }
};

struct half_maximum
{
    HIPCUB_HOST_DEVICE inline
    test_utils::half operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hlt(a, b) ? b : a;
        #else
        return test_utils::native_half(a) < test_utils::native_half(b) ? b : a;
        #endif
    }
};

struct half_minimum
{
    HIPCUB_HOST_DEVICE inline
    test_utils::half operator()(const test_utils::half& a, const test_utils::half& b) const
    {
        #if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
        return __hlt(a, b) ? a : b;
        #else
        return test_utils::native_half(a) < test_utils::native_half(b) ? a : b;
        #endif
    }
};

}

inline std::ostream& operator<<(std::ostream& stream, const test_utils::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}
#endif // HIPCUB_TEST_HIPCUB_TEST_UTILS_HALF_HPP_
