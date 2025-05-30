// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_TEST_UTILS_FUNCTIONAL_HPP_
#define HIPCUB_TEST_TEST_UTILS_FUNCTIONAL_HPP_

#ifdef __HIP_PLATFORM_AMD__
    #include <rocprim/type_traits.hpp>
#endif

#include "test_utils_bfloat16.hpp"
#include "test_utils_half.hpp"

namespace test_utils
{
struct less
{
    template<typename T>
    HIPCUB_HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const
    {
        return a < b;
    }
};

struct less_equal
{
    template<typename T>
    HIPCUB_HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

struct greater
{
    template<typename T>
    HIPCUB_HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

struct greater_equal
{
    template<typename T>
    HIPCUB_HOST_DEVICE constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

struct plus
{
    template<class T>
    HIPCUB_HOST_DEVICE inline constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

struct minus
{
    template<class T>
    HIPCUB_HOST_DEVICE inline constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

struct multiplies
{
    template<class T>
    HIPCUB_HOST_DEVICE inline constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

// HALF
template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::less::operator()<test_utils::half>(const test_utils::half& a,
                                                   const test_utils::half& b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hlt(a, b);
#else
    return test_utils::native_half(a) < test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::less_equal::operator()<test_utils::half>(const test_utils::half& a,
                                                         const test_utils::half& b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hle(a, b);
#else
    return test_utils::native_half(a) <= test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::greater::operator()<test_utils::half>(const test_utils::half& a,
                                                      const test_utils::half& b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hgt(a, b);
#else
    return test_utils::native_half(a) > test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::greater_equal::operator()<test_utils::half>(const test_utils::half& a,
                                                            const test_utils::half& b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hge(a, b);
#else
    return test_utils::native_half(a) >= test_utils::native_half(b);
#endif
}
// END HALF

// BFLOAT16

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::less::operator()<test_utils::bfloat16>(const test_utils::bfloat16& a,
                                                       const test_utils::bfloat16& b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a < b;
#else
    return test_utils::native_bfloat16(a) < test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::less_equal::operator()<test_utils::bfloat16>(const test_utils::bfloat16& a,
                                                             const test_utils::bfloat16& b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a <= b;
#else
    return test_utils::native_bfloat16(a) <= test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::greater::operator()<test_utils::bfloat16>(const test_utils::bfloat16& a,
                                                          const test_utils::bfloat16& b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a > b;
#else
    return test_utils::native_bfloat16(a) > test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool
    test_utils::greater_equal::operator()<test_utils::bfloat16>(const test_utils::bfloat16& a,
                                                                const test_utils::bfloat16& b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a >= b;
#else
    return test_utils::native_bfloat16(a) >= test_utils::native_bfloat16(b);
#endif
}
// END BFLOAT16

} // namespace test_utils
#endif // HIPCUB_TEST_TEST_UTILS_FUNCTIONAL_HPP_
