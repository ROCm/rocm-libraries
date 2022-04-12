// MIT License
//
// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HIPCUB_TEST_TEST_UTILS_SORT_COMPARATOR_HPP_
#define HIPCUB_TEST_TEST_UTILS_SORT_COMPARATOR_HPP_

#ifdef __HIP_PLATFORM_AMD__
#include <rocprim/type_traits.hpp>
#endif

#include "test_utils_half.hpp"
#include "test_utils_bfloat16.hpp"

namespace test_utils
{

template<class T>
constexpr auto is_floating_nan_host(const T& a)
    -> typename std::enable_if<std::is_floating_point<T>::value, bool>::type
{
    return (a != a);
}

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit, bool ShiftLess = (StartBit == 0 && EndBit == sizeof(Key) * 8), class Enable = void>
struct key_comparator {};

template <class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, false, typename std::enable_if<std::is_integral<Key>::value>::type>
{
    static constexpr Key radix_mask_upper  = (Key(1) << EndBit) - 1;
    static constexpr Key radix_mask_bottom = (Key(1) << StartBit) - 1;
    static constexpr Key radix_mask = radix_mask_upper ^ radix_mask_bottom;

    bool operator()(const Key& lhs, const Key& rhs)
    {
        Key l = lhs & radix_mask;
        Key r = rhs & radix_mask;
        return Descending ? (r < l) : (l < r);
    }
};

template <class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, false, typename std::enable_if<std::is_floating_point<Key>::value>::type>
{
    // Floating-point types do not support StartBit and EndBit.
    bool operator()(const Key&, const Key&)
    {
        return false;
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true, typename std::enable_if<std::is_integral<Key>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        return Descending ? (rhs < lhs) : (lhs < rhs);
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true, typename std::enable_if<std::is_floating_point<Key>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        if(is_floating_nan_host(lhs) && is_floating_nan_host(rhs) && std::signbit(lhs) == std::signbit(rhs)){
            return false;
        }
        if(Descending){
            if(is_floating_nan_host(lhs)) return !std::signbit(lhs);
            if(is_floating_nan_host(rhs)) return std::signbit(rhs);
            return (rhs < lhs);
        }else{
            if(is_floating_nan_host(lhs)) return std::signbit(lhs);
            if(is_floating_nan_host(rhs)) return !std::signbit(rhs);
            return (lhs < rhs);
        }
    }
};

// TODO: Merge this half specialization with bfloat16 below, once NVidia fixed its __half2float conversion for -NaN
template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true,
                      typename std::enable_if<std::is_same<Key, test_utils::half>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        test_utils::native_half lhs_native(lhs);
        test_utils::native_half rhs_native(rhs);
        return key_comparator<float, Descending, 0, sizeof(float) * 8>()(lhs_native, rhs_native);
    }
};

template<class Key, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_comparator<Key, Descending, StartBit, EndBit, true,
                      typename std::enable_if<std::is_same<Key, test_utils::bfloat16>::value>::type>
{
    bool operator()(const Key& lhs, const Key& rhs)
    {
        // HIP's half and bfloat16 doesn't have __host__ comparison operators, use floats instead
        return key_comparator<float, Descending, 0, sizeof(float) * 8>()(lhs, rhs);
    }
};

template<class Key, class Value, bool Descending, unsigned int StartBit, unsigned int EndBit>
struct key_value_comparator
{
    bool operator()(const std::pair<Key, Value>& lhs, const std::pair<Key, Value>& rhs)
    {
        return key_comparator<Key, Descending, StartBit, EndBit>()(lhs.first, rhs.first);
    }
};

struct less
{
    template<typename T>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(const T & a, const T & b) const
    {
        return a < b;
    }
};

struct less_equal
{
    template<typename T>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(const T & a, const T & b) const
    {
        return a <= b;
    }
};

struct greater
{
    template<typename T>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(const T & a, const T & b) const
    {
        return a > b;
    }
};

struct greater_equal
{
    template<typename T>
    HIPCUB_HOST_DEVICE inline constexpr bool operator()(const T & a, const T & b) const
    {
        return a >= b;
    }
};

// HALF
template<>
HIPCUB_HOST_DEVICE inline bool test_utils::less::operator()<test_utils::half>(
    const test_utils::half & a,
    const test_utils::half & b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hlt(a, b);
#else
    return test_utils::native_half(a) < test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::less_equal::operator()<test_utils::half>(
    const test_utils::half & a,
    const test_utils::half & b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hle(a, b);
#else
    return test_utils::native_half(a) <= test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::greater::operator()<test_utils::half>(
    const test_utils::half & a,
    const test_utils::half & b) const
{
#if defined(__HIP_DEVICE_COMPILE__) && defined(__HIP_PLATFORM_AMD__) || __CUDA_ARCH__ >= 530
    return __hgt(a, b);
#else
    return test_utils::native_half(a) > test_utils::native_half(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::greater_equal::operator()<test_utils::half>(
    const test_utils::half & a,
    const test_utils::half & b) const
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
HIPCUB_HOST_DEVICE inline bool test_utils::less::operator()<test_utils::bfloat16>(
    const test_utils::bfloat16 & a,
    const test_utils::bfloat16 & b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a < b;
#else
    return test_utils::native_bfloat16(a) < test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::less_equal::operator()<test_utils::bfloat16>(
    const test_utils::bfloat16 & a,
    const test_utils::bfloat16 & b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a <= b;
#else
    return test_utils::native_bfloat16(a) <= test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::greater::operator()<test_utils::bfloat16>(
    const test_utils::bfloat16 & a,
    const test_utils::bfloat16 & b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a > b;
#else
    return test_utils::native_bfloat16(a) > test_utils::native_bfloat16(b);
#endif
}

template<>
HIPCUB_HOST_DEVICE inline bool test_utils::greater_equal::operator()<test_utils::bfloat16>(
    const test_utils::bfloat16 & a,
    const test_utils::bfloat16 & b) const
{
#if defined(__HIP_DEVICE_COMPILE__)
    return a >= b;
#else
    return test_utils::native_bfloat16(a) >= test_utils::native_bfloat16(b);
#endif
}
// END BFLOAT16



}
#endif // TEST_UTILS_SORT_COMPARATOR_HPP_
