// Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCPRIM_TEST_UTILS_HALF_HPP
#define ROCPRIM_TEST_UTILS_HALF_HPP

namespace test_utils
{
using half = rocprim::half;
// Support half operators on host side

ROCPRIM_HOST inline rocprim::native_half half_to_native(const rocprim::half& x)
{
    return *reinterpret_cast<const rocprim::native_half*>(&x);
}

ROCPRIM_HOST inline rocprim::half native_to_half(const rocprim::native_half& x)
{
    return *reinterpret_cast<const rocprim::half*>(&x);
}

struct half_less
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return half_to_native(a) < half_to_native(b);
        #endif
    }
};

struct half_less_equal
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a <= b;
        #else
        return half_to_native(a) <= half_to_native(b);
        #endif
    }
};

struct half_greater
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a > b;
        #else
        return half_to_native(a) > half_to_native(b);
        #endif
    }
};

struct half_greater_equal
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a >= b;
        #else
        return half_to_native(a) >= half_to_native(b);
        #endif
    }
};

struct half_equal_to
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return half_to_native(a) == half_to_native(b);
        #endif
    }
};

struct half_not_equal_to
{
    ROCPRIM_HOST_DEVICE inline bool operator()(const rocprim::half& a,
                                               const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a != b;
        #else
        return half_to_native(a) != half_to_native(b);
        #endif
    }
};

struct half_plus
{
    ROCPRIM_HOST_DEVICE inline rocprim::half operator()(const rocprim::half& a,
                                                        const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_half(half_to_native(a) + half_to_native(b));
        #endif
    }
};

struct half_minus
{
    ROCPRIM_HOST_DEVICE inline rocprim::half operator()(const rocprim::half& a,
                                                        const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a - b;
        #else
        return native_to_half(half_to_native(a) - half_to_native(b));
        #endif
    }
};

struct half_multiplies
{
    ROCPRIM_HOST_DEVICE inline rocprim::half operator()(const rocprim::half& a,
                                                        const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a * b;
        #else
        return native_to_half(half_to_native(a) * half_to_native(b));
        #endif
    }
};

struct half_maximum
{
    ROCPRIM_HOST_DEVICE inline rocprim::half operator()(const rocprim::half& a,
                                                        const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? b : a;
        #else
        return half_to_native(a) < half_to_native(b) ? b : a;
        #endif
    }
};

struct half_minimum
{
    ROCPRIM_HOST_DEVICE inline rocprim::half operator()(const rocprim::half& a,
                                                        const rocprim::half& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? a : b;
        #else
        return half_to_native(a) < half_to_native(b) ? a : b;
        #endif
    }
};

} // end of namespace test_utils

// For better Google Test reporting and debug output of half values
inline std::ostream& operator<<(std::ostream& stream, const rocprim::half& value)
{
    stream << static_cast<float>(value);
    return stream;
}

#endif //ROCPRIM_TEST_UTILS_HALF_HPP
