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

#ifndef ROCPRIM_TEST_UTILS_BFLOAT16_HPP
#define ROCPRIM_TEST_UTILS_BFLOAT16_HPP

namespace test_utils {
using bfloat16 = rocprim::bfloat16;

// Support bfloat16 operators on host side
ROCPRIM_HOST inline
    rocprim::native_bfloat16 bfloat16_to_native(const rocprim::bfloat16& x)
{
    return rocprim::native_bfloat16(x);
}

ROCPRIM_HOST inline
    rocprim::bfloat16 native_to_bfloat16(const rocprim::native_bfloat16& x)
{
    return rocprim::bfloat16(x);
}

struct bfloat16_less
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_less_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a <= b;
        #else
        return bfloat16_to_native(a) <= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a > b;
        #else
        return bfloat16_to_native(a) > bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a >= b;
        #else
        return bfloat16_to_native(a) >= bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a == b;
        #else
        return bfloat16_to_native(a) == bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    bool operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a != b;
        #else
        return bfloat16_to_native(a) != bfloat16_to_native(b);
        #endif
    }
};

struct bfloat16_plus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a + b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) + bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_minus
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a - b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) - bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_multiplies
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a * b;
        #else
        return native_to_bfloat16(bfloat16_to_native(a) * bfloat16_to_native(b));
        #endif
    }
};

struct bfloat16_maximum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? b : a;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? b : a;
        #endif
    }
};

struct bfloat16_minimum
{
    ROCPRIM_HOST_DEVICE inline
    rocprim::bfloat16 operator()(const rocprim::bfloat16& a, const rocprim::bfloat16& b) const
    {
        #if __HIP_DEVICE_COMPILE__
        return a < b ? a : b;
        #else
        return bfloat16_to_native(a) < bfloat16_to_native(b) ? a : b;
        #endif
    }
};

}

#endif //ROCPRIM_TEST_UTILS_BFLOAT16_HPP
