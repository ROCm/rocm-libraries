// Copyright (c) 2017-2019 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCPRIM_FUNCTIONAL_HPP_
#define ROCPRIM_FUNCTIONAL_HPP_

#include <functional>

// Meta configuration for rocPRIM
#include "config.hpp"

BEGIN_ROCPRIM_NAMESPACE

/// \addtogroup utilsmodule_functional
/// @{

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T max(const T& a, const T& b)
{
    return a < b ? b : a;
}

template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T min(const T& a, const T& b)
{
    return a < b ? a : b;
}

template<class T>
ROCPRIM_HOST_DEVICE inline
void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

template<class T = void>
struct less
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a < b;
    }
};

template<>
struct less<void>
{
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const U& b) const
    {
        return a < b;
    }
};

template<class T>
struct less_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

template<class T>
struct greater
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

template<class T>
struct greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

template<class T>
struct equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

template<class T>
struct not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template<class T>
struct plus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<class T>
struct minus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

template<class T>
struct multiplies
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

template<class T>
struct maximum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template<class T>
struct minimum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

template<class T>
struct identity
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};

/// @}
// end of group utilsmodule_functional

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_FUNCTIONAL_HPP_
