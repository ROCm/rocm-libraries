// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#define ROCPRIM_PRINT_ERROR_ONCE(message) \
{                                          \
    unsigned int idx = threadIdx.x + (blockIdx.x * blockDim.x); \
    idx += threadIdx.y + (blockIdx.y * blockDim.y);             \
    idx += threadIdx.z + (blockIdx.z * blockDim.z);             \
    if (idx == 0)                                                        \
        printf("%s\n", #message);                                        \
}

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

template<class T = void>
struct less_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

template<>
struct less_equal<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

template<class T = void>
struct greater
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

template<>
struct greater<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

template<class T = void>
struct greater_equal
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

template<>
struct greater_equal<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

template<class T = void>
struct equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

template<>
struct equal_to<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

template<class T = void>
struct not_equal_to
{
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template<>
struct not_equal_to<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

template<class T = void>
struct plus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<>
struct plus<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

template<class T = void>
struct minus
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

template<>
struct minus<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

template<class T = void>
struct multiplies
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

template<>
struct multiplies<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

template<class T = void>
struct maximum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template<>
struct maximum<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

template<class T = void>
struct minimum
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

template<>
struct minimum<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

template<class T = void>
struct identity
{
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};

template<>
struct identity<void>
{
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};

/**
 * \brief Statically determine log2(N), rounded up.
 *
 * For example:
 *     Log2<8>::VALUE   // 3
 *     Log2<3>::VALUE   // 2
 */
template <int N, int CURRENT_VAL = N, int COUNT = 0>
struct Log2
{
    /// Static logarithm value
    enum { VALUE = Log2<N, (CURRENT_VAL >> 1), COUNT + 1>::VALUE };         // Inductive case
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <int N, int COUNT>
struct Log2<N, 0, COUNT>
{
    enum {VALUE = (1 << (COUNT - 1) < N) ?                                  // Base case
        COUNT :
        COUNT - 1 };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

/******************************************************************************
 * Conditional types
 ******************************************************************************/

/**
 * \brief Type equality test
 */
template <typename A, typename B>
struct Equals
{
    enum {
        VALUE = 0,
        NEGATE = 1
    };
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

template <typename A>
struct Equals <A, A>
{
    enum {
        VALUE = 1,
        NEGATE = 0
    };
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

template <int A>
struct Int2Type
{
   enum {VALUE = A};
};

/// @}
// end of group utilsmodule_functional

END_ROCPRIM_NAMESPACE

#endif // ROCPRIM_FUNCTIONAL_HPP_
