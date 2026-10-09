// Copyright (c) 2017-2026 Advanced Micro Devices, Inc. All rights reserved.
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

/// \addtogroup utilsmodule_functional
/// @{

BEGIN_ROCPRIM_NAMESPACE

/// \brief Unused macro.
/// \deprecated Will be removed in ROCm 9.0
#define ROCPRIM_PRINT_ERROR_ONCE(message) \
    {}

/// \brief Returns the maximum of its arguments.
template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T max(const T& a, const T& b)
{
    return a < b ? b : a;
}

/// \brief Returns the minimum of its arguments.
template<class T>
ROCPRIM_HOST_DEVICE inline
constexpr T min(const T& a, const T& b)
{
    return a < b ? a : b;
}

/// \brief Swaps two values.
template<class T>
ROCPRIM_HOST_DEVICE inline
void swap(T& a, T& b)
{
    T c = a;
    a = b;
    b = c;
}

namespace detail
{

enum class swap_method
{
    ternary  = 0,
    branched = 1
};

/// \brief The ternary conditional swap and the branched conditional swap are optimized
/// differently by the compiler, which can lead to performance variations. This internal API
/// allows switching between these two swap implementations, enabling the selection of
/// the most suitable one for a given algorithm.
/// \tparam Method Enable or disable branched conditional swap
/// \tparam T [inferred] the type of inputs
template<swap_method Method, class T>
ROCPRIM_HOST_DEVICE ROCPRIM_FORCE_INLINE
void swap_if(bool cond, T& a, T& b)
{
    if constexpr(Method == rocprim::detail::swap_method::branched)
    {
        if(cond)
        {
            ::rocprim::swap(a, b);
        }
    }
    else if constexpr(Method == rocprim::detail::swap_method::ternary)
    {
        T a_ = a;
        T b_ = b;
        a    = cond ? b_ : a_;
        b    = cond ? a_ : b_;
    }
    else
    {
        static_assert(false, "This swap method is not supported yet");
    }
}

} // namespace detail

/// \brief Returns true if a < b. Otherwise returns false.
template<class T = void>
struct less
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a < b;
    }
};

/// \brief Returns true if a < b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct less<void>
{
    /// \brief Invocation operator
    template<class T, class U>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const U& b) const
    {
        return a < b;
    }
};

/// \brief Functor that returns true if a <= b. Otherwise returns false.
template<class T = void>
struct less_equal
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

/// \brief Functor that returns true if a <= b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct less_equal<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a <= b;
    }
};

/// \brief Functor that returns true if a > b. Otherwise returns false.
template<class T = void>
struct greater
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

/// \brief Functor that returns true if a > b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct greater<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a > b;
    }
};

/// \brief Functor that returns true if a >= b. Otherwise returns false.
template<class T = void>
struct greater_equal
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

/// \brief Functor that returns true if a >= b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct greater_equal<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a >= b;
    }
};

/// \brief Functor that returns true if a == b. Otherwise returns false.
template<class T = void>
struct equal_to
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

/// \brief Functor that returns true if a == b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct equal_to<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a == b;
    }
};

/// \brief Functor that returns true if a != b. Otherwise returns false.
template<class T = void>
struct not_equal_to
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

/// \brief Functor that returns true if a != b. Otherwise returns false.
/// This version is a specialization for type void.
template<>
struct not_equal_to<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr bool operator()(const T& a, const T& b) const
    {
        return a != b;
    }
};

/// \brief Functor that returns a + b.
template<class T = void>
struct plus
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

/// \brief Functor that returns a + b.
/// This version is a specialization for type void.
template<>
struct plus<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a + b;
    }
};

/// \brief Functor that returns a - b.
template<class T = void>
struct minus
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

/// \brief Functor that returns a - b.
/// This version is a specialization for type void.
template<>
struct minus<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a - b;
    }
};

/// \brief Functor that returns a * b.
template<class T = void>
struct multiplies
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

/// \brief Functor that returns a * b.
/// This version is a specialization for type void.
template<>
struct multiplies<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a * b;
    }
};

/// \brief Functor that returns the maximum of its arguments.
template<class T = void>
struct maximum
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

/// \brief Functor that returns the maximum of its arguments.
/// This version is a specialization for type void.
template<>
struct maximum<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? b : a;
    }
};

/// \brief Functor that returns the minimum of its arguments.
template<class T = void>
struct minimum
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

/// \brief Functor that returns the minimum of its arguments.
/// This version is a specialization for type void.
template<>
struct minimum<void>
{
    /// \brief Invocation operator
    template <typename T>
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a, const T& b) const
    {
        return a < b ? a : b;
    }
};

/// \brief Functor that returns its argument.
template<class T = void>
struct identity
{
    /// \brief Invocation operator
    ROCPRIM_HOST_DEVICE inline
    constexpr T operator()(const T& a) const
    {
        return a;
    }
};

/// \brief Functor that returns its argument.
/// This version is a specialization for type void.
template<>
struct identity<void>
{
    /// \brief Invocation operator
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

template <int A>
struct Int2Type
{
   enum {VALUE = A};
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

END_ROCPRIM_NAMESPACE

/// @}
// end of group utilsmodule_functional

#endif // ROCPRIM_FUNCTIONAL_HPP_
