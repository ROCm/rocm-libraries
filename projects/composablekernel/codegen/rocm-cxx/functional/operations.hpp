/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */
#ifndef ROCM_GUARD_FUNCTIONAL_OPERATIONS_HPP
#define ROCM_GUARD_FUNCTIONAL_OPERATIONS_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ROCM_FUNCTIONAL_BINARY_OP(name, op, result)                                  \
    template <class T = void>                                                        \
    struct name                                                                      \
    {                                                                                \
        constexpr result operator()(const T& x, const T& y) const { return x op y; } \
    };                                                                               \
    template <>                                                                      \
    struct name<void>                                                                \
    {                                                                                \
        using is_transparent = void;                                                 \
        template <class T, class U>                                                  \
        constexpr auto operator()(T&& x, U&& y) const                                \
            noexcept(noexcept(static_cast<T&&>(x) op static_cast<U&&>(y)))           \
                -> decltype(static_cast<T&&>(x) op static_cast<U&&>(y))              \
        {                                                                            \
            return static_cast<T&&>(x) op static_cast<U&&>(y);                       \
        }                                                                            \
    };

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define ROCM_FUNCTIONAL_UNARY_OP(name, op, result)                                        \
    template <class T = void>                                                             \
    struct name                                                                           \
    {                                                                                     \
        constexpr result operator()(const T& x) const { return op x; }                    \
    };                                                                                    \
    template <>                                                                           \
    struct name<void>                                                                     \
    {                                                                                     \
        using is_transparent = void;                                                      \
        template <class T>                                                                \
        constexpr auto operator()(T&& x) const noexcept(noexcept(op static_cast<T&&>(x))) \
            -> decltype(op static_cast<T&&>(x))                                           \
        {                                                                                 \
            return op static_cast<T&&>(x);                                                \
        }                                                                                 \
    };

ROCM_FUNCTIONAL_BINARY_OP(plus, +, T)
ROCM_FUNCTIONAL_BINARY_OP(minus, -, T)
ROCM_FUNCTIONAL_BINARY_OP(multiplies, *, T)
ROCM_FUNCTIONAL_BINARY_OP(divides, /, T)
ROCM_FUNCTIONAL_BINARY_OP(modulus, %, T)
// NOLINTBEGIN(hicpp-signed-bitwise)
ROCM_FUNCTIONAL_BINARY_OP(bit_and, &, T)
ROCM_FUNCTIONAL_BINARY_OP(bit_or, |, T)
ROCM_FUNCTIONAL_BINARY_OP(bit_xor, ^, T)
// NOLINTEND(hicpp-signed-bitwise)

ROCM_FUNCTIONAL_BINARY_OP(equal_to, ==, bool)
ROCM_FUNCTIONAL_BINARY_OP(not_equal_to, !=, bool)
ROCM_FUNCTIONAL_BINARY_OP(greater, >, bool)
ROCM_FUNCTIONAL_BINARY_OP(less, <, bool)
ROCM_FUNCTIONAL_BINARY_OP(greater_equal, >=, bool)
ROCM_FUNCTIONAL_BINARY_OP(less_equal, <=, bool)
ROCM_FUNCTIONAL_BINARY_OP(logical_and, and, bool)
ROCM_FUNCTIONAL_BINARY_OP(logical_or, or, bool)

ROCM_FUNCTIONAL_UNARY_OP(negate, -, T)
ROCM_FUNCTIONAL_UNARY_OP(logical_not, not, bool)
// NOLINTNEXTLINE(hicpp-signed-bitwise)
ROCM_FUNCTIONAL_UNARY_OP(bit_not, ~, T)

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // MIGRAPHX_GUARD_FUNCTIONAL_OPERATIONS_HPP
