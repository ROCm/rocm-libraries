/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP
#define ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP

#include <rocm/config.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T, T V>
struct integral_constant
{
    static constexpr T value = V;
    using value_type         = T;
    using type               = integral_constant;
    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
    static constexpr type to() { return {}; }
};

// NOLINTNEXTLINE
#define ROCM_INTEGRAL_CONSTANT_BINARY_OP(op)                                    \
    template <class T, T V, class U, U w>                                       \
    constexpr inline integral_constant<decltype(V op w), (V op w)> operator op( \
        integral_constant<T, V>, integral_constant<U, w>) noexcept              \
    {                                                                           \
        return {};                                                              \
    }

// NOLINTNEXTLINE
#define ROCM_INTEGRAL_CONSTANT_UNARY_OP(op)                                 \
    template <class T, T V>                                                 \
    constexpr inline integral_constant<decltype(op V), (op V)> operator op( \
        integral_constant<T, V>) noexcept                                   \
    {                                                                       \
        return {};                                                          \
    }

ROCM_INTEGRAL_CONSTANT_BINARY_OP(+)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(-)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(*)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(/)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(%)
// NOLINTBEGIN(hicpp-signed-bitwise)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>>)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<<)
// NOLINTEND(hicpp-signed-bitwise)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(<=)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(>=)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(==)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(!=)
// NOLINTBEGIN(hicpp-signed-bitwise)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(&)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(^)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(|)
// NOLINTEND(hicpp-signed-bitwise)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(and)
ROCM_INTEGRAL_CONSTANT_BINARY_OP(or)

ROCM_INTEGRAL_CONSTANT_UNARY_OP(not)
// NOLINTNEXTLINE(hicpp-signed-bitwise)
ROCM_INTEGRAL_CONSTANT_UNARY_OP(~)
ROCM_INTEGRAL_CONSTANT_UNARY_OP(+)
ROCM_INTEGRAL_CONSTANT_UNARY_OP(-)

template <bool B>
using bool_constant = integral_constant<bool, B>;

using true_type  = bool_constant<true>;
using false_type = bool_constant<false>;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_INTEGRAL_CONSTANT_HPP
