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
#ifndef ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP
#define ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP

#include <rocm/config.hpp>
#include <rocm/stddef.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <class T, T... Ints>
struct integer_sequence
{
    using value_type = T;

    static constexpr size_t size() noexcept { return sizeof...(Ints); }
};

template <size_t... Ints>
using index_sequence = integer_sequence<size_t, Ints...>;

template <class T, T N>
using make_integer_sequence = __make_integer_seq<integer_sequence, T, N>;

template <size_t N>
using make_index_sequence = make_integer_sequence<size_t, N>;

template <class... Ts>
using index_sequence_for = make_index_sequence<sizeof...(Ts)>;

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_UTILITY_INTEGER_SEQUENCE_HPP
