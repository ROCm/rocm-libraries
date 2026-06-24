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
#ifndef ROCM_GUARD_ROCM_BIT_HPP
#define ROCM_GUARD_ROCM_BIT_HPP

#include <rocm/assert.hpp>
#include <rocm/config.hpp>
#include <rocm/type_traits.hpp>
#include <rocm/limits.hpp>

namespace rocm {
inline namespace ROCM_INLINE_NS {

template <typename To,
          typename From,
          ROCM_REQUIRES(rocm::is_trivially_copyable<To>{} and
                        rocm::is_trivially_copyable<From>{} and sizeof(To) == sizeof(From))>
constexpr To bit_cast(From fr) noexcept
{
    return __builtin_bit_cast(To, fr);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countl_zero(T x) noexcept
{
    return __builtin_clzg(x, numeric_limits<T>::digits);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countl_one(T x) noexcept
{
    return countl_zero(T(~x));
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countr_zero(T x) noexcept
{
    return __builtin_ctzg(x, numeric_limits<T>::digits);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int countr_one(T x) noexcept
{
    return countr_zero(T(~x));
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int popcount(T x) noexcept
{
    return __builtin_popcountg(x);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr int bit_width(T x) noexcept
{
    return numeric_limits<T>::digits - countl_zero(x);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T bit_floor(T x) noexcept
{
    if(x != 0)
    {
        unsigned shift = bit_width(x) - 1;
        return T{1} << shift;
    }
    return 0;
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T bit_ceil(T x) noexcept
{
    if(x <= 1)
        return 1;
    unsigned e = bit_width(T(x - 1));
    ROCM_ASSERT(e < numeric_limits<T>::digits);
    if constexpr(is_same<T, decltype(+x)>{})
        return T{1} << e;
    constexpr unsigned offset_for_ub = numeric_limits<unsigned>::digits - numeric_limits<T>::digits;
    return 1u << (e + offset_for_ub) >> offset_for_ub;
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr bool has_single_bit(T x) noexcept
{
    return popcount(x) == 1;
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T rotl(T x, int s) noexcept
{
    constexpr int n = numeric_limits<T>::digits;
    int r           = s % n;

    if(r == 0)
        return x;

    if(r > 0)
    {
        unsigned ur    = r;
        unsigned shift = n - r;
        // NOLINTNEXTLINE(hicpp-signed-bitwise)
        return (x << ur) | (x >> shift);
    }

    unsigned ur    = -r;
    unsigned shift = n + r;
    // NOLINTNEXTLINE(hicpp-signed-bitwise)
    return (x >> ur) | (x << shift);
}

template <class T, ROCM_REQUIRES(rocm::is_unsigned<T>{})>
constexpr T rotr(T x, int s) noexcept
{
    return rotl(x, -s);
}

} // namespace ROCM_INLINE_NS
} // namespace rocm
#endif // ROCM_GUARD_ROCM_BIT_HPP
