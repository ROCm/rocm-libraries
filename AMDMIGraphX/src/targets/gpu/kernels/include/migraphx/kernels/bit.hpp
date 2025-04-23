/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_KERNELS_BIT_HPP
#define MIGRAPHX_GUARD_KERNELS_BIT_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/debug.hpp>

namespace migraphx {

constexpr bool get_bit(uint32_t x, uint32_t i) noexcept
{
    MIGRAPHX_ASSERT(i < 32);
    return ((x >> i) & 1u) != 0;
}

constexpr uint64_t bit_ceil(uint64_t x) noexcept
{
    if(x <= 1)
        return 1;
    --x;
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    x |= x >> 32u;
    return x + 1;
}

constexpr uint32_t bit_ceil(uint32_t x) noexcept
{
    if(x <= 1)
        return 1;
    --x;
    x |= x >> 1u;
    x |= x >> 2u;
    x |= x >> 4u;
    x |= x >> 8u;
    x |= x >> 16u;
    return x + 1;
}

constexpr uint32_t popcount(uint32_t x) noexcept { return __popc(x); }

constexpr uint32_t popcount(uint64_t x) noexcept { return __popcll(x); }

constexpr uint32_t countr_zero(uint32_t x) noexcept
{
    // popcount(~(x | −x))
    return __builtin_ctz(x);
}

constexpr uint32_t countr_zero(uint64_t x) noexcept { return __builtin_ctzll(x); }

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_BIT_HPP
