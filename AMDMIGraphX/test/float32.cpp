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
 */
#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/bit_cast.hpp>
#include <migraphx/ranges.hpp>
#include "test.hpp"
#include <iostream>

#include <limits>

using fp32 = migraphx::generic_float<23, 8>;

template <class T, class U>
static bool bit_equal(const T& x, const U& y)
{
    static_assert(sizeof(T) == sizeof(U));
    using type = std::array<char, sizeof(T)>;
    return migraphx::bit_cast<type>(x) == migraphx::bit_cast<type>(y);
}
// NOLINTNEXTLINE
#define MIGRAPHX_CHECK_FLOAT(x, y)     \
    CHECK(bit_equal(x, y));            \
    CHECK(bit_equal(x, y.to_float())); \
    CHECK(bit_equal(fp32{x}, y));      \
    CHECK(bit_equal(fp32{x}.to_float(), y.to_float()))

TEST_CASE(fp32_values_working)
{
    MIGRAPHX_CHECK_FLOAT(1.0f, fp32{1.0f});
    MIGRAPHX_CHECK_FLOAT(-1.0f, fp32{-1.0f});
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::min(), fp32::min());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::lowest(), fp32::lowest());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::max(), fp32::max());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::epsilon(), fp32::epsilon());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::denorm_min(), fp32::denorm_min());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::infinity(), fp32::infinity());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::quiet_NaN(), fp32::qnan());
    MIGRAPHX_CHECK_FLOAT(std::numeric_limits<float>::signaling_NaN(), fp32::snan());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
