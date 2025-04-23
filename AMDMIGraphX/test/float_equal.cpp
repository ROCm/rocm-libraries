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
#include <migraphx/float_equal.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/half.hpp>
#include "test.hpp"

#include <limits>

template <class T, class U>
struct float_equal_expression
{
    T lhs;
    U rhs;

    operator bool() const { return migraphx::float_equal(lhs, rhs); }

    bool operator not() const { return not bool(*this); }

    friend std::ostream& operator<<(std::ostream& s, const float_equal_expression& self)
    {
        s << "migraphx::float_equal(" << self.lhs << ", " << self.rhs << ")";
        return s;
    }
};

template <class T, class U>
static auto test_float_equal(T x, U y)
{
    return test::make_lhs_expression(float_equal_expression<T, U>{x, y});
}

template <class T, class U>
static void test_equality()
{
    auto x1 = T(0.125);
    auto x2 = U(0.0);
    auto x3 = U(1.0);
    EXPECT(test_float_equal(x1, x1));
    EXPECT(test_float_equal(x2, x2));
    EXPECT(test_float_equal(x3, x3));

    EXPECT(not test_float_equal(x1, x2));
    EXPECT(not test_float_equal(x2, x1));
    EXPECT(not test_float_equal(x1, x3));
    EXPECT(not test_float_equal(x3, x1));
    EXPECT(not test_float_equal(x2, x3));
    EXPECT(not test_float_equal(x3, x2));
}

TEST_CASE_REGISTER(test_equality<double, float>);
TEST_CASE_REGISTER(test_equality<double, int>);
TEST_CASE_REGISTER(test_equality<double, migraphx::half>);
TEST_CASE_REGISTER(test_equality<double, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_equality<double, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_equality<double, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_equality<double, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_equality<float, int>);
TEST_CASE_REGISTER(test_equality<float, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_equality<float, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_equality<float, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_equality<float, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_equality<migraphx::half, int>);
TEST_CASE_REGISTER(test_equality<migraphx::half, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_equality<migraphx::half, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_equality<migraphx::half, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_equality<migraphx::half, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_equality<migraphx::fp8::fp8e4m3fnuz, int>);
TEST_CASE_REGISTER(test_equality<migraphx::fp8::fp8e5m2fnuz, int>);
TEST_CASE_REGISTER(test_equality<migraphx::fp8::fp8e4m3fn, int>);
TEST_CASE_REGISTER(test_equality<migraphx::fp8::fp8e5m2, int>);

template <class T, class U>
static void test_limits()
{
    auto max1 = std::numeric_limits<T>::max();
    auto max2 = std::numeric_limits<U>::max();

    auto min1 = std::numeric_limits<T>::lowest();
    auto min2 = std::numeric_limits<U>::lowest();

    EXPECT(test_float_equal(max1, max1));
    EXPECT(test_float_equal(max2, max2));

    EXPECT(not test_float_equal(max1, max2));
    EXPECT(not test_float_equal(max2, max1));

    EXPECT(test_float_equal(min1, min1));
    EXPECT(test_float_equal(min2, min2));

    EXPECT(not test_float_equal(min1, min2));
    EXPECT(not test_float_equal(min2, min1));

    EXPECT(not test_float_equal(max1, min1));
    EXPECT(not test_float_equal(min1, max1));
    EXPECT(not test_float_equal(max2, min2));
    EXPECT(not test_float_equal(min2, max2));

    EXPECT(not test_float_equal(max1, min2));
    EXPECT(not test_float_equal(min2, max1));

    EXPECT(not test_float_equal(max2, min1));
    EXPECT(not test_float_equal(min1, max2));
}

TEST_CASE_REGISTER(test_limits<double, float>);
TEST_CASE_REGISTER(test_limits<double, int>);
TEST_CASE_REGISTER(test_limits<double, migraphx::half>);
TEST_CASE_REGISTER(test_limits<double, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_limits<double, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_limits<double, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_limits<double, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_limits<float, int>);
TEST_CASE_REGISTER(test_limits<float, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_limits<float, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_limits<float, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_limits<float, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_limits<int, migraphx::half>);
TEST_CASE_REGISTER(test_limits<int, migraphx::fp8::fp8e4m3fnuz>);
TEST_CASE_REGISTER(test_limits<int, migraphx::fp8::fp8e5m2fnuz>);
TEST_CASE_REGISTER(test_limits<int, migraphx::fp8::fp8e4m3fn>);
TEST_CASE_REGISTER(test_limits<int, migraphx::fp8::fp8e5m2>);
TEST_CASE_REGISTER(test_limits<migraphx::fp8::fp8e4m3fnuz, migraphx::half>);
TEST_CASE_REGISTER(test_limits<migraphx::fp8::fp8e5m2fnuz, migraphx::half>);
TEST_CASE_REGISTER(test_limits<migraphx::fp8::fp8e4m3fn, migraphx::half>);
TEST_CASE_REGISTER(test_limits<migraphx::fp8::fp8e5m2, migraphx::half>);

#ifndef _WIN32
// On Windows, types int and long have the same min and max values.
TEST_CASE_REGISTER(test_limits<long, int>);
#endif
TEST_CASE_REGISTER(test_limits<long, char>);

int main(int argc, const char* argv[]) { test::run(argc, argv); }
