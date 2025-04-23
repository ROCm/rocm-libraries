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
#include <basic_ops.hpp>
#include <migraphx/autocast_fp8.hpp>
#include <migraphx/float8.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <test.hpp>

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::autocast_fp8_pass{}, migraphx::eliminate_identity{}});
}

// with return
template <migraphx::shape::type_t DType>
static void autocast_fp8_1()
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {DType, {1}});
        auto y   = m1.add_parameter("y", {DType, {1}});
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({sum});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto y_fp32 = m2.add_parameter("y", {migraphx::shape::float_type, {1}});
        auto x_fp32 = m2.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto y_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), y_fp32);
        auto x_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), x_fp32);
        auto sum_fp8  = m2.add_instruction(migraphx::make_op("add"), x_fp8, y_fp8);
        auto sum_fp32 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sum_fp8);
        m2.add_return({sum_fp32});
    }
    EXPECT(m1 == m2);
}
TEST_CASE_REGISTER(autocast_fp8_1<migraphx::shape::fp8e4m3fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_1<migraphx::shape::fp8e5m2fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_1<migraphx::shape::fp8e4m3fn_type>);
TEST_CASE_REGISTER(autocast_fp8_1<migraphx::shape::fp8e5m2_type>);

// without return
template <migraphx::shape::type_t DType>
static void autocast_fp8_2()
{
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {DType, {1}});
        auto y = m1.add_parameter("y", {DType, {1}});
        m1.add_instruction(migraphx::make_op("sub"), x, y);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto y_fp32 = m2.add_parameter("y", {migraphx::shape::float_type, {1}});
        auto x_fp32 = m2.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto y_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), y_fp32);
        auto x_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), x_fp32);
        m2.add_instruction(migraphx::make_op("sub"), x_fp8, y_fp8);
    }
    EXPECT(m1 == m2);
}
TEST_CASE_REGISTER(autocast_fp8_2<migraphx::shape::fp8e4m3fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_2<migraphx::shape::fp8e5m2fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_2<migraphx::shape::fp8e4m3fn_type>);
TEST_CASE_REGISTER(autocast_fp8_2<migraphx::shape::fp8e5m2_type>);

// multiple inputs (of same type) to return
template <migraphx::shape::type_t DType>
static void autocast_fp8_3()
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {DType, {1}});
        auto y    = m1.add_parameter("y", {DType, {1}});
        auto sum  = m1.add_instruction(migraphx::make_op("add"), x, y);
        auto diff = m1.add_instruction(migraphx::make_op("sub"), x, y);
        m1.add_return({sum, diff});
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto y_fp32 = m2.add_parameter("y", {migraphx::shape::float_type, {1}});
        auto x_fp32 = m2.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto y_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), y_fp32);
        auto x_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), x_fp32);
        auto sum_fp8  = m2.add_instruction(migraphx::make_op("add"), x_fp8, y_fp8);
        auto diff_fp8 = m2.add_instruction(migraphx::make_op("sub"), x_fp8, y_fp8);
        auto sum_fp32 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sum_fp8);
        auto diff_fp32 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), diff_fp8);
        m2.add_return({sum_fp32, diff_fp32});
    }
    EXPECT(m1 == m2);
}
TEST_CASE_REGISTER(autocast_fp8_3<migraphx::shape::fp8e4m3fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_3<migraphx::shape::fp8e5m2fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_3<migraphx::shape::fp8e4m3fn_type>);
TEST_CASE_REGISTER(autocast_fp8_3<migraphx::shape::fp8e5m2_type>);

// multiple inputs (of different types) to return
template <migraphx::shape::type_t DType>
static void autocast_fp8_4()
{
    migraphx::module m1;
    {
        auto x1   = m1.add_parameter("x1", {DType, {3, 4}, {1, 3}});
        auto y1   = m1.add_parameter("y1", {DType, {3, 4}, {1, 3}});
        auto x2   = m1.add_parameter("x2", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto y2   = m1.add_parameter("y2", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto sum1 = m1.add_instruction(migraphx::make_op("add"), x1, y1);
        auto sum2 = m1.add_instruction(migraphx::make_op("add"), x2, y2);
        m1.add_return({sum1, sum2});
    }
    run_pass(m1);

    migraphx::module m2;
    {

        auto x2     = m2.add_parameter("x2", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto y2     = m2.add_parameter("y2", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto y1     = m2.add_parameter("y1", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto x1     = m2.add_parameter("x1", {migraphx::shape::float_type, {3, 4}, {1, 3}});
        auto y1_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), y1);
        auto x1_fp8 =
            m2.add_instruction(migraphx::make_op("convert", {{"target_type", DType}}), x1);
        auto sum1        = m2.add_instruction(migraphx::make_op("add"), x1_fp8, y1_fp8);
        auto sum2        = m2.add_instruction(migraphx::make_op("add"), x2, y2);
        auto result_sum1 = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), sum1);
        m2.add_return({result_sum1, sum2});
    }
    EXPECT(m1 == m2);
}
TEST_CASE_REGISTER(autocast_fp8_4<migraphx::shape::fp8e4m3fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_4<migraphx::shape::fp8e5m2fnuz_type>);
TEST_CASE_REGISTER(autocast_fp8_4<migraphx::shape::fp8e4m3fn_type>);
TEST_CASE_REGISTER(autocast_fp8_4<migraphx::shape::fp8e5m2_type>);

// autocast pass does not do any changes
TEST_CASE(autocast_fp8_5)
{
    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::float_type, {1}});
        auto y   = m1.add_parameter("y", {migraphx::shape::float_type, {1}});
        auto sum = m1.add_instruction(migraphx::make_op("add"), x, y);
        m1.add_return({sum});
    }
    migraphx::module m2 = m1;
    run_pass(m1);
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
