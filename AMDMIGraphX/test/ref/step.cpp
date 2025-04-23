/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(step_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 4 * 6);
    std::iota(data.begin(), data.end(), 2);
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto r  = mm->add_instruction(
        migraphx::make_op("step", {{"axes", {0, 2, 3}}, {"steps", {2, 2, 3}}}), l0);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    migraphx::shape s2{migraphx::shape::float_type, {1, 1, 2, 2}};
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(step_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(2 * 4 * 6);
    std::iota(data.begin(), data.end(), 2);
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4, 6}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    auto tl =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), l0);
    auto r = mm->add_instruction(
        migraphx::make_op("step", {{"axes", {0, 1, 2}}, {"steps", {2, 2, 3}}}), tl);
    mm->add_return({r});
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    migraphx::shape s2{migraphx::shape::float_type, {1, 2, 2, 1}};
    EXPECT(result.get_shape() == s2);
}
