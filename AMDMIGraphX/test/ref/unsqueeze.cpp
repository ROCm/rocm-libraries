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
#include <migraphx/generate.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

TEST_CASE(unsqueeze_test_1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(4 * 3 * 3);
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(unsqueeze_test_2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<float> data(4 * 3 * 3);
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 3, 1, 3}};
    auto l0 = mm->add_literal(migraphx::literal{s1, data});
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(unsqueeze_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s1{migraphx::shape::float_type, {{1, 4}, {3, 3}, {3, 3}}};
    auto p0 = mm->add_parameter("x", s1);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), p0);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data(4 * 3 * 3);
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {4, 3, 3}};
    params0["x"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    migraphx::shape s2{migraphx::shape::float_type, {4, 1, 3, 3}};
    EXPECT(result.get_shape() == s2);
}

TEST_CASE(unsqueeze_transpose_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {4, 3, 3}};
    auto l0 = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_trans =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0_trans);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {3, 4, 1, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(unsqueeze_multibroadcast_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {4, 1, 3}};
    auto l0 = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_brcst =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 4, 3, 3}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), l0_brcst);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {4, 4, 1, 3, 3}});
    EXPECT(result == expected_result);
}

TEST_CASE(unsqueeze_slice_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s1{migraphx::shape::float_type, {2, 3, 4, 4}};
    auto l0       = mm->add_literal(migraphx::generate_literal(s1));
    auto l0_slice = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {2}}, {"ends", {3}}}), l0);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l0_slice);
    auto p_uncompiled   = p;
    auto* mm_uncompiled = p_uncompiled.get_main_module();
    mm_uncompiled->add_instruction(migraphx::make_op("contiguous"),
                                   std::prev(mm_uncompiled->end()));
    p.compile(migraphx::make_target("ref"));
    auto result          = p.eval({}).back();
    auto expected_result = p_uncompiled.eval({}).back();
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::float_type, {2, 1, 3, 4, 1}});
    EXPECT(result == expected_result);
}
