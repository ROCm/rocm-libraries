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

TEST_CASE(logical_xor_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::bool_type, {4}};
    std::vector<bool> data1{true, false, true, false};
    std::vector<bool> data2{true, true, false, false};
    auto l1 = mm->add_literal(migraphx::literal{s, data1});
    auto l2 = mm->add_literal(migraphx::literal{s, data2});
    mm->add_instruction(migraphx::make_op("logical_xor"), l1, l2);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, true, false};
    std::transform(
        data1.begin(), data1.end(), data2.begin(), gold.begin(), [](bool n1, bool n2) -> bool {
            return n1 ^ n2;
        });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(logical_xor_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dd{{2, 6, {4}}};
    migraphx::shape s{migraphx::shape::bool_type, dd};
    auto left  = mm->add_parameter("l", s);
    auto right = mm->add_parameter("r", s);
    mm->add_instruction(migraphx::make_op("logical_xor"), left, right);
    p.compile(migraphx::make_target("ref"));

    std::vector<char> left_data{1, 0, 1, 0};
    std::vector<char> right_data{1, 1, 0, 0};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::bool_type, {4}};
    params0["l"] = migraphx::argument(input_fixed_shape0, left_data.data());
    params0["r"] = migraphx::argument(input_fixed_shape0, right_data.data());
    auto result  = p.eval(params0).back();
    std::vector<char> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<bool> gold = {false, true, true, false};
    std::transform(left_data.begin(),
                   left_data.end(),
                   right_data.begin(),
                   gold.begin(),
                   [](bool n1, bool n2) -> bool { return n1 ^ n2; });
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
