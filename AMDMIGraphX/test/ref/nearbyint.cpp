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

TEST_CASE(nearbyint_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4, 4}};
    auto l = mm->add_literal(migraphx::literal{s,
                                               {-3.51,
                                                -3.5,
                                                -3.49,
                                                -2.51,
                                                -2.50,
                                                -2.49,
                                                -1.6,
                                                -1.5,
                                                -0.51,
                                                -0.5,
                                                0.5,
                                                0.6,
                                                2.4,
                                                2.5,
                                                3.5,
                                                4.5}});
    mm->add_instruction(migraphx::make_op("nearbyint"), l);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        -4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(nearbyint_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape::dynamic_dimension dd{4, 10};
    migraphx::shape s{migraphx::shape::float_type, {dd}};
    auto input = mm->add_parameter("X", s);
    mm->add_instruction(migraphx::make_op("nearbyint"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data{-3.51,
                                  -3.5,
                                  -3.49,
                                  -2.51,
                                  -2.50,
                                  -2.49,
                                  -1.6,
                                  -1.5,
                                  -0.51,
                                  -0.5,
                                  0.5,
                                  0.6,
                                  2.4,
                                  2.5,
                                  3.5,
                                  4.5};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {16}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {
        -4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
