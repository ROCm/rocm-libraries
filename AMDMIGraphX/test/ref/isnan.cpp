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

TEST_CASE(isnan_test)
{
    // float test
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {2, 3}};
        auto nan_val             = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> data0 = {1.2, 5.2, nan_val, nan_val, 0., 100.};
        auto l1                  = mm->add_literal(migraphx::literal{s, data0});
        mm->add_instruction(migraphx::make_op("isnan"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 0, 1, 1, 0, 0};
        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }

    // half test
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::half_type, {2, 3}};
        auto nan_val = std::numeric_limits<migraphx::half>::quiet_NaN();
        migraphx::half a{1.2};
        migraphx::half b{5.2};
        std::vector<migraphx::half> data0 = {a, b, nan_val, nan_val, b, a};
        auto l1                           = mm->add_literal(migraphx::literal{s, data0});
        mm->add_instruction(migraphx::make_op("isnan"), l1);
        p.compile(migraphx::make_target("ref"));
        auto result = p.eval({}).back();
        std::vector<float> results_vector;
        result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
        std::vector<float> gold = {0, 0, 1, 1, 0, 0};
        EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
    }
}

TEST_CASE(isnan_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {{2, 2}, {3, 8}}};
    auto input   = mm->add_parameter("X", s);
    auto nan_val = std::numeric_limits<float>::quiet_NaN();
    mm->add_instruction(migraphx::make_op("isnan"), input);
    p.compile(migraphx::make_target("ref"));

    std::vector<float> input_data = {1.2, 5.2, nan_val, nan_val, 0., 100.};
    migraphx::parameter_map params0;
    migraphx::shape input_fixed_shape0{migraphx::shape::float_type, {2, 3}};
    params0["X"] = migraphx::argument(input_fixed_shape0, input_data.data());
    auto result  = p.eval(params0).back();
    std::vector<float> results_vector;
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0, 0, 1, 1, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
