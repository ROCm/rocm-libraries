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

TEST_CASE(clip_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l       = mm->add_literal(migraphx::literal{s, {-1.0, 0.0, 10.0}});
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), min_val);
    max_val =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), max_val);
    mm->add_instruction(migraphx::make_op("clip"), l, min_val, max_val);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

TEST_CASE(clip_dyn_test)
{
    migraphx::program p;
    auto* mm                                            = p.get_main_module();
    std::vector<migraphx::shape::dynamic_dimension> dds = {{2, 8, {3}}};
    migraphx::shape s{migraphx::shape::float_type, dds};
    auto l       = mm->add_parameter("X", s);
    auto min_val = mm->add_literal(0.0f);
    auto max_val = mm->add_literal(6.0f);
    min_val      = mm->add_instruction(migraphx::make_op("multibroadcast"), min_val, l);
    max_val      = mm->add_instruction(migraphx::make_op("multibroadcast"), max_val, l);
    mm->add_instruction(migraphx::make_op("clip"), l, min_val, max_val);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape static_shape{migraphx::shape::float_type, {3}};
    migraphx::parameter_map params;
    std::vector<float> data = {-1.0, 0.0, 10.0};
    params["X"]             = migraphx::argument(static_shape, data.data());
    auto result             = p.eval(params).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.0, 0.0, 6.0};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
