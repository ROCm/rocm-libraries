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
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/register_target.hpp>
#include "test.hpp"

TEST_CASE(run_on_target_shape_tests)
{
    {
        test::throws([]() { migraphx::make_op("run_on_target"); });
    }
    {
        migraphx::program p;
        auto s        = migraphx::shape{migraphx::shape::float_type, {12, 12}};
        auto* mm      = p.get_main_module();
        auto x        = mm->add_parameter("x", s);
        auto* run_mod = p.create_module("run_mod");
        run_mod->add_return({x, x});
        test::throws(
            [&]() { mm->add_instruction(migraphx::make_op("run_on_target"), {x}, {run_mod}); });
        test::throws([&]() {
            mm->add_instruction(migraphx::make_op("run_on_target"), {x}, {run_mod, run_mod});
        });
    }
}

TEST_CASE(eval_run_on_target)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {3}};
    auto l         = mm->add_literal(migraphx::literal{s, {4.0, 16.0, 64.0}});
    auto* ref_mod  = p.create_module("ref_mod");
    auto ref_rsqrt = ref_mod->add_instruction(migraphx::make_op("rsqrt"), l);
    ref_mod->add_return({ref_rsqrt});
    auto run_on_ins =
        mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}), {}, {ref_mod});
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), run_on_ins);
    p.compile({migraphx::make_target("ref")});
    auto result = p.eval({}).back();
    std::vector<float> results_vector(3);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.5, 0.25, 0.125};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
