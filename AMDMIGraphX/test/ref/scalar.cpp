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

TEST_CASE(imagescaler_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {1, 3, 2, 2}};
    auto img           = mm->add_literal(migraphx::literal{s,
                                                           {0.2,
                                                            0.3,
                                                            0.5,
                                                            0.4,

                                                            0.7,
                                                            0.8,
                                                            0.1,
                                                            0.9,

                                                            0.15,
                                                            0.25,
                                                            0.35,
                                                            0.45}});
    auto scale_val     = mm->add_literal(2.f);
    auto scaled_tensor = mm->add_instruction(
        migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), scale_val);
    auto img_scaled = mm->add_instruction(migraphx::make_op("mul"), img, scaled_tensor);
    auto bias_vals  = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {3}}, {0.01, 0.02, 0.03}});
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s.lens()}}), bias_vals);
    mm->add_instruction(migraphx::make_op("add"), img_scaled, bias_bcast);
    p.compile(migraphx::make_target("ref"));
    auto result = p.eval({}).back();
    std::vector<float> results_vector(12);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0.41,
                               0.61,
                               1.01,
                               0.81,

                               1.42,
                               1.62,
                               0.22,
                               1.82,

                               0.33,
                               0.53,
                               0.73,
                               0.93};
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold));
}
