/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <onnx_test.hpp>

TEST_CASE(qlinearmul_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto a = mm->add_parameter("A", {migraphx::shape::uint8_type, {64}});
    auto b = mm->add_parameter("B", {migraphx::shape::uint8_type, {64}});

    auto sc_a   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_a = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {0}});

    auto sc_b   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_b = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {16}});

    auto sc_c   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05}});
    auto z_pt_c = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {100}});

    auto scale_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_a);

    auto z_pt_a_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_a);

    auto fp_a =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), a, scale_a_bcast, z_pt_a_bcast);

    auto scale_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_b);

    auto z_pt_b_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_b);

    auto fp_b =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), b, scale_b_bcast, z_pt_b_bcast);

    auto fp_c = mm->add_instruction(migraphx::make_op("mul"), fp_a, fp_b);

    auto scale_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), sc_c);

    auto z_pt_c_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {64}}}), z_pt_c);

    auto c =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_c, scale_c_bcast, z_pt_c_bcast);

    mm->add_return({c});

    auto prog = read_onnx("qlinearmul_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
