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

TEST_CASE(qlinearconcat_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto sc_y   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.5}});
    auto z_pt_y = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {2}});

    auto sc_0   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.5}});
    auto z_pt_0 = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {1}});

    auto sc_1   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.25}});
    auto z_pt_1 = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {0}});

    auto t0 = mm->add_parameter("t0", {migraphx::shape::int8_type, {2}});
    auto t1 = mm->add_parameter("t1", {migraphx::shape::int8_type, {3}});

    auto scale_0_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), sc_0);

    auto z_pt_0_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2}}}), z_pt_0);

    auto fp_0 =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), t0, scale_0_bcast, z_pt_0_bcast);

    auto scale_1_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), sc_1);

    auto z_pt_1_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3}}}), z_pt_1);

    auto fp_1 =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), t1, scale_1_bcast, z_pt_1_bcast);

    auto fp_y = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), fp_0, fp_1);

    auto scale_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), sc_y);

    auto z_pt_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {5}}}), z_pt_y);

    auto y =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_y, scale_y_bcast, z_pt_y_bcast);

    mm->add_return({y});

    auto prog = read_onnx("qlinearconcat_test.onnx");

    EXPECT(p == prog);
}
