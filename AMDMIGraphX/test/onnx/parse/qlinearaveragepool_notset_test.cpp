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
#include <migraphx/op/pooling.hpp>

TEST_CASE(qlinearaveragepool_notset_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto sc_x   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.5}});
    auto z_pt_x = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {0}});

    auto sc_y   = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.5}});
    auto z_pt_y = mm->add_literal(migraphx::literal{migraphx::shape::int8_type, {10}});

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::int8_type, {1, 1, 5, 5}});

    auto scale_x_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 5, 5}}}), sc_x);

    auto z_pt_x_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 5, 5}}}), z_pt_x);

    auto fp_x =
        mm->add_instruction(migraphx::make_op("dequantizelinear"), x, scale_x_bcast, z_pt_x_bcast);

    auto fp_y =
        mm->add_instruction(migraphx::make_op("pooling",
                                              {{"mode", migraphx::op::pooling_mode::average},
                                               {"padding", {2, 2, 2, 2}},
                                               {"stride", {2, 2}},
                                               {"lengths", {6, 6}}}),
                            fp_x);

    fp_y = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {2, 2}}}), fp_y);

    auto scale_y_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1}}}), sc_y);

    auto z_pt_y_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 1, 1}}}), z_pt_y);

    auto y =
        mm->add_instruction(migraphx::make_op("quantizelinear"), fp_y, scale_y_bcast, z_pt_y_bcast);

    mm->add_return({y});
    auto prog = read_onnx("qlinearaveragepool_notset_test.onnx");

    EXPECT(p == prog);
}
