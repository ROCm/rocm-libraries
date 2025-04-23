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

TEST_CASE(dynamicquantizelinear_2d_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto x_dims = {3, 4};
    auto x_type = migraphx::shape::float_type;
    auto x      = mm->add_parameter("x", {x_type, x_dims});

    auto l0 = mm->add_literal({0.f});

    std::vector<size_t> axes(x->get_shape().lens().size());
    std::iota(axes.begin(), axes.end(), 0);

    auto reduce_max_x = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), x);
    auto max_x        = add_common_op(*mm, migraphx::make_op("max"), {l0, reduce_max_x});

    auto reduce_min_x = mm->add_instruction(migraphx::make_op("reduce_min", {{"axes", axes}}), x);
    auto min_x        = add_common_op(*mm, migraphx::make_op("min"), {l0, reduce_min_x});

    auto q_range = mm->add_literal(migraphx::literal{
        migraphx::shape{x_type, max_x->get_shape().lens()},
        {std::numeric_limits<uint8_t>::max() - std::numeric_limits<uint8_t>::min()}});
    auto q_min   = mm->add_literal(migraphx::literal{
        migraphx::shape{x_type, max_x->get_shape().lens()}, {std::numeric_limits<uint8_t>::min()}});
    auto q_max   = mm->add_literal(migraphx::literal{
        migraphx::shape{x_type, min_x->get_shape().lens()}, {std::numeric_limits<uint8_t>::max()}});

    auto sub0    = mm->add_instruction(migraphx::make_op("sub"), max_x, min_x);
    auto y_scale = mm->add_instruction(migraphx::make_op("div"), sub0, q_range);

    auto div1      = add_common_op(*mm, migraphx::make_op("div"), {min_x, y_scale});
    auto interm_zp = add_common_op(*mm, migraphx::make_op("sub"), {q_min, div1});

    auto saturate     = mm->add_instruction(migraphx::make_op("clip"), interm_zp, q_min, q_max);
    auto round        = mm->add_instruction(migraphx::make_op("nearbyint"), saturate);
    auto y_zero_point = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::uint8_type}}), round);

    auto scale_y_bcast =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_dims}}), y_scale);

    auto y_pt_c_bcast = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", x_dims}}), y_zero_point);

    mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale_y_bcast, y_pt_c_bcast);

    auto prog = optimize_onnx("dynamicquantizelinear_2d_test.onnx");
    EXPECT(p == prog);
}
