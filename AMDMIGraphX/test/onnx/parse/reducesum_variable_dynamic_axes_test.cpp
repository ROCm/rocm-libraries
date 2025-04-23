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

TEST_CASE(reducesum_variable_dynamic_axes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto x   = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4, 5, 6}});
    const std::vector<migraphx::shape::dynamic_dimension> axes_dims{{0, 3}};
    auto axes = mm->add_parameter("axes", migraphx::shape{migraphx::shape::int64_type, axes_dims});

    auto reduce_input_axes =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {}}}), x, axes);
    std::vector<int64_t> all_axes(x->get_shape().ndim());
    std::iota(all_axes.begin(), all_axes.end(), 0);
    auto all_axes_lit = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::type_t::int64_type, {all_axes.size()}}, all_axes});
    auto reduce_all_axes =
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {}}}), x, all_axes_lit);

    auto zero_lit =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {0u}});
    auto axes_size = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 1}}), axes);
    auto is_axes_empty = mm->add_instruction(migraphx::make_op("equal"), axes_size, zero_lit);
    auto where         = mm->add_instruction(
        migraphx::make_op("where"), is_axes_empty, reduce_all_axes, reduce_input_axes);
    mm->add_return({where});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["axes"] = axes->get_shape().dyn_dims();
    auto prog = read_onnx("reducesum_variable_dynamic_axes_test.onnx", options);
    EXPECT(p == prog);
}
