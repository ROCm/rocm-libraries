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

TEST_CASE(clip_dyn_min_max_test)
{
    migraphx::program p;
    auto* mm                                            = p.get_main_module();
    auto min_val                                        = mm->add_literal(0.0f);
    auto max_val                                        = mm->add_literal(6.0f);
    std::vector<migraphx::shape::dynamic_dimension> dds = {{2, 8, {3}}};
    auto l0 = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, dds});
    auto bl0 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(dds)}}),
                            l0,
                            min_val,
                            max_val);
    min_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(dds)}}), min_val, bl0);
    max_val = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(dds)}}), max_val, bl0);
    auto ret = mm->add_instruction(migraphx::make_op("clip"), bl0, min_val, max_val);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 8, {3}};
    auto prog                     = read_onnx("clip_dyn_min_max_test.onnx", options);

    EXPECT(p == prog);
}
