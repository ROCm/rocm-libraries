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

TEST_CASE(nms_dynamic_boxes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sb{migraphx::shape::float_type, {{1, 1}, {6, 20}, {4, 4}}};
    auto b = mm->add_parameter("boxes", sb);
    migraphx::shape ss{migraphx::shape::float_type, {{1, 1}, {1, 1}, {6, 20}}};
    auto s = mm->add_parameter("scores", ss);
    migraphx::shape smo{migraphx::shape::int64_type, {1}};
    auto mo = mm->add_parameter("max_output_boxes_per_class", smo);
    migraphx::shape siou{migraphx::shape::float_type, {1}};
    auto iou = mm->add_parameter("iou_threshold", siou);
    migraphx::shape sst{migraphx::shape::float_type, {1}};
    auto st  = mm->add_parameter("score_threshold", sst);
    auto ret = mm->add_instruction(
        migraphx::make_op("nonmaxsuppression", {{"use_dyn_output", true}}), b, s, mo, iou, st);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {6, 20};
    options.use_dyn_output        = true;

    auto prog = read_onnx("nms_dynamic_boxes_test.onnx", options);
    EXPECT(p == prog);
}
