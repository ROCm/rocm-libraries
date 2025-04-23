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

TEST_CASE(onehot_static_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // depth literal that is parsed but not used
    mm->add_literal(migraphx::literal(migraphx::shape{migraphx::shape::int32_type, {1}, {1}}, {3}));
    auto indices =
        mm->add_parameter("indices", migraphx::shape{migraphx::shape::int32_type, {5, 2}});
    auto values = mm->add_parameter("values", migraphx::shape{migraphx::shape::half_type, {2}});
    auto ret    = mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", 0}, {"depth", 3}}), indices, values);
    mm->add_return({ret});

    auto prog = read_onnx("onehot_static_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(onehot_dyn_test0)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    // depth literal that is parsed but not used
    mm->add_literal(migraphx::literal(migraphx::shape{migraphx::shape::int32_type, {1}, {1}}, {3}));
    auto indices = mm->add_parameter(
        "indices", migraphx::shape{migraphx::shape::int32_type, {{2, 10}, {2, 2}}});
    auto values = mm->add_parameter("values", migraphx::shape{migraphx::shape::half_type, {2}});
    auto ret    = mm->add_instruction(
        migraphx::make_op("onehot", {{"axis", -1}, {"depth", 3}}), indices, values);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 10};
    auto prog                     = read_onnx("onehot_dyn_test0.onnx", options);
    EXPECT(p == prog);
}

TEST_CASE(onehot_dyn_test1)
{
    migraphx::program p;
    auto* mm     = p.get_main_module();
    auto indices = mm->add_parameter(
        "indices", migraphx::shape{migraphx::shape::int32_type, {{2, 10}, {2, 2}}});
    auto values = mm->add_parameter("values", migraphx::shape{migraphx::shape::float_type, {2}});
    auto depth  = mm->add_parameter("depth", migraphx::shape{migraphx::shape::int64_type, {1}});
    auto ret =
        mm->add_instruction(migraphx::make_op("onehot", {{"axis", 1}}), indices, depth, values);
    mm->add_return({ret});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 10};
    auto prog                     = read_onnx("onehot_dyn_test1.onnx", options);
    EXPECT(p == prog);
}
