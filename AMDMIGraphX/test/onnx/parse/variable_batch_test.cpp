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

TEST_CASE(variable_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {1, 3, 16, 16}});
    mm->add_instruction(migraphx::make_op("identity"), l0);
    auto prog = optimize_onnx("variable_batch_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 2};

    auto prog = read_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{2, 5}, {3, 3}, {16, 16}, {16, 16}}});
    auto r = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dyn_dim_value = {2, 5};

    auto prog = read_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test3)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{2, 5}, {3, 3}, {16, 16}, {16, 16}}});
    auto r = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 5}, {3, 3}, {16, 16}, {16, 16}};

    auto prog = read_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test4)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.default_dim_value = 2;

    auto prog = read_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}

TEST_CASE(variable_batch_user_input_test5)
{
    // Error using default_dim_value and default_dyn_dim_value
    migraphx::onnx_options options;
    options.default_dim_value     = 2;
    options.default_dyn_dim_value = {1, 2};

    EXPECT(test::throws([&] { read_onnx("variable_batch_test.onnx", options); }));
}

TEST_CASE(variable_batch_user_input_test6)
{
    // Error using both map_dyn_input_dims and map_input_dims
    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 5}, {3, 3}, {16, 16}, {16, 16}};
    options.map_input_dims["0"]     = {2, 3, 16, 16};

    EXPECT(test::throws([&] { read_onnx("variable_batch_test.onnx", options); }));
}

TEST_CASE(variable_batch_user_input_test7)
{
    // if entry in map_dyn_input_dims is all fixed dynamic_dimensions, convert it to a static
    // shape
    migraphx::program p;
    auto* mm = p.get_main_module();
    auto l0  = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {2, 3, 16, 16}});
    auto r   = mm->add_instruction(migraphx::make_op("identity"), l0);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["0"] = {{2, 2, {2}}, {3, 3}, {16, 16}, {16, 16}};

    auto prog = read_onnx("variable_batch_test.onnx", options);

    EXPECT(p == prog);
}
