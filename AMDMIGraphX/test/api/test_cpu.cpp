/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/file_buffer.hpp>
#include "test.hpp"

TEST_CASE(load_and_run)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("ref"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(load_and_run_init_list)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("ref"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    auto param_shapes = p.get_parameter_shapes();
    EXPECT(param_shapes.size() == 3);
    auto names   = param_shapes.names();
    auto outputs = p.eval({{names[0], migraphx::argument::generate(param_shapes[names[0]])},
                           {names[1], migraphx::argument::generate(param_shapes[names[1]])},
                           {names[2], migraphx::argument::generate(param_shapes[names[2]])}});
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(quantize_fp16)
{
    auto p1        = migraphx::parse_onnx("gemm_test.onnx");
    const auto& p2 = p1;
    const auto& p3 = p1;
    migraphx::quantize_fp16(p1);

    migraphx::quantize_op_names names;
    migraphx::quantize_fp16(p2, names);
    CHECK(bool{p1 == p2});

    names.add("dot");
    migraphx::quantize_fp16(p3, names);
    CHECK(bool{p1 == p3});
}

TEST_CASE(quantize_int8)
{
    auto p1        = migraphx::parse_onnx("gemm_test.onnx");
    const auto& p2 = p1;
    auto t         = migraphx::target("ref");
    migraphx::quantize_int8_options options;
    migraphx::quantize_int8(p1, t, options);

    migraphx::program_parameters pp;
    auto param_shapes = p1.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    options.add_calibration_data(pp);
    options.add_op_name("dot");

    migraphx::quantize_int8(p2, t, options);
    CHECK(bool{p1 == p2});
}

TEST_CASE(quantize_fp8)
{
    auto p1        = migraphx::parse_onnx("gemm_test.onnx");
    const auto& p2 = p1;
    auto t         = migraphx::target("ref");
    migraphx::quantize_fp8_options options;
    migraphx::quantize_fp8(p1, t, options);

    migraphx::program_parameters pp;
    auto param_shapes = p1.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    options.add_calibration_data(pp);

    migraphx::quantize_fp8(p2, t, options);
    CHECK(bool{p1 == p2});
}

TEST_CASE(load_and_run_user_input_shape)
{
    migraphx::onnx_options options;
    options.set_input_parameter_shape("0", {2, 3, 64, 64});
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx", options);
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("ref"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(zero_parameter)
{
    auto p             = migraphx::parse_onnx("constant_fill_test.onnx");
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("ref"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(set_scalar_parameter)
{
    auto p1 = migraphx::parse_onnx("implicit_add_bcast_test.onnx");
    migraphx::shape s1(migraphx_shape_float_type, {3, 4, 1});
    auto param_shapes = p1.get_parameter_shapes();
    auto s1_orig      = param_shapes["1"];
    CHECK(bool{s1 == s1_orig});

    migraphx::onnx_options option;
    option.set_input_parameter_shape("1", {});
    auto p2 = migraphx::parse_onnx("implicit_add_bcast_test.onnx", option);
    migraphx::shape s_scalar(migraphx_shape_float_type);
    auto param_shapes_1 = p2.get_parameter_shapes();
    auto s_scalar_after = param_shapes_1["1"];
    CHECK(bool{s_scalar == s_scalar_after});
}

TEST_CASE(scalar_shape)
{
    auto s = migraphx::shape(migraphx_shape_float_type);
    EXPECT(s.lengths().size() == 1);
    EXPECT(s.strides().size() == 1);
    EXPECT(s.lengths().front() == 1);
    EXPECT(s.strides().front() == 0);
}

TEST_CASE(strided_shape)
{
    std::vector<std::size_t> lens    = {2, 2};
    std::vector<std::size_t> strides = {1, 2};
    auto s                           = migraphx::shape(migraphx_shape_float_type, lens, strides);
    EXPECT(s.lengths() == lens);
    EXPECT(s.strides() == strides);
}

TEST_CASE(get_main_module)
{
    auto p              = migraphx::parse_onnx("constant_fill_test.onnx");
    migraphx::module mm = p.get_main_module();
    mm.print();
    p.print();
}

TEST_CASE(set_loop_default_iter_num)
{
    migraphx::onnx_options option;
    option.set_default_loop_iterations(15);
    auto p                             = migraphx::parse_onnx("loop_default_test.onnx", option);
    auto out_shapes                    = p.get_output_shapes();
    std::vector<std::size_t> out_lens0 = {1};
    EXPECT(out_shapes[0].lengths() == out_lens0);
    std::vector<std::size_t> out_lens1 = {15, 1};
    EXPECT(out_shapes[1].lengths() == out_lens1);
}

TEST_CASE(set_loop_limit_iterations)
{
    migraphx::onnx_options option;
    option.set_default_loop_iterations(15);
    option.set_limit_loop_iterations(10);
    auto p                             = migraphx::parse_onnx("loop_default_test.onnx", option);
    auto out_shapes                    = p.get_output_shapes();
    std::vector<std::size_t> out_lens0 = {1};
    EXPECT(out_shapes[0].lengths() == out_lens0);
    std::vector<std::size_t> out_lens1 = {10, 1};
    EXPECT(out_shapes[1].lengths() == out_lens1);
}

TEST_CASE(set_loop_limit_iterations2)
{
    migraphx::onnx_options option;
    option.set_limit_loop_iterations(10);
    auto p          = migraphx::parse_onnx("loop_test_implicit_tripcnt.onnx", option);
    auto out_shapes = p.get_output_shapes();
    std::vector<std::size_t> out_lens0 = {1};
    EXPECT(out_shapes[0].lengths() == out_lens0);
    std::vector<std::size_t> out_lens1 = {10, 1};
    EXPECT(out_shapes[1].lengths() == out_lens1);
}

TEST_CASE(set_external_data_path)
{
    migraphx::onnx_options options;
    std::string model_path = "ext_path/external_data_test.onnx";
    auto onnx_buffer       = migraphx::read_string(model_path);
    options.set_external_data_path(migraphx::fs::path(model_path).parent_path().string());
    auto p             = migraphx::parse_onnx_buffer(onnx_buffer, options);
    auto shapes_before = p.get_output_shapes();
    p.compile(migraphx::target("ref"));
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto outputs = p.eval(pp);
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
