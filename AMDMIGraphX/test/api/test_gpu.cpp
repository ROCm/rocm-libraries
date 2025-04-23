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
#include <numeric>
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <migraphx/manage_ptr.hpp>
#include "test.hpp"

TEST_CASE(load_and_run)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
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

using hip_ptr    = MIGRAPHX_MANAGE_PTR(void, hipFree);
using stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

static stream_ptr get_stream()
{
    hipStream_t stream;
    auto err = hipStreamCreateWithFlags(&stream, 0);
    EXPECT(err == hipSuccess);
    return stream_ptr{stream};
}

static hip_ptr get_hip_buffer(size_t size)
{
    void* ptr;
    auto err = hipMalloc(&ptr, size);
    EXPECT(err == hipSuccess);
    return hip_ptr{ptr};
}

// TODO: placeholder until we have a way to copy tuple arguments to/from device through c++ api
// TEST_CASE(dynamic_batch_load_and_run)
//{
//    migraphx::onnx_options o_options;
//    migraphx::dynamic_dimensions dyn_dims = {{1, 4, {2, 4}}, {3, 3}, {4, 4}, {4, 4}};
//    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
//    dyn_dims = {{2, 2}, {3, 3}, {3, 3}, {3, 3}};
//    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
//    auto p = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
//    migraphx::compile_options c_options;
//    c_options.set_split_single_dyn_dim();
//    p.compile(migraphx::target("gpu"), c_options);
//    auto out_shapes = p.get_output_shapes();
//    CHECK(out_shapes.size() == 1);
//    EXPECT(out_shapes[0].dynamic());
//
//    std::vector<float> a(0.12, 2*3*4*4);
//    std::vector<float> c(0.75, 2*3*3*3);
//
//    auto param_shapes = p.get_parameter_shapes();
//    int batch_size    = 2;
//    std::unordered_map<std::string, migraphx::argument> arg_map;
//
//    arg_map["0"] = migraphx::argument(param_shapes["0"].to_static(batch_size), a.data());
//    arg_map["1"] = migraphx::argument(param_shapes["1"].to_static(batch_size), c.data());
//
//    migraphx::program_parameters pp;
//    std::vector<hip_ptr> buffs;
//    std::vector<migraphx::argument> args;
//
//    // copy to GPU and create parameter map
//    for(auto&& name : param_shapes.names())
//    {
//        if(arg_map.find(name) != arg_map.end())
//        {
//            args.push_back(arg_map.at(name));
//        }
//        else
//        {
//            migraphx::shape static_shape = param_shapes[name].to_static(batch_size);
//            auto output_arg              = migraphx::argument(static_shape);
//            args.push_back(output_arg);
//        }
//        buffs.push_back(get_hip_buffer(args.rbegin()->get_shape().bytes()));
//        auto err = hipMemcpy(buffs.rbegin()->get(),
//                             args.rbegin()->data(),
//                             args.rbegin()->get_shape().bytes(),
//                             hipMemcpyHostToDevice);
//        EXPECT(err == hipSuccess);
//        pp.add(name, migraphx::argument(args.rbegin()->get_shape(), buffs.rbegin()->get()));
//    }
//
//    auto output = p.eval(pp)[0];
//
//    // copy output back to host
//    auto host_arg = migraphx::argument(output.get_shape());
//    auto err      = hipMemcpy(
//        host_arg.data(), output.data(), output.get_shape().bytes(), hipMemcpyDeviceToHost);
//    EXPECT(err == hipSuccess);
//}

TEST_CASE(dynamic_batch_load_and_run_offload)
{
    migraphx::onnx_options o_options;
    migraphx::dynamic_dimensions dyn_dims = {migraphx::dynamic_dimension{1, 4, {2, 4}},
                                             migraphx::dynamic_dimension{3, 3},
                                             migraphx::dynamic_dimension{4, 4},
                                             migraphx::dynamic_dimension{4, 4}};
    o_options.set_dyn_input_parameter_shape("0", dyn_dims);
    dyn_dims = {migraphx::dynamic_dimension{2, 2},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3},
                migraphx::dynamic_dimension{3, 3}};
    o_options.set_dyn_input_parameter_shape("1", dyn_dims);
    auto p             = migraphx::parse_onnx("conv_dynamic_batch_test.onnx", o_options);
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options c_options;
    c_options.set_offload_copy();
    p.compile(migraphx::target("gpu"), c_options);
    auto out_shapes = p.get_output_shapes();
    EXPECT(out_shapes.size() == 1);
    EXPECT(out_shapes[0].dynamic());

    // batch size = 2
    std::vector<float> a(2 * 3 * 4 * 4, 0.12);
    std::vector<float> c(2 * 3 * 3 * 3, 0.75);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    pp.add("0",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 4, 4}), a.data()));
    pp.add("1",
           migraphx::argument(migraphx::shape(migraphx_shape_float_type, {2, 3, 3, 3}), c.data()));
    auto outputs = p.eval(pp);

    EXPECT(shapes_before.size() == outputs.size());
    EXPECT(bool{outputs.front().get_shape() ==
                migraphx::shape(migraphx_shape_float_type, {2, 2, 2, 2})});
}

TEST_CASE(load_and_run_async)
{
    auto p             = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    auto shapes_before = p.get_output_shapes();
    migraphx::compile_options options;
    options.set_offload_copy(false);
    p.compile(migraphx::target("gpu"), options);
    auto shapes_after = p.get_output_shapes();
    CHECK(shapes_before.size() == 1);
    CHECK(shapes_before.size() == shapes_after.size());
    CHECK(bool{shapes_before.front() == shapes_after.front()});
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();

    stream_ptr stream = get_stream();

    std::vector<hip_ptr> buffs;
    std::vector<migraphx::argument> args;
    for(auto&& name : param_shapes.names())
    {
        args.push_back(migraphx::argument::generate(param_shapes[name]));
        buffs.push_back(get_hip_buffer(args.rbegin()->get_shape().bytes()));

        auto err = hipMemcpy(buffs.rbegin()->get(),
                             args.rbegin()->data(),
                             args.rbegin()->get_shape().bytes(),
                             hipMemcpyHostToDevice);
        EXPECT(err == hipSuccess);
        pp.add(name, migraphx::argument(args.rbegin()->get_shape(), buffs.rbegin()->get()));
    }

    auto outputs = p.run_async(pp, stream.get());
    CHECK(shapes_before.size() == outputs.size());
    CHECK(bool{shapes_before.front() == outputs.front().get_shape()});
}

TEST_CASE(load_and_run_ctx)
{
    auto p = migraphx::parse_onnx("conv_relu_maxpool_test.onnx");
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    auto param_shapes = p.get_parameter_shapes();
    for(auto&& name : param_shapes.names())
    {
        pp.add(name, migraphx::argument::generate(param_shapes[name]));
    }
    auto ctx = p.experimental_get_context();
    EXPECT(ctx.get_queue<hipStream_t>() != nullptr);
    p.eval(pp);
    ctx.finish();
}

TEST_CASE(if_pl_test)
{
    auto run_prog = [&](auto cond) {
        auto p             = migraphx::parse_onnx("if_pl_test.onnx");
        auto shapes_before = p.get_output_shapes();
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        auto shapes_after = p.get_output_shapes();
        CHECK(shapes_before.size() == 1);
        CHECK(bool{shapes_before.front() == shapes_after.front()});

        migraphx::program_parameters pp;
        auto param_shapes = p.get_parameter_shapes();
        auto xs           = param_shapes["x"];
        std::vector<float> xd(xs.elements(), 1.0);
        pp.add("x", migraphx::argument(xs, xd.data()));
        auto ys = param_shapes["y"];
        std::vector<float> yd(ys.elements(), 2.0);
        pp.add("y", migraphx::argument(ys, yd.data()));
        char ccond = cond;
        pp.add("cond", migraphx::argument(param_shapes["cond"], &ccond));

        auto outputs = p.eval(pp);
        auto output  = outputs[0];
        return output.as_vector<float>();
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {2, 3, 4, 5, 6, 7};
        EXPECT(result_vector == gold);
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {1, 2, 3, 4, 5, 6};
        EXPECT(result_vector == gold);
    }
}

TEST_CASE(loop_test)
{
    auto run_prog = [&](int64_t max_iter_num) {
        migraphx::onnx_options parse_options;
        parse_options.set_default_loop_iterations(max_iter_num);
        auto p             = migraphx::parse_onnx("loop_default_test.onnx", parse_options);
        auto shapes_before = p.get_output_shapes();
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        auto shapes_after = p.get_output_shapes();
        CHECK(shapes_before.size() == 2);
        CHECK(bool{shapes_before.front() == shapes_after.front()});

        migraphx::program_parameters pp;
        auto param_shapes     = p.get_parameter_shapes();
        auto aas              = param_shapes["a"];
        std::vector<float> xd = {1.0f};
        pp.add("a", migraphx::argument(aas, xd.data()));
        auto bbs              = param_shapes["b"];
        std::vector<float> yd = {2.0};
        pp.add("b", migraphx::argument(bbs, yd.data()));

        auto outputs = p.eval(pp);
        auto output  = outputs[0];
        std::vector<std::vector<float>> ret;
        ret.push_back(output.as_vector<float>());

        output = outputs[1];
        ret.push_back(output.as_vector<float>());

        return ret;
    };

    {
        auto result_vector       = run_prog(10);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }

    {
        auto result_vector       = run_prog(15);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }
}

TEST_CASE(loop_test_limit_max_iter)
{
    auto run_prog = [&](int64_t limit_max_iterations) {
        migraphx::onnx_options parse_options;
        parse_options.set_limit_loop_iterations(limit_max_iterations);
        auto p             = migraphx::parse_onnx("loop_test_implicit_tripcnt.onnx", parse_options);
        auto shapes_before = p.get_output_shapes();
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        auto shapes_after = p.get_output_shapes();
        CHECK(shapes_before.size() == 2);
        CHECK(bool{shapes_before.front() == shapes_after.front()});

        migraphx::program_parameters pp;
        auto param_shapes     = p.get_parameter_shapes();
        auto aas              = param_shapes["a"];
        std::vector<float> xd = {1.0f};
        pp.add("a", migraphx::argument(aas, xd.data()));
        auto bbs              = param_shapes["b"];
        std::vector<float> yd = {2.0};
        pp.add("b", migraphx::argument(bbs, yd.data()));

        auto cs   = param_shapes["keep_going_cond"];
        bool cond = true;
        pp.add("keep_going_cond", migraphx::argument(cs, &cond));

        auto outputs = p.eval(pp);
        auto output  = outputs[0];
        std::vector<std::vector<float>> ret;
        ret.push_back(output.as_vector<float>());

        output = outputs[1];
        ret.push_back(output.as_vector<float>());

        return ret;
    };

    {
        auto result_vector       = run_prog(5);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }

    {
        auto result_vector       = run_prog(20);
        std::vector<float> gold0 = {2.0f};
        EXPECT(result_vector.at(0) == gold0);
        std::vector<float> gold1 = {-2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        EXPECT(result_vector.at(1) == gold1);
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
