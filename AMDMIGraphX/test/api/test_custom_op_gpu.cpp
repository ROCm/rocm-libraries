/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <hip/hip_runtime_api.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include <numeric>
#include <stdexcept>
#include "test.hpp"

#define MIGRAPHX_HIP_ASSERT(x) (EXPECT(x == hipSuccess))

struct half_copy_host final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "half_copy_host"; }

    virtual bool runs_on_offload_target() const override { return false; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // This custom op simply sets first half size_bytes of the input to 0, and rest of the half
        // bytes are copied. for this custom_op, it does its computation on the host. Therefore,
        // `runs_on_offload_target()` is set to false. MIGraphX would inject necessary buffer copies
        // to and from GPU to Host based on `runs_on_offload_targe()` flag for input buffers as well
        // as the output buffers
        auto* input_buffer_ptr  = inputs[0].data();
        auto* output_buffer_ptr = inputs[1].data();
        auto input_bytes        = inputs[0].get_shape().bytes();
        auto copy_bytes         = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(output_buffer_ptr,
                                           input_buffer_ptr,
                                           input_bytes,
                                           hipMemcpyHostToHost,
                                           ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(
            hipMemsetAsync(output_buffer_ptr, 0, copy_bytes, ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        return inputs[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(not inputs[0].standard() or not inputs[1].standard())
        {
            throw std::runtime_error("Input args must be standard shaped");
        }
        if(inputs.size() != 2)
        {
            throw std::runtime_error("number of inputs must be 2");
        }
        return inputs.back();
    }
};

struct half_copy_device final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "half_copy_device"; }

    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // This custom op simply sets first half size_bytes of the input to 0, and rest of the half
        // bytes are copied. for this custom_op, it does its computation on the "GPU". Therefore,
        // `runs_on_offload_target()` is set to "true".
        auto* input_buffer_ptr  = inputs[0].data();
        auto* output_buffer_ptr = inputs[1].data();
        auto input_bytes        = inputs[0].get_shape().bytes();
        auto copy_bytes         = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMemcpyAsync(output_buffer_ptr,
                                           input_buffer_ptr,
                                           input_bytes,
                                           hipMemcpyDeviceToDevice,
                                           ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        MIGRAPHX_HIP_ASSERT(
            hipMemsetAsync(output_buffer_ptr, 0, copy_bytes, ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        return inputs[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(not inputs[0].standard() or not inputs[1].standard())
        {
            throw std::runtime_error("Input args must be standard shaped");
        }
        if(inputs.size() != 2)
        {
            throw std::runtime_error("number of inputs must be 2");
        }
        return inputs.back();
    }
};

// overwrites input buffer
struct half_copy_device_same_buffer final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "half_copy_device_same_buffer"; }

    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // This custom op simply sets first half size_bytes of the input 0, and rest of the half
        // bytes are copied. for this custom_op, it does its computation on the "device". Therefore,
        // `runs_on_offload_target()` is set to "true"
        auto* buffer_ptr = inputs[0].data();
        auto input_bytes = inputs[0].get_shape().bytes();
        auto copy_bytes  = input_bytes / 2;
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(
            hipMemsetAsync(buffer_ptr, 0, copy_bytes, ctx.get_queue<hipStream_t>()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        return inputs[0];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(not inputs[0].standard())
        {
            throw std::runtime_error("Input arg must be standard shaped");
        }
        return inputs.front();
    }
};

TEST_CASE(register_half_copy_op)
{
    half_copy_host hch;
    migraphx::register_experimental_custom_op(hch);
    auto op = migraphx::operation("half_copy_host");
    EXPECT(op.name() == "half_copy_host");

    half_copy_device hcd;
    migraphx::register_experimental_custom_op(hcd);
    op = migraphx::operation("half_copy_device");
    EXPECT(op.name() == "half_copy_device");

    half_copy_device_same_buffer hcdsb;
    migraphx::register_experimental_custom_op(hcdsb);
    op = migraphx::operation("half_copy_device_same_buffer");
    EXPECT(op.name() == "half_copy_device_same_buffer");
}

TEST_CASE(half_copy_custom_op_test)
{
    auto run_test_prog = [](const std::string& op_name, bool buffer_alloc) {
        migraphx::program p;
        migraphx::module m = p.get_main_module();
        migraphx::shape s{migraphx_shape_float_type, {4, 3}};
        auto x                        = m.add_parameter("x", s);
        migraphx::instructions inputs = {x};
        if(buffer_alloc)
        {
            auto alloc = m.add_allocation(s);
            inputs     = {x, alloc};
        }
        auto half_copy_ins = m.add_instruction(migraphx::operation(op_name.c_str()), inputs);
        m.add_return({half_copy_ins});
        migraphx::compile_options options;
        options.set_offload_copy();
        p.compile(migraphx::target("gpu"), options);
        migraphx::program_parameters pp;
        std::vector<float> x_data(12);
        std::iota(x_data.begin(), x_data.end(), 0);
        pp.add("x", migraphx::argument(s, x_data.data()));
        auto results    = p.eval(pp);
        auto result     = results[0];
        auto result_vec = result.as_vector<float>();
        std::vector<float> expected_result(12, 0);
        std::iota(expected_result.begin() + 6, expected_result.end(), 6);
        EXPECT(bool{result == migraphx::argument(s, expected_result.data())});
    };

    // register all the ops
    half_copy_host hch;
    migraphx::register_experimental_custom_op(hch);

    half_copy_device hcd;
    migraphx::register_experimental_custom_op(hcd);

    half_copy_device_same_buffer hcdsb;
    migraphx::register_experimental_custom_op(hcdsb);

    std::vector<std::pair<std::string, bool>> tests_config = {
        {"half_copy_host", true},
        {"half_copy_device", true},
        {"half_copy_device_same_buffer", false}};
    for(const auto& i : tests_config)
    {
        run_test_prog(i.first, i.second);
    }
}

struct stride_two final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "stride_two"; }
    virtual migraphx::argument
    compute(migraphx::context, migraphx::shape out_shape, migraphx::arguments inputs) const override
    {
        return {out_shape, inputs[0].data()};
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 1)
        {
            throw std::runtime_error("stride_two op must have only one input argument");
        };
        if(not inputs[0].standard())
        {
            throw std::runtime_error("stride_two op only works on the standard input shapes");
        }
        migraphx::shape input_s  = inputs[0];
        std::vector<size_t> dims = input_s.lengths();
        std::vector<size_t> new_dims;
        std::vector<size_t> strides = input_s.strides();
        std::vector<size_t> new_strides;
        std::for_each(dims.begin(), dims.end(), [&](auto i) { new_dims.push_back(i / 2); });
        std::for_each(
            strides.begin(), strides.end(), [&](auto i) { new_strides.push_back(i * 2); });
        migraphx::shape output_shape{input_s.type(), new_dims, new_strides};
        return output_shape;
    }

    virtual bool runs_on_offload_target() const override { return true; }
    virtual std::vector<size_t> output_alias(migraphx::shapes) const override { return {0}; };
};

TEST_CASE(stride_two_custom_op_test)
{
    stride_two st;
    migraphx::register_experimental_custom_op(st);

    migraphx::program p;
    migraphx::module m = p.get_main_module();
    migraphx::shape s{migraphx_shape_float_type, {4, 4, 4}};
    auto x              = m.add_parameter("x", s);
    auto stride_two_ins = m.add_instruction(migraphx::operation("stride_two"), {x});
    m.add_return({stride_two_ins});
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(64);
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    auto result_vec                    = result.as_vector<float>();
    std::vector<float> expected_result = {0, 2, 8, 10, 32, 34, 40, 42};
    EXPECT(result_vec == expected_result);
}

TEST_CASE(custom_op_with_pre_and_post_subgraph_test)
{
    half_copy_host hco;
    migraphx::register_experimental_custom_op(hco);

    stride_two st;
    migraphx::register_experimental_custom_op(st);

    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {4, 6}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    // pre-subgraph
    auto neg_ins = m.add_instruction(migraphx::operation("neg"), x);
    auto trans_ins =
        m.add_instruction(migraphx::operation("transpose", "{permutation: [1, 0]}"), {neg_ins});
    auto cont_ins = m.add_instruction(migraphx::operation("contiguous"), {trans_ins});
    // custom_op
    migraphx::shape trans_shape{migraphx_shape_float_type, {6, 4}};
    auto alloc = m.add_allocation(trans_shape);
    auto half_copy_ins =
        m.add_instruction(migraphx::operation("half_copy_host"), {cont_ins, alloc});
    // post-subgraph
    auto abs_ins = m.add_instruction(migraphx::operation("abs"), {half_copy_ins});
    // another custom_op
    auto stride_two_ins = m.add_instruction(migraphx::operation("stride_two"), {abs_ins});
    // post-subgraph
    auto relu_ins = m.add_instruction(migraphx::operation("relu"), {stride_two_ins});
    m.add_return({relu_ins});
    migraphx::compile_options options;
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(s.elements());
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    auto result_vec                    = result.as_vector<float>();
    std::vector<float> expected_result = {0, 0, 0, 0, 4, 16};
    EXPECT(bool{result == migraphx::argument(migraphx::shape{migraphx_shape_float_type, {3, 2}},
                                             expected_result.data())});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
