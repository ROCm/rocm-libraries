/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <iostream>
#include <vector>
#include <hip/hip_runtime_api.h>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <test.hpp>
#include <basic_ops.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/make_op.hpp>

#define MIGRAPHX_HIP_ASSERT(x) (EXPECT(x == hipSuccess))

TEST_CASE(host_same_buffer_copy)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4, 2}};
    auto a           = mm->add_parameter("a", ss);
    auto b           = mm->add_parameter("b", ss);
    auto aa          = mm->add_instruction(migraphx::make_op("add"), a, a);
    auto gpu_out     = mm->add_instruction(migraphx::make_op("hip::copy_from_gpu"), aa);
    auto stream_sync = mm->add_instruction(migraphx::make_op("hip::sync_stream"), gpu_out);
    auto pass        = mm->add_instruction(unary_pass_op{}, stream_sync);
    auto alloc       = mm->add_instruction(
        migraphx::make_op("hip::allocate", {{"shape", migraphx::to_value(ss)}}));
    auto gpu_in = mm->add_instruction(migraphx::make_op("hip::copy_to_gpu"), pass, alloc);
    auto aab    = mm->add_instruction(migraphx::make_op("add"), gpu_in, b);
    mm->add_return({aab});
    migraphx::parameter_map pp;
    std::vector<float> a_vec(ss.elements(), -1);
    std::vector<float> b_vec(ss.elements(), 2);
    pp["a"] = migraphx::argument(ss, a_vec.data());
    pp["b"] = migraphx::argument(ss, b_vec.data());
    std::vector<float> gpu_result;
    migraphx::target gpu_t = migraphx::make_target("gpu");
    migraphx::compile_options options;
    options.offload_copy = true;
    p.compile(gpu_t, options);
    auto result = p.eval(pp).back();
    std::vector<float> results_vector(ss.elements(), -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold_vec(ss.elements(), 0);
    EXPECT(migraphx::verify::verify_rms_range(results_vector, gold_vec));
}

TEST_CASE(arguments_lifetime)
{
    auto use_on_gpu = [](const migraphx::argument& arg, int c) {
        auto* arg_ptr = arg.data();
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        MIGRAPHX_HIP_ASSERT(hipMemset(arg_ptr, c, arg.get_shape().bytes()));
        MIGRAPHX_HIP_ASSERT(hipDeviceSynchronize());
        return;
    };

    auto f = [use_on_gpu](const migraphx::argument& input) {
        auto a = migraphx::gpu::register_on_gpu(input);
        auto s = a.get_shape();
        {
            auto b = migraphx::gpu::register_on_gpu(input);
            use_on_gpu(b, 0);
            std::vector<float> expected_b(s.elements(), 0);
            auto gold = migraphx::argument(s, expected_b.data());
        }
        use_on_gpu(a, 1);
        return true;
    };

    migraphx::shape ss{migraphx::shape::float_type, {4, 2}};
    std::vector<float> x_data(ss.elements(), -1);
    migraphx::argument x{ss, x_data.data()};
    EXPECT(f(x));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
