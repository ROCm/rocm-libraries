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

#include <iostream>
#include <vector>
#include <migraphx/register_target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/kernel.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/gpu/target.hpp>
#include "test.hpp"

using hip_stream_ptr = MIGRAPHX_MANAGE_PTR(hipStream_t, hipStreamDestroy);

constexpr uint32_t stream_sync_test_val = 1337;

// NOLINTNEXTLINE
const std::string compare_numbers = R"__migraphx__(
#include <hip/hip_runtime.h>

extern "C" {
__global__ void compare(float* data) 
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (data[i] != 1337) 
    {
        abort();
    }
}
    
}

int main() {}

)__migraphx__";

static migraphx::src_file make_src_file(const std::string& name, const std::string& content)
{
    return {name, content};
}

static hip_stream_ptr get_stream()
{
    hipStream_t stream;

    auto status = hipStreamCreate(&stream);
    if(status != hipSuccess)
    {
        MIGRAPHX_THROW("Failed to get stream");
    }

    return hip_stream_ptr{stream};
}

TEST_CASE(test_stream_sync_compare_kernel)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("check_stuff.cpp", compare_numbers)}, {}, migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::gpu::kernel k1{binaries.front(), "compare"};

    auto input =
        migraphx::fill_argument({migraphx::shape::float_type, {128}}, stream_sync_test_val);

    auto ginput = migraphx::gpu::to_gpu(input);

    hip_stream_ptr pstream = get_stream();

    k1.launch(pstream.get(), input.get_shape().elements(), 1024)(ginput.cast<float>());

    auto output = migraphx::gpu::from_gpu(ginput);
    EXPECT(output == input);
}

TEST_CASE(test_stream_sync)
{
    auto binaries = migraphx::gpu::compile_hip_src(
        {make_src_file("check_stuff.cpp", compare_numbers)}, {}, migraphx::gpu::get_device_name());
    EXPECT(binaries.size() == 1);

    migraphx::gpu::kernel k1{binaries.front(), "compare"};
    const unsigned int m = 128;
    const unsigned int k = 8192;

    // Setup empty GPU memory buffer
    migraphx::shape input_shape{migraphx::shape::float_type, {m, k}};
    migraphx::shape output_shape{migraphx::shape::float_type, {m, m}};
    auto input  = migraphx::fill_argument(input_shape, 0);
    auto ginput = migraphx::gpu::to_gpu(input);

    auto output  = migraphx::fill_argument(output_shape, 0);
    auto goutput = migraphx::gpu::to_gpu(output);

    hip_stream_ptr pstream = get_stream();

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {m, k}});
    auto y = mm->add_literal(
        migraphx::generate_literal(migraphx::shape{migraphx::shape::float_type, {k, m}}));

    std::vector<float> data(m * m, stream_sync_test_val);
    auto test_val = mm->add_literal(output_shape, data);
    auto mult_out = mm->add_instruction(migraphx::make_op("dot"), x, y);
    mm->add_instruction(migraphx::make_op("add"), mult_out, test_val);

    p.compile(migraphx::make_target("gpu"));

    // Run network and then verify with kernel
    auto args = p.eval({{"x", ginput}, {"output", goutput}}, {pstream.get(), true});
    k1.launch(pstream.get(), m * m, 1024)(goutput.cast<float>());

    output = migraphx::gpu::from_gpu(goutput);
    EXPECT(output != input);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
