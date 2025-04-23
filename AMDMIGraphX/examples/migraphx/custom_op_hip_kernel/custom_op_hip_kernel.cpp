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
#include <algorithm>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
#include <numeric>

#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))
/*
 * Square each element in the array A and write to array C.
 */
template <typename T>
__global__ void vector_square(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

struct square_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "square_custom_op"; }

    // flag to identify whether custom op runs on the GPU or on the host.
    // Based on this flag MIGraphX would inject necessary copies to and from GPU for the input and
    // output buffers as necessary. Therefore if custom_op runs on GPU then it can assume its input
    // buffers are in GPU memory, and similarly for the host
    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument
    compute(migraphx::context ctx, migraphx::shape, migraphx::arguments inputs) const override
    {
        // if compile options has offload_copy = true then, parameters and outputs will be
        // automatically copied to and from GPUs' memory. Here assume that `inputs` arguments are
        // already in the GPU, so no need to do Malloc, Free or Memcpy. Last element in the `inputs`
        // is output argument, so it should be returned from compute method.
        auto* input_buffer  = reinterpret_cast<float*>(inputs[0].data());
        auto* output_buffer = reinterpret_cast<float*>(inputs[1].data());
        size_t n_elements   = inputs[0].get_shape().elements();
        MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
        const unsigned blocks            = 512;
        const unsigned threads_per_block = 256;
        // cppcheck-suppress migraphx-UseDeviceLaunch
        hipLaunchKernelGGL(vector_square,
                           dim3(blocks),
                           dim3(threads_per_block),
                           0,
                           ctx.get_queue<hipStream_t>(),
                           output_buffer,
                           input_buffer,
                           n_elements);
        return inputs[1];
    }
    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("square_custom_op must have 2 arguments");
        }
        if(inputs[0] != inputs[1])
        {
            throw std::runtime_error("Inputs to the square_custom_op must have same Shape");
        }
        return inputs.back();
    }
};

int main(int argc, const char* argv[])
{
    square_custom_op square_op;
    migraphx::register_experimental_custom_op(square_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {32, 256}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto neg_ins       = m.add_instruction(migraphx::operation("neg"), x);
    // add allocation for the custom_kernel's output buffer
    auto alloc = m.add_allocation(s);
    auto custom_kernel =
        m.add_instruction(migraphx::operation("square_custom_op"), {neg_ins, alloc});
    auto relu_ins = m.add_instruction(migraphx::operation("relu"), {custom_kernel});
    m.add_return({relu_ins});
    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(s.elements());
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::pow(i, 2); });
    if(bool{result == migraphx::argument(s, expected_result.data())})
    {
        std::cout << "Successfully executed custom HIP kernel example\n";
    }
    else
    {
        std::cout << "Custom HIP kernel example failed\n";
    }
    return 0;
}
