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
#include <algorithm>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
#include <numeric>
#include <stdexcept>

#define MIGRAPHX_ROCBLAS_ASSERT(x) (assert((x) == rocblas_status::rocblas_status_success))
#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))

rocblas_handle create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    MIGRAPHX_ROCBLAS_ASSERT(rocblas_create_handle(&handle));
    return rocblas_handle{handle};
}

rocblas_handle create_rocblas_handle_ptr(migraphx::context& ctx)
{
    MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
    rocblas_handle rb = create_rocblas_handle_ptr();
    auto* stream      = ctx.get_queue<hipStream_t>();
    MIGRAPHX_ROCBLAS_ASSERT(rocblas_set_stream(rb, stream));
    return rb;
}

struct sscal_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "sscal_custom_op"; }

    // flag to identify whether custom op runs on the GPU or on the host.
    // Based on this flag MIGraphX would inject necessary copies to and from GPU for the input and
    // output buffers as necessary. Therefore if custom_op runs on GPU then it can assume its input
    // buffers are in GPU memory, and similarly for the host
    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument compute(migraphx::context ctx,
                                       migraphx::shape output_shape,
                                       migraphx::arguments args) const override
    {
        // create rocblas stream handle
        auto rb_handle = create_rocblas_handle_ptr(ctx);
        MIGRAPHX_ROCBLAS_ASSERT(rocblas_set_pointer_mode(rb_handle, rocblas_pointer_mode_device));
        rocblas_int n  = args[1].get_shape().lengths()[0];
        float* alpha   = reinterpret_cast<float*>(args[0].data());
        float* vec_ptr = reinterpret_cast<float*>(args[1].data());
        MIGRAPHX_ROCBLAS_ASSERT(rocblas_sscal(rb_handle, n, alpha, vec_ptr, 1));
        MIGRAPHX_ROCBLAS_ASSERT(rocblas_destroy_handle(rb_handle));
        return args[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("sscal_custom_op must have 2 input arguments");
        }
        if(inputs[0].lengths().size() != 1 or inputs[0].lengths()[0] != 1)
        {
            throw std::runtime_error("first input argument to sscal_custom_op must be a scalar");
        }
        if(inputs[1].lengths().size() != 1)
        {
            throw std::runtime_error(
                "second input argument to sscal_custom_op must be a vector with dimension one");
        }
        return inputs.back();
    }
};

int main(int argc, const char* argv[])
{
    // computes ReLU(neg(x) * scale)
    sscal_custom_op sscal_op;
    migraphx::register_experimental_custom_op(sscal_op);
    migraphx::program p;
    migraphx::shape x_shape{migraphx_shape_float_type, {8192}};
    migraphx::shape scale_shape{migraphx_shape_float_type, {1}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", x_shape);
    auto scale         = m.add_parameter("scale", scale_shape);
    auto neg_ins       = m.add_instruction(migraphx::operation("neg"), {x});
    auto custom_kernel =
        m.add_instruction(migraphx::operation("sscal_custom_op"), {scale, neg_ins});
    auto relu_ins = m.add_instruction(migraphx::operation("relu"), {custom_kernel});
    m.add_return({relu_ins});

    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters pp;
    std::vector<float> x_data(x_shape.elements());
    std::vector<float> scale_data{-1};
    std::iota(x_data.begin(), x_data.end(), 0);
    pp.add("x", migraphx::argument(x_shape, x_data.data()));
    pp.add("scale", migraphx::argument(scale_shape, scale_data.data()));
    auto results                       = p.eval(pp);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    if(bool{result == migraphx::argument(x_shape, expected_result.data())})
    {
        std::cout << "Successfully executed custom rocBLAS kernel example\n";
    }
    else
    {
        std::cout << "Custom rocBLAS kernel example failed\n";
    }
    return 0;
}
