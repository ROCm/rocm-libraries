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
#include <algorithm>
#include <hip/hip_runtime.h>
#include <migraphx/migraphx.h>
#include <miopen/miopen.h>
#include <migraphx/migraphx.hpp> // MIGraphX's C++ API
#include <numeric>
#include <stdexcept>

#define MIGRAPHX_MIOPEN_ASSERT(x) (assert((x) == miopenStatusSuccess))
#define MIGRAPHX_HIP_ASSERT(x) (assert((x) == hipSuccess))

inline miopenTensorDescriptor_t make_miopen_tensor(const migraphx::shape& s)
{
    miopenTensorDescriptor_t t;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateTensorDescriptor(&t));
    // Convert to ints
    auto s_lens = s.lengths();
    std::vector<int> lens(s_lens.begin(), s_lens.end());
    auto s_strides = s.strides();
    std::vector<int> strides(s_strides.begin(), s_strides.end());
    miopenDataType_t d;
    if(s.type() == migraphx_shape_float_type)
        d = miopenFloat;
    else if(s.type() == migraphx_shape_half_type)
        d = miopenHalf;
    else if(s.type() == migraphx_shape_int32_type)
        d = miopenInt32;
    else if(s.type() == migraphx_shape_int8_type)
        d = miopenInt8;
    else
        throw("MAKE_TENSOR: unsupported type");
    miopenSetTensorDescriptor(t, d, s_lens.size(), lens.data(), strides.data());
    return t;
}

inline auto make_miopen_handle(migraphx::context& ctx)
{
    MIGRAPHX_HIP_ASSERT(hipSetDevice(0));
    auto* stream = ctx.get_queue<hipStream_t>();
    miopenHandle_t out;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateWithStream(&out, stream));
    return out;
}

inline auto make_activation_descriptor(miopenActivationMode_t mode,
                                       double alpha = 0,
                                       double beta  = 0,
                                       double gamma = 0)
{
    miopenActivationDescriptor_t ad;
    MIGRAPHX_MIOPEN_ASSERT(miopenCreateActivationDescriptor(&ad));
    miopenSetActivationDescriptor(ad, mode, alpha, beta, gamma);
    return ad;
}

struct abs_custom_op final : migraphx::experimental_custom_op_base
{
    virtual std::string name() const override { return "abs_custom_op"; }

    // flag to identify whether custom op runs on the GPU or on the host.
    // Based on this flag MIGraphX would inject necessary copies to and from GPU for the input and
    // output buffers as necessary. Therefore if custom_op runs on GPU then it can assume its input
    // buffers are in GPU memory, and similarly for the host
    virtual bool runs_on_offload_target() const override { return true; }

    virtual migraphx::argument compute(migraphx::context ctx,
                                       migraphx::shape output_shape,
                                       migraphx::arguments args) const override
    {
        float alpha = 1;
        float beta  = 0;
        // MIOpen kernel call takes raw buffer pointers for the TensorData. These Buffer pointers
        // must be accompanied with Tensor Description e.g. shape, type, strides, dimensionality.
        // Following `make_miopen_tensor` makes such tensor descriptors to pass as parameter to
        // MIOpen kernel call.
        auto y_desc = make_miopen_tensor(output_shape);
        auto x_desc = make_miopen_tensor(args[0].get_shape());
        // create MIOpen stream handle
        auto miopen_handle = make_miopen_handle(ctx);
        // MIOpen has generic kernel for many different kinds of activation functions.
        // Each such generic call must be accompanied with description of what kind of activation
        // computation to perform
        auto ad = make_activation_descriptor(miopenActivationABS, 0, 0, 0);
        miopenActivationForward(
            miopen_handle, ad, &alpha, x_desc, args[0].data(), &beta, y_desc, args[1].data());
        return args[1];
    }

    virtual migraphx::shape compute_shape(migraphx::shapes inputs) const override
    {
        if(inputs.size() != 2)
        {
            throw std::runtime_error("abs_custom_op must have two input arguments");
        }
        if(inputs[0] != inputs[1])
        {
            throw std::runtime_error("Input arguments to abs_custom_op must have same shape");
        }
        return inputs.back();
    }
};

int main(int argc, const char* argv[])
{
    abs_custom_op abs_op;
    migraphx::register_experimental_custom_op(abs_op);
    migraphx::program p;
    migraphx::shape s{migraphx_shape_float_type, {32, 256}};
    migraphx::module m = p.get_main_module();
    auto x             = m.add_parameter("x", s);
    auto neg_ins       = m.add_instruction(migraphx::operation("neg"), {x});
    // add allocation for the custom_kernel's output buffer
    auto alloc         = m.add_allocation(s);
    auto custom_kernel = m.add_instruction(migraphx::operation("abs_custom_op"), {neg_ins, alloc});
    auto relu_ins      = m.add_instruction(migraphx::operation("relu"), {custom_kernel});
    m.add_return({relu_ins});

    migraphx::compile_options options;
    // set offload copy to true for GPUs
    options.set_offload_copy();
    p.compile(migraphx::target("gpu"), options);
    migraphx::program_parameters prog_params;
    std::vector<float> x_data(s.bytes() / sizeof(s.type()));
    std::iota(x_data.begin(), x_data.end(), 0);
    prog_params.add("x", migraphx::argument(s, x_data.data()));
    auto results                       = p.eval(prog_params);
    auto result                        = results[0];
    std::vector<float> expected_result = x_data;
    std::transform(expected_result.begin(),
                   expected_result.end(),
                   expected_result.begin(),
                   [](auto i) { return std::abs(i); });
    if(bool{result == migraphx::argument(s, expected_result.data())})
    {
        std::cout << "Successfully executed custom MIOpen kernel example with MIGraphX\n";
    }
    else
    {
        std::cout << "Custom MIOpen kernel example failed\n";
    }
    return 0;
}
