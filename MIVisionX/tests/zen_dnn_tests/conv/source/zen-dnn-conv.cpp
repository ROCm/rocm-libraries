/*
MIT License

Copyright (c) 2022 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <assert.h>

#include <chrono>
#include <numeric>
#include <vector>
#include <unordered_map>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

using namespace zendnn;

using namespace std;

memory::dim product(const memory::dims &dims)
{
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

void conv_layer_test(engine::kind engine_kind, int times)
{

    using tag = memory::format_tag;
    using dt = memory::data_type;

    // Start: Initialize engine and stream
    engine eng(engine_kind, 0);
    stream s(eng);
    // End: Initialize engine and stream

    // Batch Size
    const memory::dim batch = 1;

    // Start: Layer 1 - conv1
    zendnnInfo(ZENDNN_TESTLOG, "Layer 1 - conv1 Setup");
    // Start: CONV1 - set dimensions
    memory::dims conv1_src_tensor_dims = {batch, 1, 28, 28}; // {nchw}
    memory::dims conv1_weights_tensor_dims = {20, 1, 5, 5};  // {oihw}
    memory::dims conv1_bias_tensor_dims = {20};
    memory::dims conv1_dst_tensor_dims = {batch, 20, 24, 24};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {0, 0};
    // End: CONV1 - set dimensions

    // Start: Allocate buffers for input, output data, weights, and bias
    std::vector<float> user_src(batch * 1 * 28 * 28);
    std::vector<float> user_dst(batch * 10);
    std::vector<float> conv1_weights(product(conv1_weights_tensor_dims));
    std::vector<float> conv1_bias(product(conv1_bias_tensor_dims));
    // End: Allocate buffers for input, output data, weights, and bias

    // Start: Create memory that describes data layout in the buffers
    // This example uses tag::nchw (batch-channels-height-width) for input data and tag::oihw for weights
    auto user_src_memory = memory({{conv1_src_tensor_dims}, dt::f32, tag::nchw}, eng);
    write_to_zendnn_memory(user_src.data(), user_src_memory);
    auto user_weights_memory = memory({{conv1_weights_tensor_dims}, dt::f32, tag::oihw}, eng);
    write_to_zendnn_memory(conv1_weights.data(), user_weights_memory);
    auto conv1_user_bias_memory = memory({{conv1_bias_tensor_dims}, dt::f32, tag::x}, eng);
    write_to_zendnn_memory(conv1_bias.data(), conv1_user_bias_memory);
    // End: Create memory that describes data layout in the buffers

    // Start: Create convolution memory descriptors with layout tag::any
    // The `any` format enables the convolution primitive to choose the data format
    // that will result in best performance based on its input parameters (convolution
    // kernel sizes, strides, padding, and so on). If the resulting format is different
    // from `nchw`, the user data must be transformed to the format required for
    // the convolution
    auto conv1_src_md = memory::desc({conv1_src_tensor_dims}, dt::f32, tag::any);
    auto conv1_bias_md = memory::desc({conv1_bias_tensor_dims}, dt::f32, tag::any);
    auto conv1_weights_md = memory::desc({conv1_weights_tensor_dims}, dt::f32, tag::any);
    auto conv1_dst_md = memory::desc({conv1_dst_tensor_dims}, dt::f32, tag::any);
    // End: Create convolution memory descriptors with layout tag::any

    // Start: Create convolution descriptor
    // by specifying: propagation kind, convolution algorithm , shapes of input,
    // weights, bias, output, convolution strides, padding, and kind of padding.
    // Propagation kind is set to prop_kind::forward_inference to optimize for
    // inference execution and omit computations that are necessary only for
    // backward propagation.
    auto conv1_desc = convolution_forward::desc(prop_kind::forward_inference,
                                                algorithm::convolution_gemm, conv1_src_md, conv1_weights_md,
                                                conv1_bias_md, conv1_dst_md, conv1_strides, conv1_padding,
                                                conv1_padding);
    // End: Create convolution descriptor

    // Start: Create a convolution primitive descriptor
    // Once created, this descriptor has specific formats instead of the `any`
    // format specified in the convolution descriptor.
    auto conv1_prim_desc = convolution_forward::primitive_desc(conv1_desc, eng);
    // End: Create a convolution primitive descriptor

    // Start: Check data and weights formats - Reorder
    // Data formats required by convolution is different from the user format.
    // In case it is different change the layout using reorder primitive.
    auto conv1_src_memory = user_src_memory;
    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc())
    {
        conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
        reorder(user_src_memory, conv1_src_memory).execute(s, user_src_memory, conv1_src_memory);
        ;
    }

    auto conv1_weights_memory = user_weights_memory;
    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc())
    {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(user_weights_memory, conv1_weights_memory).execute(s, user_weights_memory, conv1_weights_memory);
    }
    // End: Check data and weights formats - Reorder

    // Start: Create a memory primitive for output
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    // End: Create a memory primitive for output

    // Start: Create a convolution primitive and add it to the net
    // Primitive arguments.
    std::unordered_map<int, memory> conv_args;
    conv_args.insert({ZENDNN_ARG_SRC, conv1_src_memory});
    conv_args.insert({ZENDNN_ARG_WEIGHTS, conv1_weights_memory});
    conv_args.insert({ZENDNN_ARG_BIAS, conv1_user_bias_memory});
    conv_args.insert({ZENDNN_ARG_DST, conv1_dst_memory});
    // End: Create a convolution primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "Layer 1 - conv1 Setup Complete");
    // End: Layer 1 - conv1

    // Create the primitive.
    auto conv_prim = convolution_forward(conv1_prim_desc);

    // Primitive execution: convolution
    conv_prim.execute(s, conv_args);

    // Wait for the computation to finalize.
    s.wait();
}

int main(int argc, char **argv)
{
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_conv_f32 test starts");
    try
    {
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                         chrono::steady_clock::now().time_since_epoch())
                         .count();
        int times = 10;
        conv_layer_test(parse_engine_kind(argc, argv), times);
        auto end = chrono::duration_cast<chrono::milliseconds>(
                       chrono::steady_clock::now().time_since_epoch())
                       .count();
        zendnnInfo(ZENDNN_TESTLOG, "Use time ", (end - begin) / (times + 0.0));
    }
    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_conv_f32 test ends\n");
    return 0;
}
