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
#include <iostream>
#include <fstream>

#include "test_utils.hpp"
#include "zendnn_logging.hpp"

#include <opencv2/opencv.hpp>

using namespace zendnn;
using namespace std;

#if USE_OPENCV_4
#define CV_BGR2GRAY COLOR_BGR2GRAY
#endif

memory::dim product(const memory::dims &dims)
{
    return std::accumulate(dims.begin(), dims.end(), (memory::dim)1,
                           std::multiplies<memory::dim>());
}

// Check function status
#define ERROR_CHECK_STATUS(call)                                                               \
    {                                                                                          \
        int status = (call);                                                                   \
        if (status != 0)                                                                       \
        {                                                                                      \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); \
            return status;                                                                     \
        }                                                                                      \
    }

// Intialize weights & bias tensors
static int initializeTensor(std::vector<float> *tensor, size_t tensorSize, FILE *fp, const char *binaryFilename)
{
    size_t itemsize = sizeof(float);
    unsigned int h[2] = {0};

    fread(h, 1, sizeof(h), fp);
    if (h[0] != 0xf00dd1e1 || (size_t)h[1] != (tensorSize * itemsize))
    {
        printf("ERROR: invalid data (magic,size)=(0x%x,%x) in %s at byte position %lu -- expected size is %ld\n", h[0], h[1], binaryFilename, ftell(fp) - sizeof(h), tensorSize * itemsize);
        return -1;
    }

    void *ptr = tensor->data();
    size_t n = fread(ptr, itemsize, tensorSize, fp);
    if (n != tensorSize)
    {
        printf("ERROR: expected char[%zu], but got char[%zu] in %s\n", tensorSize * itemsize, n * itemsize, binaryFilename);
        return -1;
    }

    return 0;
}

// MNIST Caffe Model Zen DNN Implementation
int mnist_caffe_setup(engine::kind engine_kind, const char *binaryFilename, const char *imageFilename, int times)
{

    using tag = memory::format_tag;
    using dt = memory::data_type;

    // Check weights file
    FILE *fp__variables = fopen(binaryFilename, "rb");
    if (!fp__variables)
    {
        printf("ERROR: unable to open: %s\n", binaryFilename);
        return -1;
    }
    {
        unsigned int magic = 0;
        fread(&magic, 1, sizeof(magic), fp__variables);
        if (magic != 0xf00dd1e0)
        {
            fclose(fp__variables);
            printf("ERROR: invalid file magic in %s\n", binaryFilename);
            return -1;
        }
    }

    // Start: Initialize engine and stream
    engine eng(engine_kind, 0);
    stream s(eng);
    // End: Initialize engine and stream

    // Start: Create network
    std::vector<primitive> net;
    std::vector<std::unordered_map<int, memory>> net_args;
    // End: Create network

    // Batch Size
    const memory::dim batch = 1;

    // Start: allocate input & output data
    std::vector<float> user_src(batch * 1 * 28 * 28);
    std::vector<float> user_dst(batch * 10);
    // End: allocate input & output data

    /*
        // Start: Load Input Image - OpenCV for Images
        cv::Mat input = cv::imread(imageFilename);
        if (input.empty())
        {
            printf("Image not found\n");
            return 0;
        }
        else
        {
            printf("Image found -- %s\n", imageFilename);
        }

        cv::Mat img = input.clone();

        // convert to grayscale image
        cv::cvtColor(img, img, cv::CV_BGR2GRAY);

        // resize to 24 x 24
        cv::resize(img, img, cv::Size(24, 24));

        // dilate image
        cv::dilate(img, img, cv::Mat::ones(2, 2, CV_8U));

        // add border to the image so that the digit will go center and become 28 x 28 image
        cv::copyMakeBorder(img, img, 2, 2, 2, 2, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        float *ptr = user_src.data();
        int imageElements = (batch * 1 * 28 * 28);
        for (int y = 0; y < imageElements; y++)
        {
            unsigned char *src = img.data + y;
            float *dst = ptr + y;
            *dst = src[0];
        }
        // End: Load Input Image
    */

    // Start: Load Input Image - Binary Files
    std::ifstream input_binary_image(imageFilename, std::ios::binary);

    input_binary_image.seekg(0, input_binary_image.end);
    int elementCount = (input_binary_image.tellg() / sizeof(float)); // total number of elements
    input_binary_image.seekg(0, input_binary_image.beg);

    if (elementCount == (batch * 1 * 28 * 28))
    {
        input_binary_image.read(reinterpret_cast<char *>(user_src.data()), user_src.size() * sizeof(float));
    }
    else
    {
        printf("ERROR: invalid Binary Image File -- %s: Total Pixels:%d Received: %d\n", imageFilename, (28 * 28), elementCount);
        return -1;
    }
    // End: Load Input Image

    // Start: MNIST Layer 1 - conv1
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 1 - conv1 Setup");
    // Start: CONV1 - set dimensions
    memory::dims conv1_src_tensor_dims = {batch, 1, 28, 28}; // {nchw}
    memory::dims conv1_weights_tensor_dims = {20, 1, 5, 5};  // {oihw}
    memory::dims conv1_bias_tensor_dims = {20};
    memory::dims conv1_dst_tensor_dims = {batch, 20, 24, 24};
    memory::dims conv1_strides = {1, 1};
    memory::dims conv1_padding = {0, 0};
    // End: CONV1 - set dimensions

    // Start: Allocate buffers for input, output data, weights, and bias
    std::vector<float> conv1_weights(product(conv1_weights_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&conv1_weights, product(conv1_weights_tensor_dims), fp__variables, binaryFilename));
    std::vector<float> conv1_bias(product(conv1_bias_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&conv1_bias, product(conv1_bias_tensor_dims), fp__variables, binaryFilename));
    // End: Allocate buffers for input, output data, weights, and bias

    // Start: Create memory that describes data layout in the buffers
    // This example uses tag::nchw (batch-channels-height-width) for input data and tag::oihw for weights
    auto user_src_memory = memory({{conv1_src_tensor_dims}, dt::f32, tag::nchw}, eng);
    write_to_zendnn_memory(user_src.data(), user_src_memory);
    auto conv1_user_weights_memory = memory({{conv1_weights_tensor_dims}, dt::f32, tag::oihw}, eng);
    write_to_zendnn_memory(conv1_weights.data(), conv1_user_weights_memory);
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
        net.push_back(reorder(user_src_memory, conv1_src_memory));
        net_args.push_back({{ZENDNN_ARG_FROM, user_src_memory},
                            {ZENDNN_ARG_TO, conv1_src_memory}});
    }

    auto conv1_weights_memory = conv1_user_weights_memory;
    if (conv1_prim_desc.weights_desc() != conv1_user_weights_memory.get_desc())
    {
        conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
        reorder(conv1_user_weights_memory, conv1_weights_memory)
            .execute(s, conv1_user_weights_memory, conv1_weights_memory);
    }
    // End: Check data and weights formats - Reorder

    // Start: Create a memory primitive for output
    auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);
    // End: Create a memory primitive for output

    // Start: Create a convolution primitive and add it to the net
    net.push_back(convolution_forward(conv1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_src_memory},
                        {ZENDNN_ARG_WEIGHTS, conv1_weights_memory},
                        {ZENDNN_ARG_BIAS, conv1_user_bias_memory},
                        {ZENDNN_ARG_DST, conv1_dst_memory}});
    // End: Create a convolution primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 1 - conv1 Setup Complete");
    // End: MNIST Layer 1 - conv1

    // Start: MNIST Layer 2 - pool1
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 2 - pool1 Setup");
    // Start: pool1 - set dimensions
    memory::dims pool1_dst_tensor_dims = {batch, 20, 12, 12};
    memory::dims pool1_kernel = {2, 2};
    memory::dims pool1_strides = {2, 2};
    memory::dims pool1_padding = {0, 0};
    // End: pool1 - set dimensions

    // Start: pool1 - set dst memory desc
    auto pool1_dst_md = memory::desc({pool1_dst_tensor_dims}, dt::f32, tag::any);
    // End: pool1 - set dst memory desc

    // Start: Create pooling 1 primitive
    // For training execution, pooling requires a private workspace memory
    // to perform the backward pass. However, pooling should not use 'workspace'
    // for inference, because this is detrimental to performance.
    auto pool1_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            algorithm::pooling_max, conv1_dst_memory.get_desc(), pool1_dst_md,
                                            pool1_strides, pool1_kernel, pool1_padding, pool1_padding);
    auto pool1_pd = pooling_forward::primitive_desc(pool1_desc, eng);
    // End: Create pooling 1 primitive

    // Start: pool1 - dst memory
    auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);
    // End: pool1 - dst memory

    // Start: Create a pooling primitive and add it to the net
    net.push_back(pooling_forward(pool1_pd));
    net_args.push_back({{ZENDNN_ARG_SRC, conv1_dst_memory},
                        {ZENDNN_ARG_DST, pool1_dst_memory}});
    // End: Create a pooling primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 2 - pool1 Setup Complete");
    // End: MNIST Layer 2 - pool1

    // Start: MNIST Layer 3 - conv2
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 3 - conv2 Setup");
    // Start: CONV2 - set dimensions
    memory::dims conv2_weights_tensor_dims = {50, 20, 5, 5};
    memory::dims conv2_bias_tensor_dims = {50};
    memory::dims conv2_dst_tensor_dims = {batch, 50, 8, 8};
    memory::dims conv2_strides = {1, 1};
    memory::dims conv2_padding = {0, 0};
    // End: CONV2 - set dimensions

    // Start: weights, and bias
    std::vector<float> conv2_weights(product(conv2_weights_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&conv2_weights, product(conv2_weights_tensor_dims), fp__variables, binaryFilename));
    std::vector<float> conv2_bias(product(conv2_bias_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&conv2_bias, product(conv2_bias_tensor_dims), fp__variables, binaryFilename));
    // End: weights, and bias

    // Start: Create memory that describes data layout in the buffers
    auto conv2_user_weights_memory = memory({{conv2_weights_tensor_dims}, dt::f32, tag::oihw}, eng);
    write_to_zendnn_memory(conv2_weights.data(), conv2_user_weights_memory);
    auto conv2_user_bias_memory = memory({{conv2_bias_tensor_dims}, dt::f32, tag::x}, eng);
    write_to_zendnn_memory(conv2_bias.data(), conv2_user_bias_memory);
    // End: Create memory that describes data layout in the buffers

    // Start: Create convolution memory descriptors with layout tag::any
    auto conv2_weights_md = memory::desc({conv2_weights_tensor_dims}, dt::f32, tag::any);
    auto conv2_bias_md = memory::desc({conv2_bias_tensor_dims}, dt::f32, tag::any);
    auto conv2_dst_md = memory::desc({conv2_dst_tensor_dims}, dt::f32, tag::any);
    // End: Create convolution memory descriptors with layout tag::any

    // Start: Create convolution prinitive descriptor
    auto conv2_desc = convolution_forward::desc(prop_kind::forward_inference,
                                                algorithm::convolution_gemm, pool1_dst_memory.get_desc(), conv2_weights_md,
                                                conv2_bias_md, conv2_dst_md, conv2_strides, conv2_padding,
                                                conv2_padding);
    auto conv2_prim_desc = convolution_forward::primitive_desc(conv2_desc, eng);
    // End: Create convolution prinitive descriptor

    // Start: Check weights formats - Reorder
    auto conv2_weights_memory = conv2_user_weights_memory;
    if (conv2_prim_desc.weights_desc() != conv2_user_weights_memory.get_desc())
    {
        conv2_weights_memory = memory(conv2_prim_desc.weights_desc(), eng);
        reorder(conv2_user_weights_memory, conv2_weights_memory)
            .execute(s, conv2_user_weights_memory, conv2_weights_memory);
    }
    // End: Check weights formats - Reorder

    // Start: Create a memory primitive for output
    auto conv2_dst_memory = memory(conv2_prim_desc.dst_desc(), eng);
    // End: Create a memory primitive for output

    // Start: Create a convolution primitive and add it to the net
    net.push_back(convolution_forward(conv2_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, pool1_dst_memory},
                        {ZENDNN_ARG_WEIGHTS, conv2_weights_memory},
                        {ZENDNN_ARG_BIAS, conv2_user_bias_memory},
                        {ZENDNN_ARG_DST, conv2_dst_memory}});
    // End: Create a convolution primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 3 - conv2 Setup Complete");
    // End: MNIST Layer 3 - conv2

    // Start: MNIST Layer 4 - pool2
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 4 - pool2 Setup");
    // Start: pool2 - set dimensions
    memory::dims pool2_dst_tensor_dims = {batch, 50, 4, 4};
    memory::dims pool2_kernel = {2, 2};
    memory::dims pool2_strides = {2, 2};
    memory::dims pool2_padding = {0, 0};
    // End: pool2 - set dimensions

    // Start: pool2 - set dst memory desc
    auto pool2_dst_md = memory::desc({pool2_dst_tensor_dims}, dt::f32, tag::any);
    // End: pool2 - set dst memory desc

    // Start: Create pooling 2 primitive
    // For training execution, pooling requires a private workspace memory
    // to perform the backward pass. However, pooling should not use 'workspace'
    // for inference, because this is detrimental to performance.
    auto pool2_desc = pooling_forward::desc(prop_kind::forward_inference,
                                            algorithm::pooling_max, conv2_dst_memory.get_desc(), pool2_dst_md,
                                            pool2_strides, pool2_kernel, pool2_padding, pool2_padding);
    auto pool2_pd = pooling_forward::primitive_desc(pool2_desc, eng);
    // End: Create pooling 2 primitive

    // Start: pool2 - dst memory
    auto pool2_dst_memory = memory(pool2_pd.dst_desc(), eng);
    // End: pool2 - dst memory

    // Start: Create a pooling primitive and add it to the net
    net.push_back(pooling_forward(pool2_pd));
    net_args.push_back({{ZENDNN_ARG_SRC, conv2_dst_memory},
                        {ZENDNN_ARG_DST, pool2_dst_memory}});
    // End: Create a pooling primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 4 - pool2 Setup Complete");
    // End: MNIST Layer 4 - pool2

    // Start: MNIST Layer 5 - ip1
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 5 - ip1 Setup");
    // Start: fc1 inner product 1 - set dimensions
    memory::dims fc1_weights_tensor_dims = {500, 50, 4, 4};
    memory::dims fc1_bias_tensor_dims = {500};
    memory::dims fc1_dst_tensor_dims = {batch, 500};
    // End: fc1 inner product 1 - set dimensions

    // Start: weights, and bias
    std::vector<float> fc1_weights(product(fc1_weights_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&fc1_weights, product(fc1_weights_tensor_dims), fp__variables, binaryFilename));
    std::vector<float> fc1_bias(product(fc1_bias_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&fc1_bias, product(fc1_bias_tensor_dims), fp__variables, binaryFilename));
    // End: weights, and bias

    // Start: Create memory that describes data layout in the buffers
    auto fc1_user_weights_memory = memory({{fc1_weights_tensor_dims}, dt::f32, tag::oihw}, eng);
    write_to_zendnn_memory(fc1_weights.data(), fc1_user_weights_memory);
    auto fc1_user_bias_memory = memory({{fc1_bias_tensor_dims}, dt::f32, tag::x}, eng);
    write_to_zendnn_memory(fc1_bias.data(), fc1_user_bias_memory);
    // End: Create memory that describes data layout in the buffers

    // Start: Create ip1 memory descriptors with layout tag::any
    auto fc1_weights_md = memory::desc({fc1_weights_tensor_dims}, dt::f32, tag::any);
    auto fc1_bias_md = memory::desc({fc1_bias_tensor_dims}, dt::f32, tag::any);
    auto fc1_dst_md = memory::desc({fc1_dst_tensor_dims}, dt::f32, tag::any);
    // End: Create ip1 memory descriptors with layout tag::any

    // Start: Create inner product primitive descriptor
    auto fc1_desc = inner_product_forward::desc(prop_kind::forward_inference,
                                                pool2_dst_memory.get_desc(), fc1_weights_md, fc1_bias_md, fc1_dst_md);
    auto fc1_prim_desc = inner_product_forward::primitive_desc(fc1_desc, eng);
    // End: Create inner product primitive descriptor

    // Start: Check weights formats - Reorder
    auto fc1_weights_memory = fc1_user_weights_memory;
    if (fc1_prim_desc.weights_desc() != fc1_user_weights_memory.get_desc())
    {
        fc1_weights_memory = memory(fc1_prim_desc.weights_desc(), eng);
        reorder(fc1_user_weights_memory, fc1_weights_memory)
            .execute(s, fc1_user_weights_memory, fc1_weights_memory);
    }
    // End: Check weights formats - Reorder

    // Start: IP1 - dst memory
    auto fc1_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);
    // End: IP1 - dst memory

    // Start: Create a IP primitive and add it to the net
    net.push_back(inner_product_forward(fc1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, pool2_dst_memory},
                        {ZENDNN_ARG_WEIGHTS, fc1_weights_memory},
                        {ZENDNN_ARG_BIAS, fc1_user_bias_memory},
                        {ZENDNN_ARG_DST, fc1_dst_memory}});
    // End: Create a IP primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 5 - ip1 Setup Complete");
    // End: MNIST Layer 5 - ip1

    // Start: MNIST Layer 6 - relu1
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 6 - relu1 Setup");
    const float alpha = 0.0f;
    const float beta = 0.0f;

    // Start: Create relu1 primitive descriptor
    // For better performance, keep the input data format for ReLU
    // (as well as for other operation primitives until another
    // convolution or inner product is encountered) the same as the one chosen
    // for convolution. Also note that ReLU can be done in-place
    auto relu1_desc = eltwise_forward::desc(prop_kind::forward_inference,
                                            algorithm::eltwise_relu, fc1_dst_memory.get_desc(),
                                            alpha, beta);
    auto relu1_prim_desc = eltwise_forward::primitive_desc(relu1_desc, eng);
    // End: Create relu1 primitive descriptor

    // Start: Create a relu1 primitive and add it to the net
    net.push_back(eltwise_forward(relu1_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, fc1_dst_memory},
                        {ZENDNN_ARG_DST, fc1_dst_memory}});
    // End: Create a relu1 primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 6 - relu1 Setup Complete");
    // End: MNIST Layer 6 - relu1

    // Start: MNIST Layer 7 - ip2
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 7 - ip2 Setup");
    // Start: fc2 inner product 2 - set dimensions
    memory::dims fc2_weights_tensor_dims = {10, 500};
    memory::dims fc2_bias_tensor_dims = {10};
    memory::dims fc2_dst_tensor_dims = {batch, 10};
    // End: fc2 inner product 2 - set dimensions

    // Start: weights, and bias
    std::vector<float> fc2_weights(product(fc2_weights_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&fc2_weights, product(fc2_weights_tensor_dims), fp__variables, binaryFilename));
    std::vector<float> fc2_bias(product(fc2_bias_tensor_dims));
    ERROR_CHECK_STATUS(initializeTensor(&fc2_bias, product(fc2_bias_tensor_dims), fp__variables, binaryFilename));
    // End: weights, and bias

    // Start: Create memory that describes data layout in the buffers
    auto fc2_user_weights_memory = memory({{fc2_weights_tensor_dims}, dt::f32, tag::nc}, eng);
    write_to_zendnn_memory(fc2_weights.data(), fc2_user_weights_memory);
    auto fc2_user_bias_memory = memory({{fc2_bias_tensor_dims}, dt::f32, tag::x}, eng);
    write_to_zendnn_memory(fc2_bias.data(), fc2_user_bias_memory);
    // End: Create memory that describes data layout in the buffers

    // Start: Create ip2 memory descriptors with layout tag::any
    auto fc2_weights_md = memory::desc({fc2_weights_tensor_dims}, dt::f32, tag::any);
    auto fc2_bias_md = memory::desc({fc2_bias_tensor_dims}, dt::f32, tag::any);
    auto fc2_dst_md = memory::desc({fc2_dst_tensor_dims}, dt::f32, tag::any);
    // End: Create ip1 memory descriptors with layout tag::any

    // Start: Create inner product primitive descriptor
    auto fc2_desc = inner_product_forward::desc(prop_kind::forward_inference,
                                                fc1_dst_memory.get_desc(), fc2_weights_md, fc2_bias_md, fc2_dst_md);
    auto fc2_prim_desc = inner_product_forward::primitive_desc(fc2_desc, eng);
    // End: Create inner product primitive descriptor

    // Start: Check weights formats - Reorder
    auto fc2_weights_memory = fc2_user_weights_memory;
    if (fc2_prim_desc.weights_desc() != fc2_user_weights_memory.get_desc())
    {
        fc2_weights_memory = memory(fc2_prim_desc.weights_desc(), eng);
        reorder(fc2_user_weights_memory, fc2_weights_memory)
            .execute(s, fc2_user_weights_memory, fc2_weights_memory);
    }
    // End: Check weights formats - Reorder

    // Start: IP2 - dst memory
    auto fc2_dst_memory = memory(fc1_prim_desc.dst_desc(), eng);
    // End: IP2 - dst memory

    // Start: Create a IP 2 primitive and add it to the net
    net.push_back(inner_product_forward(fc2_prim_desc));
    net_args.push_back({{ZENDNN_ARG_SRC, fc1_dst_memory},
                        {ZENDNN_ARG_WEIGHTS, fc2_weights_memory},
                        {ZENDNN_ARG_BIAS, fc2_user_bias_memory},
                        {ZENDNN_ARG_DST, fc2_dst_memory}});
    // End: Create a IP primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 7 - ip2 Setup Complete");
    // End: MNIST Layer 7 - ip2

    // Start: MNIST Layer 8 - softMax
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 8 - softmax Setup");

    // Softmax axis.
    const int sm_axis = 1;

    // Start: Create softmax primitive
    auto softmax_desc = softmax_forward::desc(prop_kind::forward_inference, fc2_dst_memory.get_desc(), sm_axis);
    auto softmax_pd = softmax_forward::primitive_desc(softmax_desc, eng);
    // End: Create softmax primitive

    // Start: Create a softmax primitive and add it to the net
    net.push_back(softmax_forward(softmax_pd));
    net_args.push_back({{ZENDNN_ARG_SRC, fc2_dst_memory},
                        {ZENDNN_ARG_DST, fc2_dst_memory}});
    // End: Create a softmax primitive and add it to the net
    zendnnInfo(ZENDNN_TESTLOG, "MNIST Layer 8 - softmax Setup Complete");
    // End: MNIST Layer 8 - softMax

    {
        unsigned int magic = 0;
        fread(&magic, 1, sizeof(magic), fp__variables);
        fclose(fp__variables);
        if (magic != 0xf00dd1e2)
        {
            printf("ERROR: invalid eoff magic in %s\n", binaryFilename);
            return -1;
        }
    }

    // Start: Execute primitives
    // For this example, the net is executed multiple times and each execution is timed individually.
    float AverageTime = 0;
    for (int j = 0; j < times; ++j)
    {
        auto start_exec_time = std::chrono::high_resolution_clock::now();
        assert(net.size() == net_args.size() && "something is missing");
        for (size_t i = 0; i < net.size(); ++i)
        {
            net.at(i).execute(s, net_args.at(i));
        }
        // Wait for the computation to finalize.
        s.wait();
        auto end_exec_time = std::chrono::high_resolution_clock::now();
        auto exec_duration = end_exec_time - start_exec_time;
        auto exec_in_millis = std::chrono::duration_cast<std::chrono::milliseconds>(exec_duration);
        printf("Execution:%d -- \t%.8f ms\n", j, (float)exec_in_millis.count());

        // Start: Read output from engine
        auto start_mem_time = std::chrono::high_resolution_clock::now();
        read_from_zendnn_memory(user_dst.data(), fc2_dst_memory);
        auto end_mem_time = std::chrono::high_resolution_clock::now();
        auto exec_mem_duration = end_mem_time - start_mem_time;
        auto mem_in_millis = std::chrono::duration_cast<std::chrono::milliseconds>(exec_mem_duration);
        //printf("Mem Transfer:%d -- \t%.8f ms\n", j, (float)mem_in_millis.count());
        // End: Read output from engine

        AverageTime += (float)exec_in_millis.count() + (float)mem_in_millis.count();
    }
    // End: Execute primitives
    printf("\nAvg Inference Time -- \t%.8f ms\n",(float)(AverageTime/times));

    // Start: Print Output Probabilty
    printf("\nMNIST Probability Result for - %s\n", imageFilename);
    for (int j = 0; j < 10; ++j)
    {
        printf("Class:%d -- %.3f\n", j, user_dst[j]);
    }
    printf("\n");
    // End: Print Output Probabilty

    return 0;
}

int main(int argc, const char **argv)
{
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_mnist_app test starts");

    // check command-line usage
    if (argc != 3)
    {
        printf(
            "\n"
            "Usage: ZENDNN_LOG_OPTS=ALL:5 ZENDNN_VERBOSE=1 ./zendnn_mnist_app [weights.bin] [imageName]\n"
            "\n"
            "   <weights.bin>: weights file to be used for the inference\n"
            "   <imageName>: image file to be used for the inference\n"
            "\n");
        return -1;
    }

    engine::kind engine_kind = zendnn::engine::kind::cpu;
    const char *weights = argv[1];
    const char *image = argv[2];
    int NumExecution = 10;

    try
    {
        auto begin = chrono::duration_cast<chrono::milliseconds>(
                         chrono::steady_clock::now().time_since_epoch())
                         .count();

        int app_status = mnist_caffe_setup(engine_kind, weights, image, NumExecution);

        auto end = chrono::duration_cast<chrono::milliseconds>(
                       chrono::steady_clock::now().time_since_epoch())
                       .count();
        zendnnInfo(ZENDNN_TESTLOG, "Use time ", (end - begin) / (NumExecution + 0.0));
    }

    catch (error &e)
    {
        std::cerr << "status: " << e.status << std::endl;
        std::cerr << "message: " << e.message << std::endl;
    }
    zendnnInfo(ZENDNN_TESTLOG, "zendnn_mnist_app test ends\n");
    return 0;
}
