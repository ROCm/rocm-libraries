// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/node/ConvolutionWgradNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestConvolutionWgradNode, PreValidateNode)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionWgradNode, PreValidateNodeMissingXTensor)
{
    ConvWgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // X tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, PreValidateNodeMissingDyTensor)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // Dy tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, PreValidateNodeMissingDwTensor)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // Dw tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, PreValidateNodeMissingConvolutionParameters)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    // Convolution parameters are missing
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, PreValidateNodeAllValuesSet)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeMissingXTensor)
{
    ConvWgradAttributes convAttributes;
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeMissingDyTensor)
{
    ConvWgradAttributes convAttributes;
    convAttributes.set_x(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeMissingDwTensor)
{
    ConvWgradAttributes convAttributes;
    convAttributes.set_x(std::make_shared<TensorAttributes>());
    convAttributes.set_dy(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, InferPropertiesNode2DConvolutionSuccess)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels (from dy)
    EXPECT_EQ(inferredDims[1], 3); // Input channels (from x)
    // Kernel height: ((32 + 1 + 1 - 1 * (32 - 1) - 1) / 1) + 1 = (32 + 2 - 31 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    EXPECT_EQ(inferredDims[3], 3);
}

TEST(TestConvolutionWgradNode, InferPropertiesNode3DConvolutionSuccess)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 16, 8, 16, 16});
    xTensor->set_stride({32768, 2048, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 32, 6, 16, 16});
    dyTensor->set_stride({49152, 1536, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 1, 1});
    convAttributes.set_post_padding({0, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 32); // Output channels (from dy)
    EXPECT_EQ(inferredDims[1], 16); // Input channels (from x)
    // Depth kernel: ((8 + 0 + 0 - 1 * (6 - 1) - 1) / 1) + 1 = (8 - 5 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    // Height kernel: ((16 + 1 + 1 - 1 * (16 - 1) - 1) / 1) + 1 = (16 + 2 - 15 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[3], 3);
    EXPECT_EQ(inferredDims[4], 3);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeWithStride2x2)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 31, 31});
    xTensor->set_stride({2883, 961, 31, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 16, 16});
    dyTensor->set_stride({16384, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({2, 2}); // 2x2 stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // Kernel height: ((31 + 1 + 1 - 2 * (16 - 1) - 1) / 1) + 1 = (31 + 2 - 30 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    EXPECT_EQ(inferredDims[3], 3);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeWithDilation2x2)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 16, 20, 20});
    xTensor->set_stride({6400, 400, 20, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 32, 20, 20});
    dyTensor->set_stride({12800, 400, 20, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 32); // Output channels
    EXPECT_EQ(inferredDims[1], 16); // Input channels
    // Kernel height: ((20 + 2 + 2 - 1 * (20 - 1) - 1) / 2) + 1 = (20 + 4 - 19 - 1) / 2 + 1 = 4 / 2 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    EXPECT_EQ(inferredDims[3], 3);
}

TEST(TestConvolutionWgradNode, PackNode)
{
    ConvWgradAttributes convAttributes;
    convAttributes.set_name("ConvolutionWgrad");

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(2)
        .set_name("DyTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_uid(3)
        .set_name("DwTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});
    convAttributes.set_convolution_mode(hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "ConvolutionWgrad");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::ConvolutionWrwAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_ConvolutionWrwAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->dw_tensor_uid(), dwTensor->get_uid());

    ASSERT_EQ(packedAttributes->pre_padding()->size(), 2);
    EXPECT_EQ(packedAttributes->pre_padding()->Get(0), 1);
    EXPECT_EQ(packedAttributes->pre_padding()->Get(1), 1);

    ASSERT_EQ(packedAttributes->post_padding()->size(), 2);
    EXPECT_EQ(packedAttributes->post_padding()->Get(0), 1);
    EXPECT_EQ(packedAttributes->post_padding()->Get(1), 1);

    ASSERT_EQ(packedAttributes->stride()->size(), 2);
    EXPECT_EQ(packedAttributes->stride()->Get(0), 1);
    EXPECT_EQ(packedAttributes->stride()->Get(1), 1);

    ASSERT_EQ(packedAttributes->dilation()->size(), 2);
    EXPECT_EQ(packedAttributes->dilation()->Get(0), 1);
    EXPECT_EQ(packedAttributes->dilation()->Get(1), 1);

    EXPECT_EQ(packedAttributes->conv_mode(), hipdnn_sdk::data_objects::ConvMode::CROSS_CORRELATION);
}

TEST(TestConvolutionWgradNode, GatherHipdnnTensor)
{
    ConvWgradAttributes convAttributes;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(2).set_name("DyTensor");
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_uid(3).set_name("DwTensor");
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dwTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3);
}

TEST(TestConvolutionWgradNode, StrideInferenceNchwLayoutSuccess)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    // No stride set - should be inferred
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 27); // K stride: 3 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 9); // C stride: 3 * 3
    EXPECT_EQ(inferredStrides[2], 3); // R stride: 3
    EXPECT_EQ(inferredStrides[3], 1); // S stride: 1
}

TEST(TestConvolutionWgradNode, StrideInferenceNhwcLayoutSuccess)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32}); // Dims in NCHW order
    xTensor->set_stride({3072, 1, 96, 3}); // NHWC strides (channels last)
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32}); // Dims in NCHW order
    dyTensor->set_stride({65536, 1, 2048, 64}); // NHWC strides (channels last)
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3}); // Dims in KCRS order
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 27); // K stride: 3 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 9); // R stride: 3 * 3
    EXPECT_EQ(inferredStrides[3], 3); // S stride: 3
}

TEST(TestConvolutionWgradNode, InferPropertiesGroupedConv2Groups)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // 128 output channels
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 128); // Output channels
    EXPECT_EQ(inferredDims[1], 64); // Input channels (assume 1 group)
    EXPECT_EQ(inferredDims[2], 3); // Kernel height
    EXPECT_EQ(inferredDims[3], 3); // Kernel width
}

TEST(TestConvolutionWgradNode, PreValidateTensorDimsTooFew)
{
    ConvWgradAttributes convAttributes;

    // Test with too few dimensions (less than 3)
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3}); // Only 2 dimensions
    xTensor->set_stride({3, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64});
    dyTensor->set_stride({64, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1});
    convAttributes.set_post_padding({1});
    convAttributes.set_stride({1});
    convAttributes.set_dilation({1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateDyDimsMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32, 32}); // 5D dy for 4D x
    dyTensor->set_stride({2097152, 32768, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateBatchSizeMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32}); // Batch size 1
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}); // Batch size 2
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateSpatialParamMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32, 32}); // 3D spatial
    xTensor->set_stride({98304, 32768, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32, 32});
    dyTensor->set_stride({2097152, 32768, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    // Only 2 spatial parameters for 3D spatial dimensions
    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateNegativeStride)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, -1}); // Negative stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateZeroDilation)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 0}); // Zero dilation

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateNegativePrePadding)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({-1, 1}); // Negative padding
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesWithLargeStride)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 22, 22});
    xTensor->set_stride({1452, 484, 22, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 8, 8});
    dyTensor->set_stride({4096, 64, 8, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({3, 3}); // Large stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // Kernel height: ((22 + 2 + 2 - 3 * (8 - 1) - 1) / 1) + 1 = (22 + 4 - 21 - 1) / 1 + 1 = 4 / 1 + 1 = 5
    EXPECT_EQ(inferredDims[2], 5);
    EXPECT_EQ(inferredDims[3], 5);
}

TEST(TestConvolutionWgradNode, InferPropertiesZeroPadding)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 28, 28});
    dyTensor->set_stride({100352, 784, 28, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 0}); // No padding
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 128); // Output channels
    EXPECT_EQ(inferredDims[1], 64); // Input channels
    // Kernel height: ((32 + 0 + 0 - 1 * (28 - 1) - 1) / 1) + 1 = (32 - 27 - 1) / 1 + 1 = 4 / 1 + 1 = 5
    EXPECT_EQ(inferredDims[2], 5);
    EXPECT_EQ(inferredDims[3], 5);
}

TEST(TestConvolutionWgradNode, InferPropertiesAsymmetricPadding)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32, 16, 16});
    xTensor->set_stride({8192, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 15, 15});
    dyTensor->set_stride({14400, 225, 15, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 1}); // Asymmetric padding
    convAttributes.set_post_padding({1, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels
    EXPECT_EQ(inferredDims[1], 32); // Input channels
    // Kernel height: ((16 + 0 + 1 - 1 * (15 - 1) - 1) / 1) + 1 = (16 + 1 - 14 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    // Kernel width: ((16 + 1 + 0 - 1 * (15 - 1) - 1) / 1) + 1 = (16 + 1 - 14 - 1) / 1 + 1 = 2 / 1 + 1 = 3
    EXPECT_EQ(inferredDims[3], 3);
}

TEST(TestConvolutionWgradNode, PreValidateGroupedConvInvalidOutputChannels)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 63, 32, 32}); // 63 output channels
    dyTensor->set_stride({64512, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({63, 32, 3, 3}); // Groups = 64/32 = 2, but 63 is not divisible by 2
    dwTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateGroupedConv2GroupsWithDwSet)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32}); // 64 output channels
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 32 input channels per group (2 groups)
    dwTensor->set_dim({64, 32, 3, 3});
    dwTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionWgradNode, PreValidateGroupedConv4GroupsWithDwSet)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 16, 16}); // 64 input channels
    xTensor->set_stride({16384, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 128, 16, 16}); // 128 output channels
    dyTensor->set_stride({32768, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 128 output channels, 16 input channels per group (4 groups)
    dwTensor->set_dim({128, 16, 3, 3});
    dwTensor->set_stride({144, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionWgradNode, PreValidateGroupedConvInvalidInputChannels)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 63, 32, 32}); // 63 input channels
    xTensor->set_stride({64512, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 63 is not divisible by 32
    dwTensor->set_dim({64, 32, 3, 3});
    dwTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferGroupedConvStrideInferenceNchwLayout)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32});
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // Set dimensions to establish groups = 2
    dwTensor->set_dim({128, 32, 3, 3});
    // No stride set - should be inferred
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 288); // N stride: 32 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 9); // C stride: 3 * 3
    EXPECT_EQ(inferredStrides[2], 3); // H stride: 3
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionWgradNode, InferGroupedConvStrideInferenceNhwcLayout)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // Dims in NCHW order
    xTensor->set_stride({65536, 1, 2048, 64}); // NHWC strides
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // Dims in NCHW order
    dyTensor->set_stride({131072, 1, 4096, 128}); // NHWC
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // Set dimensions to establish groups = 2
    dwTensor->set_dim({128, 32, 3, 3}); // Dims in KCRS order
    // No stride set - should be inferred
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // KRSC strides
    EXPECT_EQ(inferredStrides[0], 288); // K stride: 32 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 96); // R stride: 3 * 32
    EXPECT_EQ(inferredStrides[3], 32); // S stride: 32
}

TEST(TestConvolutionWgradNode, InferGroupedConvWithDilation)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 32, 20, 20});
    xTensor->set_stride({12800, 400, 20, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 20, 20});
    dyTensor->set_stride({25600, 400, 20, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 16 input channels per group (groups = 32/16 = 2)
    dwTensor->set_dim({64, 16, 3, 3});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 144); // K stride: 16 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 9); // C stride: 3 * 3
    EXPECT_EQ(inferredStrides[2], 3); // R stride: 3
    EXPECT_EQ(inferredStrides[3], 1); // S stride: 1
}

TEST(TestConvolutionWgradNode, InferGroupedConvWithLargeStride)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 22, 22});
    xTensor->set_stride({30976, 484, 22, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 128, 8, 8});
    dyTensor->set_stride({8192, 64, 8, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 128 output channels, 8 input channels per group (groups = 64/8 = 8)
    dwTensor->set_dim({128, 8, 5, 5});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({3, 3}); // Large stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 200); // N stride: 8 * 5 * 5
    EXPECT_EQ(inferredStrides[1], 25); // C stride: 5 * 5
    EXPECT_EQ(inferredStrides[2], 5); // H stride: 5
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionWgradNode, InferGroupedConv3D)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32, 8, 16, 16});
    xTensor->set_stride({65536, 2048, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 8, 16, 16});
    dyTensor->set_stride({131072, 2048, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 8 input channels per group (groups = 32/8 = 4)
    dwTensor->set_dim({64, 8, 3, 3, 3});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 5);
    EXPECT_EQ(inferredStrides[0], 216); // N stride: 8 * 3 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 27); // C stride: 3 * 3 * 3
    EXPECT_EQ(inferredStrides[2], 9); // D stride: 3 * 3
    EXPECT_EQ(inferredStrides[3], 3); // H stride: 3
    EXPECT_EQ(inferredStrides[4], 1); // W stride: 1
}

TEST(TestConvolutionWgradNode, InferGroupedConvDepthwiseSeparable)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 32, 112, 112}); // 32 input channels
    xTensor->set_stride({401408, 12544, 112, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 32, 112, 112}); // 32 output channels (same as input for depthwise)
    dyTensor->set_stride({401408, 12544, 112, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    // Depthwise: 32 output channels, 1 input channel per group (32 groups)
    dwTensor->set_dim({32, 1, 3, 3});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dwTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 9); // N stride: 1 * 3 * 3
    EXPECT_EQ(inferredStrides[1], 9); // C stride: 3 * 3
    EXPECT_EQ(inferredStrides[2], 3); // H stride: 3
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionWgradNode, InferPropertiesInvalidSpatialDimensions)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 3, 3}); // Very small input
    xTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 10, 10}); // Large output
    dyTensor->set_stride({6400, 100, 10, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    // Will fail because: 3 + 0 + 0 - 1 * (10 - 1) - 1 = 3 - 9 - 1 = -7 (negative)
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesIncompatibleDilation)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 16, 20, 20});
    xTensor->set_stride({6400, 400, 20, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 32, 19, 19});
    dyTensor->set_stride({11552, 361, 19, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    // numerator = 20 + 2 + 2 - 1 * (19 - 1) - 1 = 20 + 4 - 18 - 1 = 5
    // 5 % 2 = 1 (not divisible by dilation)
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferProperties1x1Convolution)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 14, 14});
    xTensor->set_stride({50176, 196, 14, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 512, 14, 14});
    dyTensor->set_stride({100352, 196, 14, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 512); // Output channels
    EXPECT_EQ(inferredDims[1], 256); // Input channels
    // Kernel height: ((14 + 0 + 0 - 1 * (14 - 1) - 1) / 1) + 1 = (14 - 13 - 1) / 1 + 1 = 0 / 1 + 1 = 1
    EXPECT_EQ(inferredDims[2], 1);
    EXPECT_EQ(inferredDims[3], 1);
}

TEST(TestConvolutionWgradNode, InferPropertiesLargeKernel7x7)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 224, 224});
    xTensor->set_stride({150528, 50176, 224, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 112, 112});
    dyTensor->set_stride({802816, 12544, 112, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({3, 3});
    convAttributes.set_post_padding({3, 3});
    convAttributes.set_stride({2, 2});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    EXPECT_EQ(inferredDims[2], 8);
    EXPECT_EQ(inferredDims[3], 8);
}

TEST(TestConvolutionWgradNode, InferPropertiesWithComplexDilation)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 32, 16, 16});
    xTensor->set_stride({8192, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 14, 14});
    dyTensor->set_stride({12544, 196, 14, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 1}); // Dilation only in one dimension

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 64); // Output channels
    EXPECT_EQ(inferredDims[1], 32); // Input channels
    // Kernel height with dilation=2: ((16 + 1 + 1 - 1 * (14 - 1) - 1) / 2) + 1 = (18 - 13 - 1) / 2 + 1 = 4 / 2 + 1 = 3
    EXPECT_EQ(inferredDims[2], 3);
    // Kernel width with dilation=1: ((16 + 1 + 1 - 1 * (14 - 1) - 1) / 1) + 1 = (18 - 13 - 1) / 1 + 1 = 4 / 1 + 1 = 5
    EXPECT_EQ(inferredDims[3], 5);
}

TEST(TestConvolutionWgradNode, PreValidateOutputChannelMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // 128 output channels
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3}); // 64 output channels, doesn't match dy
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesLargeKernel5x5WithZeroPadding)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 28, 28});
    xTensor->set_stride({50176, 784, 28, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 24, 24});
    dyTensor->set_stride({73728, 576, 24, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dwTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 128); // Output channels
    EXPECT_EQ(inferredDims[1], 64); // Input channels
    // Kernel height: ((28 + 0 + 0 - 1 * (24 - 1) - 1) / 1) + 1 = (28 - 23 - 1) / 1 + 1 = 4 / 1 + 1 = 5
    EXPECT_EQ(inferredDims[2], 5);
    EXPECT_EQ(inferredDims[3], 5);
}

TEST(TestConvolutionWgradNode, PreValidateDwDimsMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3, 3}); // 5D weight for 4D input
    dwTensor->set_stride({81, 27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeNegativeStrideValues)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({-1, 1}); // Negative stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeNegativeDilationValues)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, -1}); // Negative dilation

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeNegativePrePaddingValues)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({-1, 1}); // Negative pre-padding
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesNodeNegativePostPaddingValues)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, -2}); // Negative post-padding
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateNegativePostPadding)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, -1}); // Negative post-padding
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, PreValidateSpatialDimensionMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 28, 28}); // Incorrect spatial dimensions
    dyTensor->set_stride({50176, 784, 28, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3}); // 3x3 kernel
    dwTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    // Expected dy size: ((32 + 1 + 1 - 3) / 1) + 1 = 32, but we have 28
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionWgradNode, InferPropertiesMissingXStrides)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    // No stride set on x tensor
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    // No stride set on dw tensor - will try to infer from x
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionWgradNode, InferPropertiesStrideDimensionMismatch)
{
    ConvWgradAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32}); // Only 3 strides for 4D tensor
    convAttributes.set_x(xTensor);

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto dwTensor = std::make_shared<TensorAttributes>();
    dwTensor->set_dim({64, 3, 3, 3});
    // No stride set on dw tensor - will try to infer from x
    convAttributes.set_dw(dwTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionWgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}
