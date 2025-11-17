// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/node/ConvolutionDgradNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestConvolutionDgradNode, PreValidateNode)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionDgradNode, PreValidateNodeMissingDyTensor)
{
    ConvDgradAttributes convAttributes;

    // Set w tensor with proper dims and strides
    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    // Set dx tensor with proper dims and strides
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // Dy tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, PreValidateNodeMissingWTensor)
{
    ConvDgradAttributes convAttributes;

    // Set dy tensor with proper dims and strides
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    // Set dx tensor with proper dims and strides
    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // W tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, PreValidateNodeMissingDxTensor)
{
    ConvDgradAttributes convAttributes;

    // Set dy tensor with proper dims and strides
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    // Set w tensor with proper dims and strides
    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // Dx tensor is missing
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, PreValidateNodeMissingConvolutionParameters)
{
    ConvDgradAttributes convAttributes;

    // Set all tensors with proper dims and strides
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    // Convolution parameters are missing
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, PreValidateNodeAllValuesSet)
{
    ConvDgradAttributes convAttributes;

    // Set all tensors with proper dims and strides
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    dxTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionDgradNode, InferPropertiesNodeMissingDyTensor)
{
    ConvDgradAttributes convAttributes;
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, InferPropertiesNodeMissingWTensor)
{
    ConvDgradAttributes convAttributes;
    convAttributes.set_dy(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, InferPropertiesNodeMissingDxTensor)
{
    ConvDgradAttributes convAttributes;
    convAttributes.set_dy(std::make_shared<TensorAttributes>());
    convAttributes.set_w(std::make_shared<TensorAttributes>());
    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionDgradNode, InferPropertiesNode2DConvolutionSuccess)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // For backward pass: dx_size = stride * (dy_size - 1) + dilated_kernel_size - pre_pad - post_pad
    // dx_size = 1 * (32 - 1) + 3 - 1 - 1 = 31 + 3 - 2 = 32
    EXPECT_EQ(inferredDims[2], 32); // Height
    EXPECT_EQ(inferredDims[3], 32); // Width
}

TEST(TestConvolutionDgradNode, InferPropertiesNode3DConvolutionSuccess)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 32, 6, 16, 16});
    dyTensor->set_stride({49152, 1536, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({32, 16, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({0, 1, 1});
    convAttributes.set_post_padding({0, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 16); // Input channels
    // Depth: 1 * (6 - 1) + 3 - 0 - 0 = 5 + 3 - 0 = 8
    EXPECT_EQ(inferredDims[2], 8);
    // Height: 1 * (16 - 1) + 3 - 1 - 1 = 15 + 3 - 2 = 16
    EXPECT_EQ(inferredDims[3], 16);
    // Width: 1 * (16 - 1) + 3 - 1 - 1 = 15 + 3 - 2 = 16
    EXPECT_EQ(inferredDims[4], 16);
}

TEST(TestConvolutionDgradNode, InferPropertiesNodeWithStride2x2)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 16, 16});
    dyTensor->set_stride({16384, 256, 16, 1}); // NCHW layout
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({2, 2}); // 2x2 stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // Height: 2 * (16 - 1) + 3 - 1 - 1 = 30 + 3 - 2 = 31
    EXPECT_EQ(inferredDims[2], 31);
    // Width: 2 * (16 - 1) + 3 - 1 - 1 = 30 + 3 - 2 = 31
    EXPECT_EQ(inferredDims[3], 31);

    // Check inferred strides
    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 2883); // N stride: 3 * 31 * 31 = 2883
    EXPECT_EQ(inferredStrides[1], 961); // C stride: 31 * 31 = 961
    EXPECT_EQ(inferredStrides[2], 31); // H stride: 31
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, InferPropertiesNodeWithDilation2x2)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 32, 20, 20});
    dyTensor->set_stride({12800, 400, 20, 1}); // NCHW layout
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({32, 16, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 16); // Input channels
    // Effective kernel size with dilation: (3-1)*2 + 1 = 5
    // Height: 1 * (20 - 1) + 5 - 2 - 2 = 19 + 5 - 4 = 20
    EXPECT_EQ(inferredDims[2], 20);
    // Width: 1 * (20 - 1) + 5 - 2 - 2 = 19 + 5 - 4 = 20
    EXPECT_EQ(inferredDims[3], 20);
}

TEST(TestConvolutionDgradNode, PackNode)
{
    ConvDgradAttributes convAttributes;
    convAttributes.set_name("ConvolutionDgrad");

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(1)
        .set_name("DyTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 64, 32, 32})
        .set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_uid(2)
        .set_name("WTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({64, 3, 3, 3})
        .set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(3)
        .set_name("DxTensor")
        .set_data_type(DataType_t::FLOAT)
        .set_dim({1, 3, 32, 32})
        .set_stride({3072, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});
    convAttributes.set_convolution_mode(hipdnn_frontend::ConvolutionMode_t::CROSS_CORRELATION);

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "ConvolutionDgrad");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_sdk::data_objects::NodeAttributes::ConvolutionBwdAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_ConvolutionBwdAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->dy_tensor_uid(), dyTensor->get_uid());
    EXPECT_EQ(packedAttributes->w_tensor_uid(), wTensor->get_uid());
    EXPECT_EQ(packedAttributes->dx_tensor_uid(), dxTensor->get_uid());

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

TEST(TestConvolutionDgradNode, GatherHipdnnTensors)
{
    ConvDgradAttributes convAttributes;
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_uid(1).set_name("DyTensor");
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_uid(2).set_name("WTensor");
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_uid(3).set_name("DxTensor");
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(dyTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(wTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(dxTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3);
}

TEST(TestConvolutionDgradNode, StrideInferenceNchwLayoutSuccess)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1}); // NCHW layout
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32});
    // No stride set - should be inferred
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // NCHW strides
    EXPECT_EQ(inferredStrides[0], 3072); // N stride: 3 * 32 * 32
    EXPECT_EQ(inferredStrides[1], 1024); // C stride: 32 * 32
    EXPECT_EQ(inferredStrides[2], 32); // H stride: 32
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, StrideInferenceNhwcLayoutSuccess)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32}); // Dims in NCHW order
    dyTensor->set_stride({65536, 1, 2048, 64}); // NHWC strides (channels last)
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({1, 3, 32, 32}); // Dims in NCHW order
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // NHWC strides (channels last)
    EXPECT_EQ(inferredStrides[0], 3072); // N stride: 32 * 32 * 3
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 96); // H stride: 32 * 3
    EXPECT_EQ(inferredStrides[3], 3); // W stride: 3
}

TEST(TestConvolutionDgradNode, InferPropertiesGroupedConv2Groups)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // 128 output channels
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 32, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 32); // Input channels (assume 1 group)
    EXPECT_EQ(inferredDims[2], 32); // Height
    EXPECT_EQ(inferredDims[3], 32); // Width
}

TEST(TestConvolutionDgradNode, PreValidateTensorDimsTooFew)
{
    ConvDgradAttributes convAttributes;

    // Test with too few dimensions (less than 3)
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64}); // Only 2 dimensions
    dyTensor->set_stride({64, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1});
    convAttributes.set_post_padding({1});
    convAttributes.set_stride({1});
    convAttributes.set_dilation({1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateWeightDimsMismatch)
{
    ConvDgradAttributes convAttributes;

    // Test with mismatched weight dimensions
    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3, 3}); // 5D weight for 4D dy
    wTensor->set_stride({81, 27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateOutputChannelMismatch)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // 128 output channels
    dyTensor->set_stride({131072, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3}); // 64 output channels, doesn't match dy
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateSpatialParamMismatch)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32, 32}); // 3D spatial
    dyTensor->set_stride({2097152, 32768, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3, 3});
    wTensor->set_stride({81, 27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    // Only 2 spatial parameters for 3D spatial dimensions
    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateNegativeStride)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, -1}); // Negative stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateZeroDilation)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 0}); // Zero dilation

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateNegativePrePadding)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({-1, 1}); // Negative padding
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, InferPropertiesWithLargeStride)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 8, 8}); // Small dy output
    dyTensor->set_stride({4096, 64, 8, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 5, 5}); // 5x5 kernel
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({3, 3}); // Large stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // Height: 3 * (8 - 1) + 5 - 2 - 2 = 21 + 5 - 4 = 22
    EXPECT_EQ(inferredDims[2], 22);
    // Width: 3 * (8 - 1) + 5 - 2 - 2 = 21 + 5 - 4 = 22
    EXPECT_EQ(inferredDims[3], 22);
}

TEST(TestConvolutionDgradNode, InferPropertiesZeroPadding)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 28, 28});
    dyTensor->set_stride({50176, 784, 28, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 5, 5}); // 5x5 kernel
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({0, 0}); // No padding
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 3); // Input channels
    // Height: 1 * (28 - 1) + 5 - 0 - 0 = 27 + 5 = 32
    EXPECT_EQ(inferredDims[2], 32);
    // Width: 1 * (28 - 1) + 5 - 0 - 0 = 27 + 5 = 32
    EXPECT_EQ(inferredDims[3], 32);
}

TEST(TestConvolutionDgradNode, InferPropertiesAsymmetricPadding)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 15, 15});
    dyTensor->set_stride({14400, 225, 15, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({0, 1}); // Asymmetric padding
    convAttributes.set_post_padding({1, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = dxTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 32); // Input channels
    // Height: 1 * (15 - 1) + 3 - 0 - 1 = 14 + 3 - 1 = 16
    EXPECT_EQ(inferredDims[2], 16);
    // Width: 1 * (15 - 1) + 3 - 1 - 0 = 14 + 3 - 1 = 16
    EXPECT_EQ(inferredDims[3], 16);
}

TEST(TestConvolutionDgradNode, PreValidateGroupedConvInvalidOutputChannels)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 63, 32, 32}); // 63 output channels
    dyTensor->set_stride({64512, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({63, 32, 3, 3}); // 32 weight channels, 63 output channels (not divisible)
    wTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // C_in = 64, C_in/G = 32 -> G = 2
    dxTensor->set_dim({1, 64, 32, 32});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    // wDims[0] % groupCount (63 % 2) will fail
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, PreValidateGroupedConv2GroupsWithDxSet)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32}); // 64 output channels
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 32 input channels per group
    wTensor->set_dim({64, 32, 3, 3});
    wTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 64 input channels total, so groups = 64/32 = 2
    dxTensor->set_dim({1, 64, 32, 32});
    dxTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionDgradNode, PreValidateGroupedConv4GroupsWithDxSet)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 128, 16, 16}); // 128 output channels
    dyTensor->set_stride({32768, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // 128 output channels, 16 input channels per group
    wTensor->set_dim({128, 16, 3, 3});
    wTensor->set_stride({144, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 64 input channels total, so groups = 64/16 = 4
    dxTensor->set_dim({2, 64, 16, 16});
    dxTensor->set_stride({16384, 256, 16, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionDgradNode, PreValidateGroupedConvInvalidInputChannels)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 32, 32});
    dyTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 3, 3});
    wTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 63 input channels is not divisible by 32
    dxTensor->set_dim({1, 63, 32, 32});
    dxTensor->set_stride({64512, 1024, 32, 1});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionDgradNode, InferGroupedConvStrideInferenceNchwLayout)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32});
    dyTensor->set_stride({131072, 1024, 32, 1}); // NCHW layout
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 32, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // Set dimensions to establish groups = 2
    dxTensor->set_dim({1, 64, 32, 32});
    // No stride set - should be inferred
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 65536); // N stride: 64 * 32 * 32
    EXPECT_EQ(inferredStrides[1], 1024); // C stride: 32 * 32
    EXPECT_EQ(inferredStrides[2], 32); // H stride: 32
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, InferGroupedConvStrideInferenceNhwcLayout)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 128, 32, 32}); // Dims in NCHW order
    dyTensor->set_stride({131072, 1, 4096, 128}); // NHWC strides (channels last)
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 32, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // Set dimensions to establish groups = 2
    dxTensor->set_dim({1, 64, 32, 32}); // Dims in NCHW order
    // No stride set - should be inferred
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 65536); // N stride: 32 * 32 * 64
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 2048); // H stride: 32 * 64
    EXPECT_EQ(inferredStrides[3], 64); // W stride: 64
}

TEST(TestConvolutionDgradNode, InferGroupedConvWithDilation)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 20, 20});
    dyTensor->set_stride({25600, 400, 20, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 16 input channels per group
    wTensor->set_dim({64, 16, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 32 input channels total, so groups = 32/16 = 2
    dxTensor->set_dim({1, 32, 20, 20});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // NCHW strides
    EXPECT_EQ(inferredStrides[0], 12800); // N stride: 32 * 20 * 20
    EXPECT_EQ(inferredStrides[1], 400); // C stride: 20 * 20
    EXPECT_EQ(inferredStrides[2], 20); // H stride: 20
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, InferGroupedConvWithLargeStride)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 128, 8, 8});
    dyTensor->set_stride({8192, 64, 8, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // 128 output channels, 8 input channels per group
    wTensor->set_dim({128, 8, 5, 5});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 64 input channels total, so groups = 64/8 = 8
    dxTensor->set_dim({2, 64, 22, 22});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({3, 3}); // Large stride
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 30976); // N stride: 64 * 22 * 22
    EXPECT_EQ(inferredStrides[1], 484); // C stride: 22 * 22
    EXPECT_EQ(inferredStrides[2], 22); // H stride: 22
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, InferGroupedConv3D)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 8, 16, 16});
    dyTensor->set_stride({131072, 2048, 256, 16, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // 64 output channels, 8 input channels per group
    wTensor->set_dim({64, 8, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 32 input channels total, so groups = 32/8 = 4
    dxTensor->set_dim({2, 32, 8, 16, 16});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 5);
    EXPECT_EQ(inferredStrides[0], 65536); // N stride: 32 * 8 * 16 * 16
    EXPECT_EQ(inferredStrides[1], 2048); // C stride: 8 * 16 * 16
    EXPECT_EQ(inferredStrides[2], 256); // D stride: 16 * 16
    EXPECT_EQ(inferredStrides[3], 16); // H stride: 16
    EXPECT_EQ(inferredStrides[4], 1); // W stride: 1
}

TEST(TestConvolutionDgradNode, InferGroupedConvDepthwiseSeparable)
{
    ConvDgradAttributes convAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 32, 112, 112}); // 32 output channels
    dyTensor->set_stride({401408, 12544, 112, 1});
    convAttributes.set_dy(dyTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    // Depthwise: 32 output channels, 1 input channel per group (32 groups)
    wTensor->set_dim({32, 1, 3, 3});
    wTensor->set_stride({9, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    // 32 input channels total, so groups = 32/1 = 32
    dxTensor->set_dim({1, 32, 112, 112});
    convAttributes.set_dx(dxTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    GraphAttributes graphAttributes;
    ConvolutionDgradNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredStrides = dxTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 401408); // N stride
    EXPECT_EQ(inferredStrides[1], 12544); // C stride
    EXPECT_EQ(inferredStrides[2], 112); // H stride
    EXPECT_EQ(inferredStrides[3], 1); // W stride
}
