// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/node/ConvolutionFpropNode.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "fake_backend/BackendTestMatchers.hpp"
#include "fake_backend/MockHipdnnBackend.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using namespace ::testing;
using namespace hipdnn_frontend::test;

TEST(TestConvolutionNode, PreValidateNode)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionNode, PreValidateNodeMissingXTensor)
{
    ConvFpropAttributes convAttributes;

    // Set w tensor with proper dims and strides
    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    // Set y tensor with proper dims and strides
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    yTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // X tensor is missing
    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, PreValidateNodeMissingWTensor)
{
    ConvFpropAttributes convAttributes;

    // Set x tensor with proper dims and strides
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    // Set y tensor with proper dims and strides
    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    yTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // W tensor is missing
    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, PreValidateNodeMissingYTensor)
{
    ConvFpropAttributes convAttributes;

    // Set x tensor with proper dims and strides
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    // Set w tensor with proper dims and strides
    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    // Y tensor is missing
    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, PreValidateNodeMissingConvolutionParameters)
{
    ConvFpropAttributes convAttributes;

    // Set all tensors with proper dims and strides
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    yTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_y(yTensor);

    // Convolution parameters are missing
    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, PreValidateNodeAllValuesSet)
{
    ConvFpropAttributes convAttributes;

    // Set all tensors with proper dims and strides
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    yTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionNode, InferPropertiesNodeMissingXTensor)
{
    ConvFpropAttributes convAttributes;
    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, InferPropertiesNodeMissingWTensor)
{
    ConvFpropAttributes convAttributes;
    convAttributes.set_x(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, InferPropertiesNodeMissingYTensor)
{
    ConvFpropAttributes convAttributes;
    convAttributes.set_x(std::make_shared<TensorAttributes>());
    convAttributes.set_w(std::make_shared<TensorAttributes>());
    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, InferPropertiesNode2DConvolutionSuccess)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 32); // Height: (32 + 1 + 1 - 3) / 1 + 1 = 32
    EXPECT_EQ(inferredDims[3], 32); // Width: (32 + 1 + 1 - 3) / 1 + 1 = 32
}

TEST(TestConvolutionNode, InferPropertiesNode3DConvolutionSuccess)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 16, 8, 16, 16});
    xTensor->set_stride({32768, 2048, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({32, 16, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 1, 1});
    convAttributes.set_post_padding({0, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 32); // Output channels
    EXPECT_EQ(inferredDims[2], 6); // Depth: (8 + 0 + 0 - 3) / 1 + 1 = 6
    EXPECT_EQ(inferredDims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferredDims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16
}

TEST(TestConvolutionNode, InferPropertiesNodeInsufficientSpatialParameters)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1}); // Missing padding for second spatial dim
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeInvalidStrideValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({0, 1}); // Invalid stride
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeInvalidDilationValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 0}); // Invalid dilation

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeNegativeOutputSize)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 3, 3}); // Small input
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 5, 5}); // Large kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, StrideInferenceMissingInputStrides)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    // No stride set on input tensor
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32}); // Pre-set dimensions
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, StrideInferenceMissingOutputDimensions)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    yTensor->set_dim({1}); // check that inferring fails when dims don't match x.
    EXPECT_EQ(node.infer_properties_node(), error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, StrideInferenceDimensionMismatch)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32}); // Missing one stride dimension
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::ATTRIBUTE_NOT_SET);
}

TEST(TestConvolutionNode, StrideInferenceNchwLayoutSuccess)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    // No stride set - should be inferred
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // NCHW strides
    EXPECT_EQ(inferredStrides[0], 65536); // N stride: 64 * 32 * 32
    EXPECT_EQ(inferredStrides[1], 1024); // C stride: 32 * 32
    EXPECT_EQ(inferredStrides[2], 32); // H stride: 32
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceNhwcLayoutSuccess)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32}); // Dims in NCHW order
    xTensor->set_stride({3072, 1, 96, 3}); // NHWC strides (channels last)
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32}); // Dims in NCHW order
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    // NHWC strides (channels last)
    EXPECT_EQ(inferredStrides[0], 65536); // N stride: 32 * 32 * 64
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 2048); // H stride: 32 * 64
    EXPECT_EQ(inferredStrides[3], 64); // W stride: 64
}

TEST(TestConvolutionNode, StrideInferencePreExistingStridesNotOverwritten)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 32, 32});
    yTensor->set_stride({65536, 1024, 32, 1}); // Pre-existing strides
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
    ;

    auto finalStrides = yTensor->get_stride();
    // Should keep the pre-existing strides
    EXPECT_EQ(finalStrides[0], 65536);
    EXPECT_EQ(finalStrides[1], 1024);
    EXPECT_EQ(finalStrides[2], 32);
    EXPECT_EQ(finalStrides[3], 1);
}

TEST(TestConvolutionNode, StrideInferenceWithStride2x2)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    // No dimensions or strides set - should be inferred
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({2, 2}); // 2x2 stride
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 16); // Height: (32 + 1 + 1 - 3) / 2 + 1 = 16
    EXPECT_EQ(inferredDims[3], 16); // Width: (32 + 1 + 1 - 3) / 2 + 1 = 16

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);

    // Expected strides for 16x16 output with NCHW layout
    EXPECT_EQ(inferredStrides[0], 16384); // N stride: 64 * 16 * 16 = 16384
    EXPECT_EQ(inferredStrides[1], 256); // C stride: 16 * 16 = 256
    EXPECT_EQ(inferredStrides[2], 16); // H stride: 16
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, GatherHipdnnTensor)
{
    ConvFpropAttributes convAttributes;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_uid(2).set_name("WTensor");
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(3).set_name("YTensor");
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;

    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(wTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 3);
}

TEST(TestConvolutionNode, StrideInferenceWithLargeKernel5x5)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 28, 28});
    xTensor->set_stride({50176, 784, 28, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 64, 5, 5}); // 5x5 kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 128); // Output channels
    EXPECT_EQ(inferredDims[2], 28); // Height: (28 + 2 + 2 - 5) / 1 + 1 = 28
    EXPECT_EQ(inferredDims[3], 28); // Width: (28 + 2 + 2 - 5) / 1 + 1 = 28

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 100352); // N stride: 128 * 28 * 28 = 100352
    EXPECT_EQ(inferredStrides[1], 784); // C stride: 28 * 28 = 784
    EXPECT_EQ(inferredStrides[2], 28); // H stride: 28
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceWithAsymmetricPadding)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 32, 16, 16});
    xTensor->set_stride({8192, 256, 16, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 1}); // Asymmetric padding
    convAttributes.set_post_padding({1, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 15); // Height: (16 + 0 + 1 - 3) / 1 + 1 = 15
    EXPECT_EQ(inferredDims[3], 15); // Width: (16 + 1 + 0 - 3) / 1 + 1 = 15

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 14400); // N stride: 64 * 15 * 15 = 14400
    EXPECT_EQ(inferredStrides[1], 225); // C stride: 15 * 15 = 225
    EXPECT_EQ(inferredStrides[2], 15); // H stride: 15
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceWithDilation2x2)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 16, 20, 20});
    xTensor->set_stride({6400, 400, 20, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({32, 16, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({2, 2}); // 2x2 dilation

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 32); // Output channels
    // Effective kernel size with dilation: (3-1)*2 + 1 = 5
    EXPECT_EQ(inferredDims[2], 20); // Height: (20 + 2 + 2 - 5) / 1 + 1 = 20
    EXPECT_EQ(inferredDims[3], 20); // Width: (20 + 2 + 2 - 5) / 1 + 1 = 20

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 12800); // N stride: 32 * 20 * 20 = 10368
    EXPECT_EQ(inferredStrides[1], 400); // C stride: 20 * 20 = 400
    EXPECT_EQ(inferredStrides[2], 20); // H stride: 20
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceWithStride3x3AndLargeKernel)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 224, 224});
    xTensor->set_stride({150528, 50176, 224, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 7, 7}); // 7x7 kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({3, 3});
    convAttributes.set_post_padding({3, 3});
    convAttributes.set_stride({3, 3}); // 3x3 stride
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 75); // Height: (224 + 3 + 3 - 7) / 3 + 1 = 75
    EXPECT_EQ(inferredDims[3], 75); // Width: (224 + 3 + 3 - 7) / 3 + 1 = 75

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 360000); // N stride: 64 * 75 * 75 = 360000
    EXPECT_EQ(inferredStrides[1], 5625); // C stride: 75 * 75 = 5625
    EXPECT_EQ(inferredStrides[2], 75); // H stride: 75
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceWith1x1ConvolutionNoPadding)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 14, 14});
    xTensor->set_stride({50176, 196, 14, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({512, 256, 1, 1}); // 1x1 kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 0}); // No padding
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 512); // Output channels
    EXPECT_EQ(inferredDims[2], 14); // Height: (14 + 0 + 0 - 1) / 1 + 1 = 14
    EXPECT_EQ(inferredDims[3], 14); // Width: (14 + 0 + 0 - 1) / 1 + 1 = 14

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 100352); // N stride: 512 * 14 * 14 = 100352
    EXPECT_EQ(inferredStrides[1], 196); // C stride: 14 * 14 = 196
    EXPECT_EQ(inferredStrides[2], 14); // H stride: 14
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInference3DConvolutionWithDilation)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 32, 16, 16, 16});
    xTensor->set_stride({131072, 4096, 256, 16, 1}); // NCDHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 3, 3, 3}); // 3x3x3 kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({2, 1, 1}); // Dilation only in depth dimension

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 14); // Depth: (16 + 1 + 1 - 5) / 1 + 1 = 14
    EXPECT_EQ(inferredDims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferredDims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 5);
    EXPECT_EQ(inferredStrides[0], 229376); // N stride: 64 * 14 * 16 * 16 = 229376
    EXPECT_EQ(inferredStrides[1], 3584); // C stride: 14 * 16 * 16 = 3584
    EXPECT_EQ(inferredStrides[2], 256); // D stride: 16 * 16 = 256
    EXPECT_EQ(inferredStrides[3], 16); // H stride: 16
    EXPECT_EQ(inferredStrides[4], 1); // W stride: 1
}

TEST(TestConvolutionNode, StrideInferenceWithNhwcLayoutAndComplexParams)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 128, 56, 56}); // Dims in NCHW order
    xTensor->set_stride({401408, 1, 7168, 128}); // NHWC strides (channels last)
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({56, 128, 5, 5}); // Dims in KCRS order
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 2});
    convAttributes.set_post_padding({2, 1});
    convAttributes.set_stride({2, 2});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 56); // Output channels
    EXPECT_EQ(inferredDims[2], 28); // Height: (56 + 1 + 2 - 5) / 2 + 1 = 28
    EXPECT_EQ(inferredDims[3], 28); // Width: (56 + 2 + 1 - 5) / 2 + 1 = 28

    // Check inferred strides (NHWC - channels last)
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 43904); // N stride: 28 * 28 * 56
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 1568); // H stride: 28 * 56
    EXPECT_EQ(inferredStrides[3], 56); // W stride: 56
}

TEST(TestConvolutionNode, StrideInferenceDepthwiseConvolution)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 32, 112, 112});
    xTensor->set_stride({401408, 12544, 112, 1}); // NCHW layout
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({32, 32, 3, 3}); // Depthwise 3x3 kernel
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({2, 2});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 32); // Output channels (same as input for depthwise)
    EXPECT_EQ(inferredDims[2], 56); // Height: (112 + 1 + 1 - 3) / 2 + 1 = 56
    EXPECT_EQ(inferredDims[3], 56); // Width: (112 + 1 + 1 - 3) / 2 + 1 = 56

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 100352); // N stride: 32 * 56 * 56 = 100352
    EXPECT_EQ(inferredStrides[1], 3136); // C stride: 56 * 56 = 3136
    EXPECT_EQ(inferredStrides[2], 56); // H stride: 56
    EXPECT_EQ(inferredStrides[3], 1); // W stride: 1
}

TEST(TestConvolutionNode, PreValidateTensorDimsEmpty)
{
    ConvFpropAttributes convAttributes;

    // Test with empty input dimensions
    auto xTensor = std::make_shared<TensorAttributes>();
    // No dimensions set
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateTensorDimsTooFew)
{
    ConvFpropAttributes convAttributes;

    // Test with too few dimensions (less than 3)
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3}); // Only 2 dimensions
    xTensor->set_stride({3, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1});
    convAttributes.set_post_padding({1});
    convAttributes.set_stride({1});
    convAttributes.set_dilation({1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateWeightDimsMismatch)
{
    ConvFpropAttributes convAttributes;

    // Test with mismatched weight dimensions
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3, 3}); // 5D weight for 4D input
    wTensor->set_stride({81, 27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateSpatialParamMismatch)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32, 32}); // 3D spatial
    xTensor->set_stride({98304, 32768, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3, 3});
    wTensor->set_stride({81, 27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    // Only 2 spatial parameters for 3D spatial dimensions
    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateGroupedConvInputChannelNotDivisible)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 48, 3, 3}); // 48 is not a divisor of 64
    wTensor->set_stride({432, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateGroupedConvOutputChannelNotDivisible)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 3, 3}); // Valid grouped setup
    wTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 96, 32, 32}); // 96 is not a multiple of 64
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesGroupedConv2Groups)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 32, 3, 3}); // 32 input channels per group, 128 output channels
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 128); // Output channels
    EXPECT_EQ(inferredDims[2], 32); // Height
    EXPECT_EQ(inferredDims[3], 32); // Width

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 131072); // N stride: 128 * 32 * 32
    EXPECT_EQ(inferredStrides[1], 1024); // C stride: 32 * 32
    EXPECT_EQ(inferredStrides[2], 32); // H stride 32
    EXPECT_EQ(inferredStrides[3], 1); // W stride 1
}

TEST(TestConvolutionNode, InferPropertiesGroupedConv4Groups)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 128, 16, 16}); // 128 input channels
    xTensor->set_stride({32768, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 32, 5, 5}); // 32 input channels per group (4 groups)
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({2, 2});
    convAttributes.set_post_padding({2, 2});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 2); // Batch size
    EXPECT_EQ(inferredDims[1], 64); // Output channels
    EXPECT_EQ(inferredDims[2], 16); // Height
    EXPECT_EQ(inferredDims[3], 16); // Width

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 16384); // N stride: 64 * 16 * 16
    EXPECT_EQ(inferredStrides[1], 256); // C stride: 16 * 16
    EXPECT_EQ(inferredStrides[2], 16); // H stride
    EXPECT_EQ(inferredStrides[3], 1); // W stride
}

TEST(TestConvolutionNode, InferPropertiesGroupedConvWithStride)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 96, 112, 112}); // 96 input channels
    xTensor->set_stride({1204224, 12544, 112, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({256, 48, 3, 3}); // 48 input channels per group (2 groups)
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({2, 2}); // Stride 2x2
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 256); // Output channels: 256
    EXPECT_EQ(inferredDims[2], 56); // Height: (112 + 1 + 1 - 3) / 2 + 1 = 56
    EXPECT_EQ(inferredDims[3], 56); // Width

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 802816); // N stride: 256 * 56 * 56
    EXPECT_EQ(inferredStrides[1], 3136); // C stride: 56 * 56
    EXPECT_EQ(inferredStrides[2], 56); // H stride
    EXPECT_EQ(inferredStrides[3], 1); // W stride
}

TEST(TestConvolutionNode, InferPropertiesGroupedConvNhwcLayout)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 28, 28}); // Dims in NCHW order
    xTensor->set_stride({50176, 1, 1792, 64}); // NHWC strides (channels last)
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 16, 3, 3}); // Dims in KCRS order, 16 input channels per group (4 groups)
    wTensor->set_stride({144, 1, 48, 16}); // KRSC strides (channels last)
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 4);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 128); // Output channels
    EXPECT_EQ(inferredDims[2], 28); // Height: (28 + 1 + 1 - 3) / 1 + 1 = 28
    EXPECT_EQ(inferredDims[3], 28); // Width: (28 + 1 + 1 - 3) / 1 + 1 = 28

    // Check inferred strides maintain NHWC layout (channels last)
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 4);
    EXPECT_EQ(inferredStrides[0], 100352); // N stride: 28 * 28 * 128
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 3584); // H stride: 28 * 128
    EXPECT_EQ(inferredStrides[3], 128); // W stride: 128
}

TEST(TestConvolutionNode, InferPropertiesGroupedConv3D)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 48, 8, 16, 16}); // 48 input channels
    xTensor->set_stride({98304, 2048, 256, 16, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({96, 16, 3, 3, 3}); // 16 input channels per group (3 groups)
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 96); // Output channels
    EXPECT_EQ(inferredDims[2], 8); // Depth
    EXPECT_EQ(inferredDims[3], 16); // Height
    EXPECT_EQ(inferredDims[4], 16); // Width

    // Check inferred strides
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 5);
    EXPECT_EQ(inferredStrides[0], 196608); // N stride: 96 * 8 * 16 * 16
    EXPECT_EQ(inferredStrides[1], 2048); // C stride: 8 * 16 * 16
    EXPECT_EQ(inferredStrides[2], 256); // D stride: 16 * 16 = 256
    EXPECT_EQ(inferredStrides[3], 16); // H stride: 16
    EXPECT_EQ(inferredStrides[4], 1); // W stride: 1
}

TEST(TestConvolutionNode, InferPropertiesGroupedConv3DNhwcLayout)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 48, 8, 16, 16}); // Dims in NCDHW order
    xTensor->set_stride({98304, 1, 12288, 768, 48}); // NDHWC strides (channels last)
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim(
        {96, 16, 3, 3, 3}); // Dims in KCDHW order, 16 input channels per group (3 groups)
    wTensor->set_stride({432, 1, 144, 48, 16}); // KDHWC strides (channels last)
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1, 1});
    convAttributes.set_post_padding({1, 1, 1});
    convAttributes.set_stride({1, 1, 1});
    convAttributes.set_dilation({1, 1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;

    // Check inferred dimensions
    auto inferredDims = yTensor->get_dim();
    EXPECT_EQ(inferredDims.size(), 5);
    EXPECT_EQ(inferredDims[0], 1); // Batch size
    EXPECT_EQ(inferredDims[1], 96); // Output channels
    EXPECT_EQ(inferredDims[2], 8); // Depth: (8 + 1 + 1 - 3) / 1 + 1 = 8
    EXPECT_EQ(inferredDims[3], 16); // Height: (16 + 1 + 1 - 3) / 1 + 1 = 16
    EXPECT_EQ(inferredDims[4], 16); // Width: (16 + 1 + 1 - 3) / 1 + 1 = 16

    // Check inferred strides maintain NDHWC layout (channels last)
    auto inferredStrides = yTensor->get_stride();
    EXPECT_EQ(inferredStrides.size(), 5);
    EXPECT_EQ(inferredStrides[0], 196608); // N stride: 8 * 16 * 16 * 96
    EXPECT_EQ(inferredStrides[1], 1); // C stride: 1 (channels last)
    EXPECT_EQ(inferredStrides[2], 24576); // D stride: 16 * 16 * 96
    EXPECT_EQ(inferredStrides[3], 1536); // H stride: 16 * 96
    EXPECT_EQ(inferredStrides[4], 96); // W stride: 96
}

TEST(TestConvolutionNode, PreValidateOutputDimsMismatchBatch)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 64, 3, 3});
    wTensor->set_stride({576, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 128, 32, 32}); // Wrong batch size
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateOutputDimsWrongCount)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32});
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({128, 64, 3, 3});
    wTensor->set_stride({576, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 128, 32, 32, 1}); // 5D output for 4D input
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeNegativeStrideValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({-1, 1}); // Negative stride in first dimension
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeNegativeDilationValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, -1}); // Negative dilation in second dimension

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeNegativePrePaddingValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({-1, 1}); // Negative pre-padding in first dimension
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, InferPropertiesNodeNegativePostPaddingValues)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, -2}); // Negative post-padding
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeNegativePrePadding)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({-1, 1}); // Negative padding
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeNegativePostPadding)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, -2}); // Negative post padding
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeZeroStride)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({0, 1}); // Zero stride
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeNegativeStride)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, -1}); // Negative stride
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeZeroDilation)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 0}); // Zero dilation

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeNegativeDilation)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({-2, 1}); // Negative dilation

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeBoundaryValueZeroPadding)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    // Zero padding is valid (boundary case)
    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionNode, PreValidateGroupedConvInvalidOutputChannels)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 32, 32}); // 64 input channels
    xTensor->set_stride({65536, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({63, 32, 3, 3}); // 32 input channels per group, 2 groups, 63 output channels
    wTensor->set_stride({288, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeInvalidOutputDimensions)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 32, 32});
    xTensor->set_stride({3072, 1024, 32, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 30, 30}); // Incorrect output size (should be 32x32)
    yTensor->set_stride({57600, 900, 30, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({1, 1});
    convAttributes.set_post_padding({1, 1});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

TEST(TestConvolutionNode, PreValidateNodeWithDroppedPixels)
{
    ConvFpropAttributes convAttributes;

    // For each spatial dim Index (0, 1, 2) & (2, 3, 4) are the two 3x3 kernels
    // applied with stride 2 on 6x6 dx 5th is dropped / unused due to stride
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 6, 6});
    xTensor->set_stride({108, 36, 6, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3});
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 2, 2});
    yTensor->set_stride({256, 4, 2, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({2, 2});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::OK) << error.err_msg;
}

TEST(TestConvolutionNode, PreValidateNodeInputTooSmall)
{
    ConvFpropAttributes convAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 3, 2, 2}); // Input smaller than kernel
    xTensor->set_stride({12, 4, 2, 1});
    convAttributes.set_x(xTensor);

    auto wTensor = std::make_shared<TensorAttributes>();
    wTensor->set_dim({64, 3, 3, 3}); // 3x3 kernel
    wTensor->set_stride({27, 9, 3, 1});
    convAttributes.set_w(wTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({1, 64, 1, 1}); // Even if output is 1x1, it's invalid
    yTensor->set_stride({64, 1, 1, 1});
    convAttributes.set_y(yTensor);

    convAttributes.set_pre_padding({0, 0});
    convAttributes.set_post_padding({0, 0});
    convAttributes.set_stride({1, 1});
    convAttributes.set_dilation({1, 1});

    const GraphAttributes graphAttributes;
    const ConvolutionFpropNode node(std::move(convAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, error_code_t::INVALID_VALUE);
}

// =============================================================================
// create_operation() Tests
// =============================================================================

class TestConvolutionNodeCreateOperation : public ::testing::Test
{
protected:
    std::shared_ptr<Mock_hipdnn_backend> _mockBackend;

    void SetUp() override
    {
        _mockBackend = std::make_shared<Mock_hipdnn_backend>();
        detail::IHipdnnBackend::setInstance(_mockBackend);
    }
    void TearDown() override
    {
        detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    static ConvolutionFpropNode makeConvNode()
    {
        ConvFpropAttributes attrs;
        auto x = std::make_shared<TensorAttributes>();
        x->set_uid(K_FPROP_TENSOR_X_UID)
            .set_name("X")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_FPROP_TENSOR_X_DIMS))
            .set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));
        auto w = std::make_shared<TensorAttributes>();
        w->set_uid(K_FPROP_TENSOR_W_UID)
            .set_name("W")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_FPROP_TENSOR_W_DIMS))
            .set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));
        auto y = std::make_shared<TensorAttributes>();
        y->set_uid(K_FPROP_TENSOR_Y_UID)
            .set_name("Y")
            .set_data_type(DataType::FLOAT)
            .set_dim(toVec(K_FPROP_TENSOR_Y_DIMS))
            .set_stride(toVec(K_FPROP_TENSOR_Y_STRIDES));

        attrs.set_x(x);
        attrs.set_w(w);
        attrs.set_y(y);
        attrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
        attrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
        attrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
        attrs.set_dilation(toVec(K_FPROP_CONV_DILATION));
        attrs.compute_data_type = DataType::FLOAT;

        const GraphAttributes graphAttrs;
        return {std::move(attrs), graphAttrs};
    }
};

TEST_F(TestConvolutionNodeCreateOperation, PropagatesBackendError)
{
    // Tensor creation succeeds, but operation creation fails
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_INTERNAL_ERROR));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendFinalize(_)).WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, getLastErrorString(_, _)).Times(AnyNumber());

    auto node = makeConvNode();
    std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<detail::ScopedHipdnnBackendDescriptor> operations;

    auto err = node.create_operation(tensorDescs, operations);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
}

TEST_F(TestConvolutionNodeCreateOperation, SuccessCreatesThreeTensorsAndOneOperation)
{
    // Expect 3 tensor descriptors + 1 operation descriptor created
    EXPECT_CALL(*_mockBackend, backendCreateDescriptor(HIPDNN_BACKEND_TENSOR_DESCRIPTOR, _))
        .Times(3)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend,
                backendCreateDescriptor(HIPDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR, _))
        .WillOnce(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendDestroyDescriptor(_))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));
    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    // Expect 4 finalize calls: 3 tensors + 1 operation
    EXPECT_CALL(*_mockBackend, backendFinalize(_))
        .Times(4)
        .WillRepeatedly(Return(HIPDNN_STATUS_SUCCESS));

    auto node = makeConvNode();
    std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<detail::ScopedHipdnnBackendDescriptor> operations;

    auto err = node.create_operation(tensorDescs, operations);
    EXPECT_TRUE(err.is_good()) << err.err_msg;
    EXPECT_EQ(tensorDescs.size(), 3u);
    EXPECT_EQ(operations.size(), 1u);
}

TEST(TestConvolutionNode, GetNodeTypeReturnsConvolutionFprop)
{
    const GraphAttributes graphAttrs;
    const ConvolutionFpropNode node(ConvFpropAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::CONVOLUTION_FPROP);
}
