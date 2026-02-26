// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/node/RMSNormNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestRMSNormNode, RMSNormNodeProperties)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = rmsnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);
    auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestRMSNormNode, PreValidateNode)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeMissingValues)
{
    RMSNormAttributes rmsnormAttributes;

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);
    auto rmsnormAttributesCopy = rmsnormAttributes;
    RMSNormNode nodeWithX(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributesCopy = rmsnormAttributes;
    RMSNormNode nodeWithScale(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributesCopy = rmsnormAttributes;
    RMSNormNode nodeWithY(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    // Now set epsilon and proper dimensions to make it pass
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    auto xTensor = rmsnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    rmsnormAttributesCopy = rmsnormAttributes;
    RMSNormNode nodeWithAllValues(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBias)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateRejectsMismatchedBiasChannelDimensions)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 128, 1, 1}); // Mismatched channel dimension
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
}

TEST(TestRMSNormNode, InferPropertiesNode)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = rmsnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestRMSNormNode, InferPropertiesNodeWithInvRms)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_inv_rms(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 8, 8})
        .set_stride({4096, 64, 8, 1});

    auto outputTensor = rmsnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    auto invRmsTensor = rmsnormAttributes.get_inv_rms();
    invRmsTensor->set_uid(5).set_name("InvRmsTensor");

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Output should inherit input shape
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{2, 64, 8, 8}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{4096, 64, 8, 1}));

    // inv_rms should get channel-only shape [1, C, 1, 1]
    EXPECT_EQ(invRmsTensor->get_dim(), (std::vector<int64_t>{1, 64, 1, 1}));
}

TEST(TestRMSNormNode, InferPropertiesNodeWithBias)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 8, 8})
        .set_stride({4096, 64, 8, 1});

    auto outputTensor = rmsnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    auto biasTensor = rmsnormAttributes.get_bias();
    biasTensor->set_uid(6).set_name("BiasTensor");

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Bias should get channel-only shape [1, C, 1, 1]
    EXPECT_EQ(biasTensor->get_dim(), (std::vector<int64_t>{1, 64, 1, 1}));
}

TEST(TestRMSNormNode, PackNode)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_name("RMSNorm");
    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    rmsnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    rmsnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(4).set_name("EpsilonTensor").set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "RMSNorm");
    EXPECT_EQ(nodeFlatbuffer->attributes_type(),
              hipdnn_data_sdk::data_objects::NodeAttributes::RMSNormAttributes);

    auto packedAttributes = nodeFlatbuffer->attributes_as_RMSNormAttributes();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->y_tensor_uid(), yTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->epsilon_tensor_uid(), epsilonTensor->get_uid());
}

TEST(TestRMSNormNode, PackNodeWithBias)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_name("RMSNorm");

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_dim({1, 2, 3, 4}).set_stride({4, 3, 2, 1});
    rmsnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_dim({1, 2, 3, 4}).set_stride({4, 3, 2, 1});
    rmsnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(4).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(5).set_dim({1, 2, 1, 1}).set_stride({2, 1, 1, 1});
    rmsnormAttributes.set_bias(biasTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    builder.Finish(offset);

    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);
    auto packedAttributes = nodeFlatbuffer->attributes_as_RMSNormAttributes();

    ASSERT_NE(packedAttributes, nullptr);
    EXPECT_TRUE(packedAttributes->bias_tensor_uid().has_value());
    EXPECT_EQ(packedAttributes->bias_tensor_uid().value(), biasTensor->get_uid());
}

TEST(TestRMSNormNode, GatherHipdnnTensors)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    rmsnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_name("Scale");
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(3).set_name("Epsilon");
    rmsnormAttributes.set_epsilon(epsilonTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(4).set_name("Y");
    rmsnormAttributes.set_y(yTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilonTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 4);
}

TEST(TestRMSNormNode, GatherHipdnnTensorsWithInvRmsAndBias)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    rmsnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_name("Scale");
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(3).set_name("Epsilon");
    rmsnormAttributes.set_epsilon(epsilonTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(4).set_name("Y");
    rmsnormAttributes.set_y(yTensor);

    auto invRmsTensor = std::make_shared<TensorAttributes>();
    invRmsTensor->set_uid(5).set_name("InvRms");
    rmsnormAttributes.set_inv_rms(invRmsTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(6).set_name("Bias");
    rmsnormAttributes.set_bias(biasTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilonTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(invRmsTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(biasTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 6);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestRMSNormNode, PreValidateRejectsMismatchedInputOutputShapes)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 16, 16}); // Mismatched spatial dimensions
    rmsnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsMismatchedChannelDimensions)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 128, 1, 1}); // Mismatched channel dimension
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsInvalidScaleTensorShape)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Should be [1, 64, 1, 1]
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestRMSNormNode, PreValidateAcceptsValid5DSpatialDimensions)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 8, 8, 8}).set_stride({32768, 512, 64, 8, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateAcceptsSingleElementSpatialDimensions)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 1, 1}).set_stride({256, 1, 1, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 256, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}
