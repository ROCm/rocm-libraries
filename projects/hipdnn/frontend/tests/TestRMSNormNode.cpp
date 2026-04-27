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
    scaleTensor->set_dim({1, 2, 3, 4});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    const GraphAttributes graphAttributes;
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
    scaleTensor->set_dim({1, 64, 32, 32});
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeMissingValues)
{
    RMSNormAttributes rmsnormAttributes;

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes = RMSNormAttributes{};
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);
    auto rmsnormAttributesCopy = rmsnormAttributes;
    const RMSNormNode nodeWithX(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributesCopy = rmsnormAttributes;
    const RMSNormNode nodeWithScale(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributesCopy = rmsnormAttributes;
    const RMSNormNode nodeWithY(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    // Now set epsilon and proper dimensions to make it pass
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    auto xTensor = rmsnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 32, 32});

    rmsnormAttributesCopy = rmsnormAttributes;
    const RMSNormNode nodeWithAllValues(std::move(rmsnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBiasNormAxis1)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 32, 32});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBiasNormAxis2)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 1, 32, 32});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 1, 32, 32});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBiasNormAxis3)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 1, 1, 32});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 1, 1, 32});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBiasSingleElement)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 1, 1, 1}).set_stride({1, 1, 1, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 1, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 1, 1, 1});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateNodeWithBiasInvalidSingleElement)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 1, 1, 1});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 1, 1, 1});
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestRMSNormNode, PreValidateRejectsMismatchedBiasChannelDimensions)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32});
    rmsnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 128, 32, 32}); // Mismatched channel dimension
    rmsnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

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
    scaleTensor->set_dim({1, 2, 3, 4});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    const GraphAttributes graphAttributes;
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

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Output should inherit input shape
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{2, 64, 8, 8}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{4096, 64, 8, 1}));

    // inv_rms should get norm stats shape [N, 1, H, W]
    EXPECT_EQ(invRmsTensor->get_dim(), (std::vector<int64_t>{2, 1, 8, 8}));
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
    scaleTensor->set_dim({1, 64, 8, 8});

    auto epsilonTensor = rmsnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5f);

    auto biasTensor = rmsnormAttributes.get_bias();
    biasTensor->set_uid(6).set_name("BiasTensor");

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Bias should inherit scale's shape
    EXPECT_EQ(biasTensor->get_dim(), (std::vector<int64_t>{1, 64, 8, 8}));
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

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

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

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

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

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsScaleWithoutTrailingMatch)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1}); // scale[3]=1 vs input[3]=32 — no trailing match
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    const std::string expected = "no trailing dims matching input";
    EXPECT_TRUE(error.get_message().find(expected) != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsScaleWithNonOneInLeadingRegion)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 128, 32, 32}); // should be [1, 1, 32, 32] or [1, 64, 32, 32]
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("leading region before normalized shape")
                != std::string::npos);
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
    scaleTensor->set_dim({1, 64, 8, 8, 8}); // 5D, full-shape
    rmsnormAttributes.set_scale(scaleTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

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

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// Scale-driven inv_rms Shape Inference Tests
// ============================================================================

TEST(TestRMSNormNode, InferInvRmsShapeFromScale5D)
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
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 4, 8, 8})
        .set_stride({16384, 256, 64, 8, 1});

    rmsnormAttributes.get_y()->set_uid(2);

    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 1, 4, 8, 8}); // Norm over trailing (D, H, W) in 5D

    rmsnormAttributes.get_epsilon()->set_dim({1}).set_value(1e-5f);

    auto invRmsTensor = rmsnormAttributes.get_inv_rms();
    invRmsTensor->set_uid(5);

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // inv_rms: where scale is non-1 (D, H, W), collapse to 1 → [N, C, 1, 1, 1]
    EXPECT_EQ(invRmsTensor->get_dim(), (std::vector<int64_t>{2, 64, 1, 1, 1}));
}

TEST(TestRMSNormNode, InferInvRmsStridesFromInputLayout)
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
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 8, 8})
        .set_stride({4096, 64, 8, 1});

    rmsnormAttributes.get_y()->set_uid(2);
    rmsnormAttributes.get_scale()->set_dim({1, 1, 8, 8});
    rmsnormAttributes.get_epsilon()->set_dim({1}).set_value(1e-5f);

    auto invRmsTensor = rmsnormAttributes.get_inv_rms();
    invRmsTensor->set_uid(5);

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(invRmsTensor->get_dim(), (std::vector<int64_t>{2, 64, 1, 1}));
    EXPECT_FALSE(invRmsTensor->get_stride().empty());

    // Row-major strides for [2, 64, 1, 1] derived from input NCHW layout: {64, 1, 1, 1}
    EXPECT_EQ(invRmsTensor->get_stride(), (std::vector<int64_t>{64, 1, 1, 1}));
}

TEST(TestRMSNormNode, InferScaleStridesFromInputLayout)
{
    RMSNormAttributes rmsnormAttributes;
    rmsnormAttributes.set_x(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    rmsnormAttributes.set_forward_phase(NormFwdPhase::INFERENCE);

    // Input is NHWC: C is innermost (stride 1), then W, then H, then N.
    auto inputTensor = rmsnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 8, 8})
        .set_stride({4096, 1, 512, 64});

    rmsnormAttributes.get_y()->set_uid(2);

    // Scale dims set, strides empty — should be inferred from x's NHWC layout.
    auto scaleTensor = rmsnormAttributes.get_scale();
    scaleTensor->set_dim({1, 1, 8, 8});

    rmsnormAttributes.get_epsilon()->set_dim({1}).set_value(1e-5f);

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    // Scale strides follow x's NHWC stride order, not default-packed NCHW.
    // For [1, 1, 8, 8] in NHWC: C(stride 1), W(stride 1, since C dim=1),
    //                            H(stride 8 = C*W), N(stride 64 = C*W*H).
    EXPECT_FALSE(scaleTensor->get_stride().empty());
    EXPECT_EQ(scaleTensor->get_stride(), (std::vector<int64_t>{64, 1, 8, 1}));
}

TEST(TestRMSNormNode, InferInvRmsPreservesUserSetDims)
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
        .set_data_type(DataType::FLOAT)
        .set_dim({2, 64, 8, 8})
        .set_stride({4096, 64, 8, 1});

    rmsnormAttributes.get_y()->set_uid(2);
    rmsnormAttributes.get_scale()->set_dim({1, 64, 8, 8});
    rmsnormAttributes.get_epsilon()->set_dim({1}).set_value(1e-5f);

    // User explicitly sets inv_rms dims — inference should not overwrite
    auto invRmsTensor = rmsnormAttributes.get_inv_rms();
    invRmsTensor->set_uid(5).set_dim({2, 1, 8, 8}).set_stride({64, 64, 8, 1});

    const GraphAttributes graphAttributes;
    RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(invRmsTensor->get_dim(), (std::vector<int64_t>{2, 1, 8, 8}));
    EXPECT_EQ(invRmsTensor->get_stride(), (std::vector<int64_t>{64, 64, 8, 1}));
}

// ============================================================================
// Scale-driven inv_rms Validation Tests
// ============================================================================

TEST(TestRMSNormNode, PreValidateAcceptsCorrectInvRmsShape)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Full-shape scale
    rmsnormAttributes.set_scale(scaleTensor);

    // Correct inv_rms shape derived from scale: where scale is non-1 → 1,
    // where scale is 1 (only batch here) → keep input dim. Yields [N, 1, 1, 1].
    auto invRmsTensor = std::make_shared<TensorAttributes>();
    invRmsTensor->set_dim({2, 1, 1, 1});
    rmsnormAttributes.set_inv_rms(invRmsTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, PreValidateRejectsInvRmsWithChannelOnlyShape)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Full-shape scale
    rmsnormAttributes.set_scale(scaleTensor);

    // Old incorrect shape [1, C, 1, 1] — should be rejected
    auto invRmsTensor = std::make_shared<TensorAttributes>();
    invRmsTensor->set_dim({1, 64, 1, 1});
    rmsnormAttributes.set_inv_rms(invRmsTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse RMS tensor") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsInvRmsWithMismatchedBatchDim)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Full-shape scale
    rmsnormAttributes.set_scale(scaleTensor);

    // Batch dim mismatch: input has N=2, inv_rms has N=4
    auto invRmsTensor = std::make_shared<TensorAttributes>();
    invRmsTensor->set_dim({4, 1, 1, 1});
    rmsnormAttributes.set_inv_rms(invRmsTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse RMS tensor") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateRejectsInvRmsWithWrongRank)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Full-shape scale
    rmsnormAttributes.set_scale(scaleTensor);

    // Wrong rank: 5D instead of 4D
    auto invRmsTensor = std::make_shared<TensorAttributes>();
    invRmsTensor->set_dim({2, 1, 1, 1, 1});
    rmsnormAttributes.set_inv_rms(invRmsTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("same rank") != std::string::npos);
}

TEST(TestRMSNormNode, PreValidateSkipsInvRmsValidationWhenDimsNotSet)
{
    RMSNormAttributes rmsnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    rmsnormAttributes.set_x(xTensor);

    rmsnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Full-shape scale
    rmsnormAttributes.set_scale(scaleTensor);

    // inv_rms with no dims set — validation should pass (dims will be inferred)
    rmsnormAttributes.set_inv_rms(std::make_shared<TensorAttributes>());

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5f);
    rmsnormAttributes.set_epsilon(epsilonTensor);

    rmsnormAttributes.set_forward_phase(NormFwdPhase::TRAINING);

    const GraphAttributes graphAttributes;
    const RMSNormNode node(std::move(rmsnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestRMSNormNode, GetNodeTypeReturnsRmsNorm)
{
    const GraphAttributes graphAttrs;
    const RMSNormNode node(RMSNormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::RMS_NORM);
}
