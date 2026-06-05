// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormNode, PreValidateNode)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateNodeMissingValues)
{
    BatchnormAttributes batchnormAttributes;

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes = BatchnormAttributes{};
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    const BatchnormNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormNode nodeWithY(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormNode nodeWithBias(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithBias.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());

    // Set proper dimensions for the final test to pass with enhanced validation
    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 64, 1, 1});

    auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5);

    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormNode nodeWithAllValues(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, InferPropertiesNode)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 2, 1, 1});

    auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5);

    const GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormNode, InferPropertiesNodeWithStats)
{
    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_prev_running_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_prev_running_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_next_running_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_next_running_variance(std::make_shared<TensorAttributes>());

    auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(3).set_name("MeanTensor");

    auto invVarianceTensor = batchnormAttributes.get_inv_variance();
    invVarianceTensor->set_uid(4).set_name("InvVarianceTensor");

    auto nextRunningMeanTensor = batchnormAttributes.get_next_running_mean();
    nextRunningMeanTensor->set_uid(5).set_name("NextRunningMeanTensor");

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 2, 1, 1});

    auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5);

    auto nextRunningVarianceTensor = batchnormAttributes.get_next_running_variance();
    nextRunningVarianceTensor->set_uid(6).set_name("NextRunningVarianceTensor");

    const GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));

    EXPECT_EQ(meanTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(invVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(nextRunningMeanTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(nextRunningVarianceTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
}

TEST(TestBatchnormNode, GatherHipdnnTensors)
{
    BatchnormAttributes batchnormAttributes;
    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("YTensor");
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(4).set_name("BiasTensor");
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(5).set_name("EpsilonTensor").set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(9).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(10).set_name("PeerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(biasTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(epsilonTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(peerStat1) != allTensors.end());
    EXPECT_TRUE(allTensors.find(peerStat2) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 7);
}

// ============================================================================
// Spatial Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateNodeRejectsInvalidSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 1, 1}).set_stride({256, 1, 1, 1}); // Invalid: N*H*W = 1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 256, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateNodeAcceptsValidSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 3, 2, 2}).set_stride({12, 4, 2, 1}); // Valid: N*H*W = 2*2*2 = 8
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 3, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 3, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateRejectsMismatchedInputOutputShapes)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 16, 16}); // Mismatched spatial dimensions
    batchnormAttributes.set_y(yTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsMismatchedChannelDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 128, 1, 1}); // Mismatched channel dimension
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 128, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidScaleTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 32, 32}); // Should be [1, 64, 1, 1] for spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidBiasTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({2, 64, 1, 1}); // Batch dimension should be 1
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidMeanTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 32, 1, 1}); // Wrong channel count
    batchnormAttributes.set_mean(meanTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Mean tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidInvVarianceTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 2, 1}); // Spatial dimension should be 1
    batchnormAttributes.set_inv_variance(invVarTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse variance tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsIncompleteRunningStats)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    // Only provide some running stats, not all
    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_mean(prevRunningMean);

    // Missing: prev_running_variance, next_running_mean, next_running_variance

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("running statistics") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateAcceptsCompleteRunningStats)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    // Provide all running stats
    auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_mean(prevRunningMean);

    auto prevRunningVar = std::make_shared<TensorAttributes>();
    prevRunningVar->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_variance(prevRunningVar);

    auto nextRunningMean = std::make_shared<TensorAttributes>();
    nextRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_next_running_mean(nextRunningMean);

    auto nextRunningVar = std::make_shared<TensorAttributes>();
    nextRunningVar->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_next_running_variance(nextRunningVar);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateAcceptsValid5DSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 8, 8, 8})
        .set_stride({32768, 512, 64, 8, 1}); // Valid: N*D*H*W = 2*8*8*8 = 1024
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalid5DSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 1, 1, 1})
        .set_stride({64, 1, 1, 1, 1}); // Invalid: N*D*H*W = 1*1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

// ============================================================================
// Required Parameter Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateRejectsScaleWithNoDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>()); // No dimensions!
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsBiasWithNoDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>()); // No dimensions!
    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("must have at least 2 dimensions") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsEpsilonWithNoDimensions)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>()); // No dimensions!

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);
    EXPECT_TRUE(error.get_message().find("Epsilon") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("dimensions are not set") != std::string::npos);
}

TEST(TestBatchnormNode, GetNodeTypeReturnsBatchnorm)
{
    const GraphAttributes graphAttrs;
    const BatchnormNode node(BatchnormAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM);
}
