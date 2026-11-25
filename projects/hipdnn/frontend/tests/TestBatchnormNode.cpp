// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormNode.hpp>

#include "fake_backend/MockHipdnnBackend.hpp"

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

// NOTE: Frontend checks only the channel dimension of scale/bias.
// Backend providers (e.g. MIOpen) may still reject unsupported shapes.
TEST(TestBatchnormNode, PreValidateAcceptsBroadcastableScaleBias)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({64});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({64});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateRejectsMismatchedScaleBiasShapes)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({64});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({32}); // intentionally mismatched
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBatchnormNode, PreValidateAcceptsRank2BroadcastableScaleBias)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    // {1, C} broadcastable shape
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateAcceptsDifferentBroadcastRanks)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    // Example broadcastable shapes
    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({64}); // rank-1
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({64}); // same shape in this example (not required)
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

namespace
{

Error preValidateWithScaleBiasDims(const std::vector<int64_t>& xDims,
                                   const std::vector<int64_t>& scaleBiasDims)
{
    BatchnormAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim(xDims);
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim(scaleBiasDims);
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim(scaleBiasDims);
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);
    return node.pre_validate_node();
}

} // namespace

TEST(TestBatchnormNode, PreValidateRejectsScaleBiasChannelMismatch)
{
    auto error = preValidateWithScaleBiasDims({2, 8, 4, 4}, {1, 6, 1, 1});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("Scale"), std::string::npos) << error.get_message();
}

TEST(TestBatchnormNode, PreValidateRejectsRank1ScaleBiasChannelMismatch)
{
    auto error = preValidateWithScaleBiasDims({2, 64, 32, 32}, {32});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBatchnormNode, PreValidateRejectsRank2ScaleBiasChannelMismatch)
{
    auto error = preValidateWithScaleBiasDims({2, 64, 32, 32}, {1, 32});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
}

TEST(TestBatchnormNode, PreValidateAcceptsReducedRankScaleBias)
{
    auto error = preValidateWithScaleBiasDims({2, 64, 32, 32}, {64, 1, 1});
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestBatchnormNode, PreValidateRejectsReducedRankScaleBiasChannelMismatch)
{
    auto error = preValidateWithScaleBiasDims({2, 64, 32, 32}, {32, 1, 1});
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_NE(error.get_message().find("(index 0) must be 64, got 32"), std::string::npos)
        << error.get_message();
}

TEST(TestBatchnormNode, PreValidateAcceptsChannelFirstRank2ScaleBias)
{
    auto error = preValidateWithScaleBiasDims({2, 64, 32, 32}, {64, 1});
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

TEST(TestBatchnormNode, PreValidateNode)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
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
    const auto xTensor = batchnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    const auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    const auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 64, 1, 1});

    const auto epsilonTensor = batchnormAttributes.get_epsilon();
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

    const auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    const auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    const auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    const auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 2, 1, 1});

    const auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5);

    const GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.infer_properties_node();
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

    const auto inputTensor = batchnormAttributes.get_x();
    inputTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({5, 6, 7, 8});

    const auto outputTensor = batchnormAttributes.get_y();
    outputTensor->set_uid(2).set_name("OutputTensor");

    const auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_uid(3).set_name("MeanTensor");

    const auto invVarianceTensor = batchnormAttributes.get_inv_variance();
    invVarianceTensor->set_uid(4).set_name("InvVarianceTensor");

    const auto nextRunningMeanTensor = batchnormAttributes.get_next_running_mean();
    nextRunningMeanTensor->set_uid(5).set_name("NextRunningMeanTensor");

    const auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 2, 1, 1});

    const auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 2, 1, 1});

    const auto epsilonTensor = batchnormAttributes.get_epsilon();
    epsilonTensor->set_dim({1}).set_value(1e-5);

    const auto nextRunningVarianceTensor = batchnormAttributes.get_next_running_variance();
    nextRunningVarianceTensor->set_uid(6).set_name("NextRunningVarianceTensor");

    const GraphAttributes graphAttributes;
    BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.infer_properties_node();
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
    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("XTensor");
    batchnormAttributes.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2).set_name("YTensor");
    batchnormAttributes.set_y(yTensor);

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(3).set_name("ScaleTensor");
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(4).set_name("BiasTensor");
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(5).set_name("EpsilonTensor").set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(9).set_name("PeerStat1");

    const auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(10).set_name("PeerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;
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

TEST(TestBatchnormNode, GatherHipdnnTensorsIncludesPeerStatRaggedOffset)
{
    // The aux tensor carries no UID; gathering it is what lets the graph assign one.
    auto peerStatOffset = std::make_shared<TensorAttributes>();
    peerStatOffset->set_name("PeerStatOffset").set_dim({5}).set_stride({1});

    auto peerStat = std::make_shared<TensorAttributes>();
    peerStat->set_uid(9).set_name("PeerStat");
    peerStat->set_ragged_offset(peerStatOffset);

    BatchnormAttributes batchnormAttributes;
    batchnormAttributes.set_peer_stats({peerStat});

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    hipdnn_frontend::detail::OrderedTensorSet allTensors;
    node.gather_hipdnn_tensors(allTensors);

    // The aux tensor directly follows the tensor that carries it.
    auto it = allTensors.find(peerStat);
    ASSERT_TRUE(it != allTensors.end());
    ++it;
    ASSERT_TRUE(it != allTensors.end());
    EXPECT_EQ(*it, peerStatOffset);
}

// ============================================================================
// Spatial Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateNodeRejectsInvalidSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 1, 1}).set_stride({256, 1, 1, 1}); // Invalid: N*H*W = 1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 256, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateNodeAcceptsValidSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 3, 2, 2}).set_stride({12, 4, 2, 1}); // Valid: N*H*W = 2*2*2 = 8
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 3, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 3, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateRejectsMismatchedInputOutputShapes)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    const auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_dim({2, 64, 16, 16}); // Mismatched spatial dimensions
    batchnormAttributes.set_y(yTensor);

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidMeanTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 32, 1, 1}); // Wrong channel count
    batchnormAttributes.set_mean(meanTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Mean tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalidInvVarianceTensorShape)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 2, 1}); // Spatial dimension should be 1
    batchnormAttributes.set_inv_variance(invVarTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse variance tensor") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsIncompleteRunningStats)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    // Only provide some running stats, not all
    const auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_mean(prevRunningMean);

    // Missing: prev_running_variance, next_running_mean, next_running_variance

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("running statistics") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateAcceptsCompleteRunningStats)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    // Provide all running stats
    const auto prevRunningMean = std::make_shared<TensorAttributes>();
    prevRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_mean(prevRunningMean);

    const auto prevRunningVar = std::make_shared<TensorAttributes>();
    prevRunningVar->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_prev_running_variance(prevRunningVar);

    const auto nextRunningMean = std::make_shared<TensorAttributes>();
    nextRunningMean->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_next_running_mean(nextRunningMean);

    const auto nextRunningVar = std::make_shared<TensorAttributes>();
    nextRunningVar->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_next_running_variance(nextRunningVar);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateAcceptsMomentumWithoutPreviousRunningStats)
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

    auto momentumTensor = std::make_shared<TensorAttributes>();
    momentumTensor->set_value(0.1);
    batchnormAttributes.set_momentum(momentumTensor);

    const GraphAttributes graphAttributes;

    // cuDNN accepts momentum without any running statistics and ignores it; it sets momentum
    // on the operation only when the previous running mean is present (BatchnormPacker.hpp).
    auto withoutStats = batchnormAttributes;
    const BatchnormNode nodeWithoutStats(std::move(withoutStats), graphAttributes);
    auto error = nodeWithoutStats.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();

    // Momentum with the complete running statistics.
    auto runningStat = [] {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_dim({1, 64, 1, 1});
        return tensor;
    };
    batchnormAttributes.set_prev_running_mean(runningStat())
        .set_prev_running_variance(runningStat())
        .set_next_running_mean(runningStat())
        .set_next_running_variance(runningStat());
    const BatchnormNode nodeWithStats(std::move(batchnormAttributes), graphAttributes);
    error = nodeWithStats.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK) << error.get_message();
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestBatchnormNode, PreValidateAcceptsValid5DSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 8, 8, 8})
        .set_stride({32768, 512, 64, 8, 1}); // Valid: N*D*H*W = 2*8*8*8 = 1024
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormNode, PreValidateRejectsInvalid5DSpatialDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 1, 1, 1})
        .set_stride({64, 1, 1, 1, 1}); // Invalid: N*D*H*W = 1*1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>()); // No dimensions!
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("must have at least 1 dimension") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsBiasWithNoDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>()); // No dimensions!
    const auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_dim({1}).set_value(1e-5);
    batchnormAttributes.set_epsilon(epsilonTensor);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("must have at least 1 dimension") != std::string::npos);
}

TEST(TestBatchnormNode, PreValidateRejectsEpsilonWithNoDimensions)
{
    BatchnormAttributes batchnormAttributes;

    const auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    const auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    const auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_epsilon(std::make_shared<TensorAttributes>()); // No dimensions!

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(batchnormAttributes), graphAttributes);

    const auto error = node.pre_validate_node();
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

namespace
{
class TestBatchnormNodeCreateOperation : public ::testing::Test
{
protected:
    std::shared_ptr<::testing::NiceMock<Mock_hipdnn_backend>> _mockBackend;
    std::array<char, 16> _fakeDescriptors{};
    size_t _nextDescriptor = 0;

    void SetUp() override
    {
        using ::testing::_;
        using ::testing::Return;

        _mockBackend = std::make_shared<::testing::NiceMock<Mock_hipdnn_backend>>();
        hipdnn_frontend::detail::IHipdnnBackend::setInstance(_mockBackend);

        ON_CALL(*_mockBackend, backendCreateDescriptor(_, _))
            .WillByDefault([this](hipdnnBackendDescriptorType_t,
                                  hipdnnBackendDescriptor_t* descriptor) {
                *descriptor = reinterpret_cast<hipdnnBackendDescriptor_t>(
                    std::addressof(_fakeDescriptors[_nextDescriptor++ % _fakeDescriptors.size()]));
                return HIPDNN_STATUS_SUCCESS;
            });
        ON_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendFinalize(_)).WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
        ON_CALL(*_mockBackend, backendDestroyDescriptor(_))
            .WillByDefault(Return(HIPDNN_STATUS_SUCCESS));
    }

    void TearDown() override
    {
        hipdnn_frontend::detail::IHipdnnBackend::resetInstance();
        _mockBackend.reset();
    }

    static std::shared_ptr<TensorAttributes>
        makeTensor(int64_t uid, const std::vector<int64_t>& dim, const std::vector<int64_t>& stride)
    {
        auto tensor = std::make_shared<TensorAttributes>();
        tensor->set_uid(uid).set_data_type(DataType::FLOAT).set_dim(dim).set_stride(stride);
        return tensor;
    }

    // Builds the required tensors (x, y, scale, bias, epsilon) and compute data type shared by
    // every case below; the caller adds momentum and running statistics on top. Epsilon is
    // marked pass-by-value: validateScalarParameter (Utilities.hpp) rejects a required scalar
    // parameter that is not, independent of the momentum behavior under test here.
    static BatchnormAttributes makeBaseAttributes()
    {
        BatchnormAttributes attributes;
        attributes.set_compute_data_type(DataType::FLOAT);
        attributes.set_x(makeTensor(1, {2, 64, 32, 32}, {65536, 1024, 32, 1}));
        attributes.set_y(makeTensor(2, {2, 64, 32, 32}, {65536, 1024, 32, 1}));
        attributes.set_scale(makeTensor(3, {1, 64, 1, 1}, {64, 1, 1, 1}));
        attributes.set_bias(makeTensor(4, {1, 64, 1, 1}, {64, 1, 1, 1}));
        auto epsilon = makeTensor(5, {1}, {1});
        epsilon->set_is_pass_by_value(true);
        attributes.set_epsilon(epsilon);
        return attributes;
    }
};
} // namespace

// cuDNN parity (B3): momentum alone, without the previous running mean and variance, has
// nothing to weight, so the lowered operation omits it (BatchnormPacker.hpp).
TEST_F(TestBatchnormNodeCreateOperation, OmitsMomentumWithoutPreviousRunningStats)
{
    using ::testing::_;
    using ::testing::AnyNumber;

    auto attributes = makeBaseAttributes();
    attributes.set_momentum(makeTensor(6, {1}, {1}));

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT, _, _, _))
        .Times(0);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<detail::ScopedHipdnnBackendDescriptor> operations;
    error = node.create_operation(tensorDescs, operations);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(operations.size(), 1u);
}

// cuDNN parity (B3): momentum is lowered exactly once when both the previous running mean and
// variance are present.
TEST_F(TestBatchnormNodeCreateOperation, SendsMomentumOnceWithFullRunningStats)
{
    using ::testing::_;
    using ::testing::AnyNumber;

    auto attributes = makeBaseAttributes();
    attributes.set_momentum(makeTensor(6, {1}, {1}));
    attributes.set_prev_running_mean(makeTensor(7, {1, 64, 1, 1}, {64, 1, 1, 1}));
    attributes.set_prev_running_variance(makeTensor(8, {1, 64, 1, 1}, {64, 1, 1, 1}));
    attributes.set_next_running_mean(makeTensor(9, {1, 64, 1, 1}, {64, 1, 1, 1}));
    attributes.set_next_running_variance(makeTensor(10, {1, 64, 1, 1}, {64, 1, 1, 1}));

    EXPECT_CALL(*_mockBackend, backendSetAttribute(_, _, _, _, _)).Times(AnyNumber());
    EXPECT_CALL(*_mockBackend,
                backendSetAttribute(_, HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT, _, _, _))
        .Times(1);

    const GraphAttributes graphAttributes;
    const BatchnormNode node(std::move(attributes), graphAttributes);

    auto error = node.pre_validate_node();
    ASSERT_EQ(error.code, ErrorCode::OK) << error.get_message();

    std::unordered_map<int64_t, detail::ScopedHipdnnBackendDescriptor> tensorDescs;
    std::vector<detail::ScopedHipdnnBackendDescriptor> operations;
    error = node.create_operation(tensorDescs, operations);

    ASSERT_TRUE(error.is_good()) << error.get_message();
    EXPECT_EQ(operations.size(), 1u);
}
