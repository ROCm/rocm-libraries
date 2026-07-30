// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormBackwardNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormBackwardNode, PreValidateNode)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, PreValidateNodeMissingValues)
{
    BatchnormBackwardAttributes batchnormAttributes;

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes = BatchnormBackwardAttributes{};
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithDy(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDy.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithDx(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDx.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithDscale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithDscale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    // For the final test to pass with enhanced validation, set proper dimensions
    auto dyTensor = batchnormAttributes.get_dy();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormBackwardNode nodeWithAllValues(std::move(batchnormAttributesCopy),
                                                  graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, InferPropertiesNode)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_uid(1)
        .set_name("InputTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({24, 12, 4, 1}); // NCHW layout

    auto dxTensor = batchnormAttributes.get_dx();
    dxTensor->set_uid(2).set_name("DxTensor");

    auto dscaleTensor = batchnormAttributes.get_dscale();
    dscaleTensor->set_uid(3).set_name("DscaleTensor");

    auto dbiasTensor = batchnormAttributes.get_dbias();
    dbiasTensor->set_uid(4).set_name("DbiasTensor");

    const GraphAttributes graphAttributes;
    BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(dxTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(dxTensor->get_stride(), (std::vector<int64_t>{24, 12, 4, 1}));

    EXPECT_EQ(dscaleTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dscaleTensor->get_stride(),
              (std::vector<int64_t>{2, 1, 1, 1})); // Inherits NCHW layout

    EXPECT_EQ(dbiasTensor->get_dim(), (std::vector<int64_t>{1, 2, 1, 1}));
    EXPECT_EQ(dbiasTensor->get_stride(),
              (std::vector<int64_t>{2, 1, 1, 1})); // Inherits NCHW layout
}

TEST(TestBatchnormBackwardNode, GatherHipdnnTensors)
{
    BatchnormBackwardAttributes batchnormAttributes;
    batchnormAttributes.set_dy(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    auto peerStat1 = std::make_shared<TensorAttributes>();
    peerStat1->set_uid(9).set_name("PeerStat1");

    auto peerStat2 = std::make_shared<TensorAttributes>();
    peerStat2->set_uid(10).set_name("PeerStat2");

    batchnormAttributes.set_peer_stats({peerStat1, peerStat2});

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(peerStat1) != allTensors.end());
    EXPECT_TRUE(allTensors.find(peerStat2) != allTensors.end());
    EXPECT_EQ(allTensors.size(), 10);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormBackwardNode, PreValidateRejectsMismatchedInputGradientShapes)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 16, 16}).set_stride({16384, 256, 16, 1}); // Mismatched spatial dims
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsMismatchedOutputGradientShapes)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto dxTensor = std::make_shared<TensorAttributes>();
    dxTensor->set_dim({2, 64, 16, 16}); // Mismatched spatial dims
    batchnormAttributes.set_dx(dxTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsMismatchedChannelDimensions)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 128, 1, 1}); // Mismatched channel dimension
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsInvalidDscaleTensorShape)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());

    auto dscaleTensor = std::make_shared<TensorAttributes>();
    dscaleTensor->set_dim({1, 64, 32, 32}); // Should be [1, 64, 1, 1]
    batchnormAttributes.set_dscale(dscaleTensor);

    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale gradient tensor") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsInvalidDbiasTensorShape)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());

    auto dbiasTensor = std::make_shared<TensorAttributes>();
    dbiasTensor->set_dim({2, 64, 1, 1}); // Batch dimension should be 1
    batchnormAttributes.set_dbias(dbiasTensor);

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias gradient tensor") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsInvalidMeanTensorShape)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 32, 1, 1}); // Wrong channel count
    batchnormAttributes.set_mean(meanTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Mean tensor") != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsInvalidInvVarianceTensorShape)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 2, 1}); // Spatial dimension should be 1
    batchnormAttributes.set_inv_variance(invVarTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse variance tensor") != std::string::npos);
}

// ============================================================================
// Spatial Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormBackwardNode, PreValidateNodeRejectsInvalidSpatialDimensions)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 256, 1, 1}).set_stride({256, 1, 1, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 1, 1}).set_stride({256, 1, 1, 1}); // Invalid: N*H*W = 1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 256, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Batch normalization backward") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestBatchnormBackwardNode, PreValidateNodeAcceptsValidSpatialDimensions)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 3, 2, 2}).set_stride({12, 4, 2, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 3, 2, 2}).set_stride({12, 4, 2, 1}); // Valid: N*H*W = 2*2*2 = 8
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 3, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, PreValidateNodeRejectsMismatchedMeanInvVariance)
{
    // Test: Setting only mean without inv_variance should fail
    {
        BatchnormBackwardAttributes batchnormAttributes;

        auto xTensor = std::make_shared<TensorAttributes>();
        xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_x(xTensor);

        auto dyTensor = std::make_shared<TensorAttributes>();
        dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_dy(dyTensor);

        auto scaleTensor = std::make_shared<TensorAttributes>();
        scaleTensor->set_dim({1, 64, 1, 1});
        batchnormAttributes.set_scale(scaleTensor);

        batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
        // Intentionally NOT setting inv_variance

        const GraphAttributes graphAttributes;
        const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
        EXPECT_TRUE(error.get_message().find("both mean and inv_variance") != std::string::npos);
    }

    // Test: Setting only inv_variance without mean should fail
    {
        BatchnormBackwardAttributes batchnormAttributes;

        auto xTensor = std::make_shared<TensorAttributes>();
        xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_x(xTensor);

        auto dyTensor = std::make_shared<TensorAttributes>();
        dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_dy(dyTensor);

        auto scaleTensor = std::make_shared<TensorAttributes>();
        scaleTensor->set_dim({1, 64, 1, 1});
        batchnormAttributes.set_scale(scaleTensor);

        batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());
        // Intentionally NOT setting mean
        batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());

        const GraphAttributes graphAttributes;
        const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
        EXPECT_TRUE(error.get_message().find("both mean and inv_variance") != std::string::npos);
    }

    // Test: Setting both should pass
    {
        BatchnormBackwardAttributes batchnormAttributes;

        auto xTensor = std::make_shared<TensorAttributes>();
        xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_x(xTensor);

        auto dyTensor = std::make_shared<TensorAttributes>();
        dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_dy(dyTensor);

        auto scaleTensor = std::make_shared<TensorAttributes>();
        scaleTensor->set_dim({1, 64, 1, 1});
        batchnormAttributes.set_scale(scaleTensor);

        batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());

        const GraphAttributes graphAttributes;
        const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }

    // Test: Setting neither should pass
    {
        BatchnormBackwardAttributes batchnormAttributes;

        auto xTensor = std::make_shared<TensorAttributes>();
        xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_x(xTensor);

        auto dyTensor = std::make_shared<TensorAttributes>();
        dyTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});
        batchnormAttributes.set_dy(dyTensor);

        auto scaleTensor = std::make_shared<TensorAttributes>();
        scaleTensor->set_dim({1, 64, 1, 1});
        batchnormAttributes.set_scale(scaleTensor);

        batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
        batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());
        // Intentionally NOT setting mean or inv_variance

        const GraphAttributes graphAttributes;
        const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

        auto error = node.pre_validate_node();
        EXPECT_EQ(error.code, ErrorCode::OK);
    }
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestBatchnormBackwardNode, PreValidateAcceptsValid5DSpatialDimensions)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({2, 64, 8, 8, 8}).set_stride({32768, 512, 64, 8, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({2, 64, 8, 8, 8})
        .set_stride({32768, 512, 64, 8, 1}); // Valid: N*D*H*W = 2*8*8*8 = 1024
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormBackwardNode, PreValidateRejectsInvalid5DSpatialDimensions)
{
    BatchnormBackwardAttributes batchnormAttributes;

    auto dyTensor = std::make_shared<TensorAttributes>();
    dyTensor->set_dim({1, 64, 1, 1, 1}).set_stride({64, 1, 1, 1, 1});
    batchnormAttributes.set_dy(dyTensor);

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 1, 1, 1})
        .set_stride({64, 1, 1, 1, 1}); // Invalid: N*D*H*W = 1*1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    batchnormAttributes.set_dx(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dscale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_dbias(std::make_shared<TensorAttributes>());

    const GraphAttributes graphAttributes;
    const BatchnormBackwardNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Batch normalization backward") != std::string::npos);
    EXPECT_TRUE(error.get_message().find("N * spatial_dimensions must be > 1")
                != std::string::npos);
}

TEST(TestBatchnormBackwardNode, GetNodeTypeReturnsBatchnormBackward)
{
    const GraphAttributes graphAttrs;
    const BatchnormBackwardNode node(BatchnormBackwardAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM_BACKWARD);
}
