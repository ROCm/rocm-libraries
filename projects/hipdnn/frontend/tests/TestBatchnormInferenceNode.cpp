// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX - License - Identifier : MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNode.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormInferenceNode, BatchnormInferenceNodeProperties)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());

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

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_dim({1, 2, 1, 1});

    auto invVarTensor = batchnormAttributes.get_inv_variance();
    invVarTensor->set_dim({1, 2, 1, 1});

    const GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);
    auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNode, PreValidateNode)
{
    BatchnormInferenceAttributes batchnormAttributes;

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
    meanTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNode, PreValidateNodeMissingValues)
{
    BatchnormInferenceAttributes batchnormAttributes;

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes = BatchnormInferenceAttributes{};
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithX(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithY(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithScale(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithBias(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithBias.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithMean(std::move(batchnormAttributesCopy), graphAttributes);

    error = nodeWithMean.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());

    // For the final test to pass with enhanced validation, set proper dimensions
    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 64, 1, 1});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_dim({1, 64, 1, 1});

    auto invVarTensor = batchnormAttributes.get_inv_variance();
    invVarTensor->set_dim({1, 64, 1, 1});

    batchnormAttributesCopy = batchnormAttributes;
    const BatchnormInferenceNode nodeWithAllValues(std::move(batchnormAttributesCopy),
                                                   graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNode, InferPropertiesNode)
{
    BatchnormInferenceAttributes batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_inv_variance(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());

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

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_dim({1, 2, 1, 1});

    auto invVarTensor = batchnormAttributes.get_inv_variance();
    invVarTensor->set_dim({1, 2, 1, 1});

    const GraphAttributes graphAttributes;
    BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNode, GatherHipdnnTensors)
{
    BatchnormInferenceAttributes bnAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1).set_name("X");
    bnAttributes.set_x(xTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(2).set_name("Scale");
    bnAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(3).set_name("Bias");
    bnAttributes.set_bias(biasTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(4).set_name("Mean");
    bnAttributes.set_mean(meanTensor);

    auto invVarianceTensor = std::make_shared<TensorAttributes>();
    invVarianceTensor->set_uid(5).set_name("InvVariance");
    bnAttributes.set_inv_variance(invVarianceTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(7).set_name("Y");
    bnAttributes.set_y(yTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(bnAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(biasTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(meanTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(invVarianceTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 6);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNode, PreValidateRejectsMismatchedInputOutputShapes)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormInferenceNode, PreValidateRejectsMismatchedChannelDimensions)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 128, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 128, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestBatchnormInferenceNode, PreValidateRejectsInvalidScaleTensorShape)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNode, PreValidateRejectsInvalidBiasTensorShape)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNode, PreValidateRejectsInvalidMeanTensorShape)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 32, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Mean tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNode, PreValidateRejectsInvalidInvVarianceTensorShape)
{
    BatchnormInferenceAttributes batchnormAttributes;

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
    meanTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 2, 1}); // Spatial dimension should be 1
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Inverse variance tensor") != std::string::npos);
}

// ============================================================================
// Spatial Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNode, PreValidateAcceptsSpatialDimensionEqualsOne)
{
    BatchnormInferenceAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 256, 1, 1})
        .set_stride({256, 1, 1, 1}); // Valid: PyTorch accepts N*H*W = 1 for inference
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 256, 1, 1}); // Spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Inference mode should accept N*spatial=1 (matches PyTorch behavior)";
}

TEST(TestBatchnormInferenceNode, PreValidateAcceptsValidSpatialDimensions)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 3, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 3, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNode, PreValidateAcceptsValid5DSpatialDimensions)
{
    BatchnormInferenceAttributes batchnormAttributes;

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

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNode, PreValidateAccepts5DSpatialDimensionEqualsOne)
{
    BatchnormInferenceAttributes batchnormAttributes;

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_dim({1, 64, 1, 1, 1})
        .set_stride({64, 1, 1, 1, 1}); // Valid for inference: N*D*H*W = 1*1*1*1 = 1
    batchnormAttributes.set_x(xTensor);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_dim({1, 64, 1, 1, 1}); // 5D spatial mode
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto invVarTensor = std::make_shared<TensorAttributes>();
    invVarTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_inv_variance(invVarTensor);

    const GraphAttributes graphAttributes;
    const BatchnormInferenceNode node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Inference mode should accept N*D*H*W=1 for 5D tensors (matches PyTorch behavior)";
}

TEST(TestBatchnormInferenceNode, GetNodeTypeReturnsBatchnormInference)
{
    const GraphAttributes graphAttrs;
    const BatchnormInferenceNode node(BatchnormInferenceAttributes{}, graphAttrs);
    EXPECT_EQ(node.getNodeType(), NodeType::BATCHNORM_INFERENCE);
}
