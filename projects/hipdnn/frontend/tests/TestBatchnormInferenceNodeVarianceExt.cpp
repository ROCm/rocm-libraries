// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/node/BatchnormInferenceNodeVarianceExt.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestBatchnormInferenceNodeVarianceExt, BatchnormInferenceNodeVarianceExtProperties)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_variance(std::make_shared<TensorAttributes>());
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

    auto varianceTensor = batchnormAttributes.get_variance();
    varianceTensor->set_dim({1, 2, 1, 1});

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);
    auto error = node.infer_properties_node();

    EXPECT_EQ(error.code, ErrorCode::OK);
    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateNode)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateNodeMissingValues)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    auto batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithX(std::move(batchnormAttributesCopy),
                                                graphAttributes);

    error = nodeWithX.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithY(std::move(batchnormAttributesCopy),
                                                graphAttributes);

    error = nodeWithY.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_scale(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithScale(std::move(batchnormAttributesCopy),
                                                    graphAttributes);

    error = nodeWithScale.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_bias(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithBias(std::move(batchnormAttributesCopy),
                                                   graphAttributes);

    error = nodeWithBias.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithMean(std::move(batchnormAttributesCopy),
                                                   graphAttributes);

    error = nodeWithMean.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::ATTRIBUTE_NOT_SET);

    batchnormAttributes.set_variance(std::make_shared<TensorAttributes>());

    // For the final test to pass with enhanced validation, set proper dimensions
    auto xTensor = batchnormAttributes.get_x();
    xTensor->set_dim({2, 64, 32, 32}).set_stride({65536, 1024, 32, 1});

    auto scaleTensor = batchnormAttributes.get_scale();
    scaleTensor->set_dim({1, 64, 1, 1});

    auto biasTensor = batchnormAttributes.get_bias();
    biasTensor->set_dim({1, 64, 1, 1});

    auto meanTensor = batchnormAttributes.get_mean();
    meanTensor->set_dim({1, 64, 1, 1});

    auto varianceTensor = batchnormAttributes.get_variance();
    varianceTensor->set_dim({1, 64, 1, 1});

    batchnormAttributesCopy = batchnormAttributes;
    BatchnormInferenceNodeVarianceExt nodeWithAllValues(std::move(batchnormAttributesCopy),
                                                        graphAttributes);

    error = nodeWithAllValues.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNodeVarianceExt, InferPropertiesNode)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;
    batchnormAttributes.set_x(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_y(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_mean(std::make_shared<TensorAttributes>());
    batchnormAttributes.set_variance(std::make_shared<TensorAttributes>());
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

    auto varianceTensor = batchnormAttributes.get_variance();
    varianceTensor->set_dim({1, 2, 1, 1});

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.infer_properties_node();
    EXPECT_EQ(error.code, ErrorCode::OK);

    EXPECT_EQ(outputTensor->get_dim(), (std::vector<int64_t>{1, 2, 3, 4}));
    EXPECT_EQ(outputTensor->get_stride(), (std::vector<int64_t>{5, 6, 7, 8}));
}

TEST(TestBatchnormInferenceNodeVarianceExt, PackNode)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;
    batchnormAttributes.set_name("BatchnormInferenceVarianceExt");

    auto xTensor = std::make_shared<TensorAttributes>();
    xTensor->set_uid(1)
        .set_name("XTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_x(xTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(2)
        .set_name("YTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 3, 4})
        .set_stride({4, 3, 2, 1});
    batchnormAttributes.set_y(yTensor);

    auto meanTensor = std::make_shared<TensorAttributes>();
    meanTensor->set_uid(3)
        .set_name("MeanTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_mean(meanTensor);

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_uid(4)
        .set_name("VarianceTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    auto scaleTensor = std::make_shared<TensorAttributes>();
    scaleTensor->set_uid(5)
        .set_name("ScaleTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_scale(scaleTensor);

    auto biasTensor = std::make_shared<TensorAttributes>();
    biasTensor->set_uid(6)
        .set_name("BiasTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1, 2, 1, 1})
        .set_stride({2, 1, 1, 1});
    batchnormAttributes.set_bias(biasTensor);

    auto epsilonTensor = std::make_shared<TensorAttributes>();
    epsilonTensor->set_uid(7)
        .set_name("EpsilonTensor")
        .set_data_type(DataType::FLOAT)
        .set_dim({1})
        .set_stride({1});
    batchnormAttributes.set_epsilon(epsilonTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    flatbuffers::FlatBufferBuilder builder;
    auto offset = node.pack_node(builder);
    EXPECT_NE(offset.o, 0);

    builder.Finish(offset);
    auto bufferPointer = builder.GetBufferPointer();
    auto nodeFlatbuffer = flatbuffers::GetRoot<hipdnn_data_sdk::data_objects::Node>(bufferPointer);

    EXPECT_STREQ(nodeFlatbuffer->name()->c_str(), "BatchnormInferenceVarianceExt");
    EXPECT_EQ(
        nodeFlatbuffer->attributes_type(),
        hipdnn_data_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributesVarianceExt);

    auto packedAttributes = nodeFlatbuffer->attributes_as_BatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(packedAttributes, nullptr);

    EXPECT_EQ(packedAttributes->x_tensor_uid(), xTensor->get_uid());
    EXPECT_EQ(packedAttributes->y_tensor_uid(), yTensor->get_uid());
    EXPECT_EQ(packedAttributes->mean_tensor_uid(), meanTensor->get_uid());
    EXPECT_EQ(packedAttributes->variance_tensor_uid(), varianceTensor->get_uid());
    EXPECT_EQ(packedAttributes->scale_tensor_uid(), scaleTensor->get_uid());
    EXPECT_EQ(packedAttributes->bias_tensor_uid(), biasTensor->get_uid());
    EXPECT_EQ(packedAttributes->epsilon_tensor_uid(), epsilonTensor->get_uid());
}

TEST(TestBatchnormInferenceNodeVarianceExt, GatherHipdnnTensors)
{
    BatchnormInferenceAttributesVarianceExt bnAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_uid(5).set_name("Variance");
    bnAttributes.set_variance(varianceTensor);

    auto yTensor = std::make_shared<TensorAttributes>();
    yTensor->set_uid(7).set_name("Y");
    bnAttributes.set_y(yTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(bnAttributes), graphAttributes);

    std::unordered_set<std::shared_ptr<TensorAttributes>> allTensors;
    node.gather_hipdnn_tensors(allTensors);

    EXPECT_TRUE(allTensors.find(xTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(scaleTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(biasTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(meanTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(varianceTensor) != allTensors.end());
    EXPECT_TRUE(allTensors.find(yTensor) != allTensors.end());

    EXPECT_EQ(allTensors.size(), 6);
}

// ============================================================================
// Shape and Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsMismatchedInputOutputShapes)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("dimension mismatch") != std::string::npos);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsMismatchedChannelDimensions)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 128, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("channel dimension") != std::string::npos);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsInvalidScaleTensorShape)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Scale tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsInvalidBiasTensorShape)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Bias tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsInvalidMeanTensorShape)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 32, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Mean tensor") != std::string::npos);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateRejectsInvalidVarianceTensorShape)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 2, 1}); // Spatial dimension should be 1
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::INVALID_VALUE);
    EXPECT_TRUE(error.get_message().find("Variance tensor") != std::string::npos);
}

// ============================================================================
// Spatial Dimension Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateAcceptsSpatialDimensionEqualsOne)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 256, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Inference mode should accept N*spatial=1 (matches PyTorch behavior)";
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateAcceptsValidSpatialDimensions)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 3, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

// ============================================================================
// 5D Tensor (NCDHW) Validation Tests
// ============================================================================

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateAcceptsValid5DSpatialDimensions)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK);
}

TEST(TestBatchnormInferenceNodeVarianceExt, PreValidateAccepts5DSpatialDimensionEqualsOne)
{
    BatchnormInferenceAttributesVarianceExt batchnormAttributes;

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

    auto varianceTensor = std::make_shared<TensorAttributes>();
    varianceTensor->set_dim({1, 64, 1, 1, 1});
    batchnormAttributes.set_variance(varianceTensor);

    GraphAttributes graphAttributes;
    BatchnormInferenceNodeVarianceExt node(std::move(batchnormAttributes), graphAttributes);

    auto error = node.pre_validate_node();
    EXPECT_EQ(error.code, ErrorCode::OK)
        << "Inference mode should accept N*D*H*W=1 for 5D tensors (matches PyTorch behavior)";
}
