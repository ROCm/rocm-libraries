// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include "engines/plans/batchnorm/BatchnormApplicabilityChecks.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

using namespace hip_kernel_provider;

TEST(TestBatchnormValidator, ValidInference)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkInferenceTensorConfigSupported(attr));
}

TEST(TestBatchnormValidator, ValidInferenceActiv)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdInferActGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs));
}

TEST(TestBatchnormValidator, ValidVarianceExtInference)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkInferenceVarianceExtTensorConfigSupported(attr));
}

TEST(TestBatchnormValidator, ValidVarianceExtInferenceActiv)
{
    auto builder
        = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceActivGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributesVarianceExt();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(
        validator.checkInferenceVarianceExtActivationTensorConfigSupported(attr, activAttrs));
}

TEST(TestBatchnormValidator, ValidFwdTraining)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkFwdTrainingTensorConfigSupported(attr));
}

TEST(TestBatchnormValidator, ValidBwd)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormBackwardAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkBwdTensorConfigSupported(attr));
}

TEST(TestBatchnormValidator, ValidInferenceActivationBackward)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferActBwdGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& bnInfNode = graph.getNode(0);
    const auto& actNode = graph.getNode(1);
    const auto& bnBwdNode = graph.getNode(2);

    const auto& bnInfAttr = *bnInfNode.attributes_as_BatchnormInferenceAttributes();
    const auto& actAttr = *actNode.attributes_as_PointwiseAttributes();
    const auto& bnBwdAttr = *bnBwdNode.attributes_as_BatchnormBackwardAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkInferenceActivationBackwardTensorConfigSupported(
        bnInfAttr, actAttr, bnBwdAttr));
}

TEST(TestBatchnormValidator, MismatchShapes)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph();
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());
    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_NO_THROW(validator.checkInferenceTensorConfigSupported(attr));
}

namespace
{

flatbuffers::FlatBufferBuilder createInvalidTypeBatchnormActivGraph(
    hipdnn_flatbuffers_sdk::data_objects::DataType xType,
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleType,
    hipdnn_flatbuffers_sdk::data_objects::DataType biasType,
    hipdnn_flatbuffers_sdk::data_objects::DataType meanType,
    hipdnn_flatbuffers_sdk::data_objects::DataType invVarianceType,
    hipdnn_flatbuffers_sdk::data_objects::DataType yType,
    hipdnn_flatbuffers_sdk::data_objects::DataType dyType,
    hipdnn_flatbuffers_sdk::data_objects::DataType computeDataType,
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode activ
    = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    std::vector<int64_t> strides{150528, 50176, 224, 1};
    std::vector<int64_t> dims{1, 3, 224, 224};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    // inputs
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", xType, &strides, &dims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 2, "scale", scaleType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 3, "bias", biasType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 4, "mean", meanType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 5, "inv_variance", invVarianceType, &derivedStrides, &derivedDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 6, "y", yType, &strides, &dims, true));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 7, "Dy", dyType, &strides, &dims, false)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        4, // mean_tensor_uid
        5, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        6 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise
    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        activ,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        6, // in_0_tensor_uid (BN_Y)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        7 // out_0_tensor_uid (Dy - not virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_fwd",
        computeDataType,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestBatchnormValidator, MismatchIOTypes)
{
    auto builder = createInvalidTypeBatchnormActivGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchAffineTypes)
{
    auto builder = createInvalidTypeBatchnormActivGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of affine tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchStatTypes)
{
    auto builder = createInvalidTypeBatchnormActivGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of stat tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchIntermediateTypes)
{
    auto builder = createInvalidTypeBatchnormActivGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of intermediate tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, InvalidActivation)
{
    auto builder = createInvalidTypeBatchnormActivGraph(
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RECIPROCAL);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Plan doesn't change to reciprocal activation mode yet
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

flatbuffers::FlatBufferBuilder
    createInvalidShapeBatchnormActivGraph(const std::vector<int64_t>& xDims,
                                          const std::vector<int64_t>& xStrides,
                                          const std::vector<int64_t>& scaleDims,
                                          const std::vector<int64_t>& scaleStrides,
                                          const std::vector<int64_t>& biasDims,
                                          const std::vector<int64_t>& biasStrides,
                                          const std::vector<int64_t>& meanDims,
                                          const std::vector<int64_t>& meansStrides,
                                          const std::vector<int64_t>& invVarianceDims,
                                          const std::vector<int64_t>& invVarianceStrides,
                                          const std::vector<int64_t>& yDims,
                                          const std::vector<int64_t>& yStrides,
                                          const std::vector<int64_t>& dyDims,
                                          const std::vector<int64_t>& dyStrides)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    // inputs
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &xStrides, &xDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &scaleStrides,
        &scaleDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &biasStrides,
        &biasDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "mean",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &meansStrides,
        &meanDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        5,
        "inv_variance",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &invVarianceStrides,
        &invVarianceDims));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &yStrides,
        &yDims,
        true));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "Dy",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &dyStrides,
        &dyDims,
        false)); // is_virtual = true

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // Node 0: Batchnorm Inference
    auto bnInfAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormInferenceAttributes(
        builder,
        1, // x_tensor_uid
        4, // mean_tensor_uid
        5, // inv_variance_tensor_uid
        2, // scale_tensor_uid
        3, // bias_tensor_uid
        6 // y_tensor_uid (BN_Y - virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "batchnorm_inference",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormInferenceAttributes,
        bnInfAttributes.Union()));

    // Node 1: Pointwise
    auto pointwiseAttributes = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        std::nullopt, // relu_lower_clip
        std::nullopt, // relu_upper_clip
        std::nullopt, // relu_lower_clip_slope
        flatbuffers::nullopt, // axis_tensor_uid
        6, // in_0_tensor_uid (BN_Y)
        flatbuffers::nullopt, // in_1_tensor_uid
        flatbuffers::nullopt, // in_2_tensor_uid
        7 // out_0_tensor_uid (Dy - not virtual)
    );

    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "relu_fwd",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        pointwiseAttributes.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);
    return builder;
}
} // anonymous namespace

TEST(TestBatchnormValidator, MismatchIOShapes)
{
    std::vector<int64_t> xDims{1, 3, 4, 4};
    std::vector<int64_t> xStrides{48, 16, 4, 1};

    std::vector<int64_t> dims{1, 3, 224, 224};
    std::vector<int64_t> strides{150528, 50176, 224, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    auto builder = createInvalidShapeBatchnormActivGraph(xDims,
                                                         xStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         dims,
                                                         strides,
                                                         dims,
                                                         strides);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchAffineShapes)
{
    std::vector<int64_t> dims{1, 3, 224, 224};
    std::vector<int64_t> strides{150528, 50176, 224, 1};

    std::vector<int64_t> scaleDims{1, 3, 4, 4};
    std::vector<int64_t> scaleStrides{48, 16, 4, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    auto builder = createInvalidShapeBatchnormActivGraph(dims,
                                                         strides,
                                                         scaleDims,
                                                         scaleStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         dims,
                                                         strides,
                                                         dims,
                                                         strides);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormValidator, MismatchStatShapes)
{
    std::vector<int64_t> meanDims{1, 3, 4, 4};
    std::vector<int64_t> meanStrides{48, 16, 4, 1};

    std::vector<int64_t> dims{1, 3, 224, 224};
    std::vector<int64_t> strides{150528, 50176, 224, 1};

    const std::vector<int64_t> derivedDims = hipdnn_data_sdk::utilities::getDerivedShape(dims);
    const std::vector<int64_t> derivedStrides = hipdnn_data_sdk::utilities::generateStrides(
        derivedDims, hipdnn_data_sdk::utilities::extractStrideOrder(strides));

    auto builder = createInvalidShapeBatchnormActivGraph(dims,
                                                         strides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         meanDims,
                                                         meanStrides,
                                                         derivedDims,
                                                         derivedStrides,
                                                         dims,
                                                         strides,
                                                         dims,
                                                         strides);

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(builder.GetBufferPointer(),
                                                                     builder.GetSize());

    const auto& node = graph.getNode(0);
    const auto& attr = *node.attributes_as_BatchnormInferenceAttributes();

    const auto& activNode = graph.getNode(1);
    const auto& activAttrs = *activNode.attributes_as_PointwiseAttributes();

    // Data type of io tensors should match, expect exception when this isn't the case
    BatchnormValidator validator(graph.getTensorMap());
    EXPECT_THROW(validator.checkInferenceActivationTensorConfigSupported(attr, activAttrs),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}
