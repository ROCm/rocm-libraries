// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "engines/hip_mlops_engine/plans/batchnorm/BatchnormFwdTrainingPlan.hpp"
#include "mocks/MockCompiledProgram.hpp"
#include "mocks/MockKernelCompiler.hpp"
#include "mocks/MockRunnableKernel.hpp"
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "../TestPlanCommon.hpp"

namespace hip_kernel_provider::batchnorm::test
{

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, ConstructGraphWithActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_NO_THROW(
        const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap()));
}

TEST(TestBatchnormFwdTrainingActivParams, InitializesRequiredTensorsFromValidGraph)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // All required tensors should be initialized
    EXPECT_NE(params.x(), nullptr);
    EXPECT_NE(params.y(), nullptr);
    EXPECT_NE(params.scale(), nullptr);
    EXPECT_NE(params.bias(), nullptr);
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_NE(params.optActivation(), std::nullopt);
    EXPECT_NE(params.activationOut(), nullptr);
}

TEST(TestBatchnormFwdTrainingActivParams, TensorPointersMatchExpectedUidsForGraphWithActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_EQ(params.x()->uid(), bnAttrs->x_tensor_uid());
    EXPECT_EQ(params.y()->uid(), bnAttrs->y_tensor_uid());
    EXPECT_EQ(params.scale()->uid(), bnAttrs->scale_tensor_uid());
    EXPECT_EQ(params.bias()->uid(), bnAttrs->bias_tensor_uid());
    EXPECT_EQ(params.mean()->uid(), bnAttrs->mean_tensor_uid());
    EXPECT_EQ(params.invVariance()->uid(), bnAttrs->inv_variance_tensor_uid());
    EXPECT_EQ(params.activationOut()->uid(), activAttrs->out_0_tensor_uid());
}

TEST(TestBatchnormFwdTrainingActivParams, ExtractsEpsilonValueCorrectly)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    // Epsilon should be extracted as double
    EXPECT_NEAR(params.epsilonValue(), 1e-5, 1e-10);
}

// ============================================================================
// Optional Feature Tests (Mean/Variance and Running Statistics)
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVariancePresent)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(true);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_TRUE(params.hasSaveMeanVariance());
    EXPECT_NE(params.mean(), nullptr);
    EXPECT_EQ(params.mean()->uid(), bnAttrs->mean_tensor_uid());
    EXPECT_NE(params.invVariance(), nullptr);
    EXPECT_EQ(params.invVariance()->uid(), bnAttrs->inv_variance_tensor_uid());
}

TEST(TestBatchnormFwdTrainingActivParams, HandlesMeanVarianceMissing)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(false);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasSaveMeanVariance());
}

TEST(TestBatchnormFwdTrainingActivParams, HasRunningStatsReturnsFalseWhenNotProvided)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph();
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    const BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());

    EXPECT_FALSE(params.hasRunningStats());
}

// ============================================================================
// Missing Tensor Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, ThrowsStdOutOfRangeForMissingEpsilonTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    // Missing epsilon tensor (ID 5)

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node referencing non-existent epsilon tensor
    auto bnormAttributes = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(
        builder,
        1,
        3,
        4,
        5, // Epsilon tensor that doesn't exist
        0,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        3);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    // Constructor uses tensorMap.at() which throws std::out_of_range for missing key
    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 std::out_of_range);
}

// ============================================================================
// Tensor Connectivity Validation Tests
// ============================================================================

TEST(TestBatchnormFwdTrainingActivParams, ThrowsWhenBnOutputDoesNotMatchActivationInput)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y_bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        6,
        "y_wrong",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node with output tensor 2
    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2, // BN output
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    // Activation with input tensor 6 (wrong - should be 2)
    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        6, // Wrong input - doesn't match BN output
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredXTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    // Missing x tensor (ID 1)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node referencing missing x tensor
    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1, // Missing x tensor
                                                                          3,
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredScaleTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    // Missing scale tensor (ID 3)
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        4,
        "bias",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node referencing missing scale tensor
    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3, // Missing scale tensor
                                                                          4,
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, ThrowsForMissingRequiredBiasTensor)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::TensorAttributes>>
        tensorAttributes;

    const std::vector<int64_t> strides = {1, 3, 14, 14};
    const std::vector<int64_t> dims = {1, 3, 14, 14};
    const std::vector<int64_t> derivedStrides = {1, 3, 1, 1};
    const std::vector<int64_t> derivedDims = {1, 3, 1, 1};

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder, 1, "x", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, &strides, &dims));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        2,
        "y",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims,
        true));
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        3,
        "scale",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &derivedStrides,
        &derivedDims));
    // Missing bias tensor (ID 4)

    const hipdnn_flatbuffers_sdk::data_objects::Float32Value epsilonVal(1e-5f);
    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributes(
        builder,
        5,
        builder.CreateString("epsilon"),
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        0,
        0,
        false,
        hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value,
        builder.CreateStruct(epsilonVal).Union()));

    tensorAttributes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateTensorAttributesDirect(
        builder,
        7,
        "final_out",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        &strides,
        &dims));

    std::vector<::flatbuffers::Offset<hipdnn_flatbuffers_sdk::data_objects::Node>> nodes;

    // BN node referencing missing bias tensor
    auto bnormAttributes
        = hipdnn_flatbuffers_sdk::data_objects::CreateBatchnormAttributes(builder,
                                                                          1,
                                                                          3,
                                                                          4, // Missing bias tensor
                                                                          5,
                                                                          0,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          2,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt,
                                                                          flatbuffers::nullopt);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "bn",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::BatchnormAttributes,
        bnormAttributes.Union()));

    auto actAttr = hipdnn_flatbuffers_sdk::data_objects::CreatePointwiseAttributes(
        builder,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        2,
        flatbuffers::nullopt,
        flatbuffers::nullopt,
        7);
    nodes.push_back(hipdnn_flatbuffers_sdk::data_objects::CreateNodeDirect(
        builder,
        "act",
        hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET,
        hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes,
        actAttr.Union()));

    auto graphOffset = hipdnn_flatbuffers_sdk::data_objects::CreateGraphDirect(
        builder,
        "test",
        hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT,
        hipdnn_flatbuffers_sdk::data_objects::DataType::HALF,
        hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16,
        &tensorAttributes,
        &nodes);
    builder.Finish(graphOffset);

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& actNode = graph.getNode(1);
    auto* activAttrs = actNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

TEST(TestBatchnormFwdTrainingActivParams, AcceptsReluFwdActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_NO_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()));
}

TEST(TestBatchnormFwdTrainingActivParams, RejectsUnsupportedSwishFwdActivation)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        false, false, hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::SWISH_FWD);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();
    ASSERT_NE(bnAttrs, nullptr);

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();
    ASSERT_NE(activAttrs, nullptr);

    EXPECT_THROW(BatchnormFwdTrainingParams(*bnAttrs, *activAttrs, graph.getTensorMap()),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

namespace
{

std::pair<flatbuffers::FlatBufferBuilder, BatchnormFwdTrainingPlan>
    createPlanFromGraph(const std::vector<int64_t>& strides = {150528, 50176, 224, 1},
                        const std::vector<int64_t>& dims = {1, 3, 224, 224},
                        bool withMeanVariance = true,
                        bool withRunningStats = true)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingActivGraph(
        withMeanVariance,
        withRunningStats,
        hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::RELU_FWD,
        strides,
        dims);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graph(
        builder.GetBufferPointer(), builder.GetSize());

    const auto& bnNode = graph.getNode(0);
    const auto* bnAttrs = bnNode.attributes_as_BatchnormAttributes();

    const auto& activNode = graph.getNode(1);
    auto* activAttrs = activNode.attributes_as_PointwiseAttributes();

    BatchnormFwdTrainingParams params(*bnAttrs, *activAttrs, graph.getTensorMap());
    return {std::move(builder), BatchnormFwdTrainingPlan{std::move(params)}};
}

} // namespace

TEST(TestBatchnormFwdTrainingActivPlan, CompileDefaultSetsCorrectDefines)
{
    const MockKernelCompiler mockCompiler;

    std::vector<std::string> capturedOptions;
    EXPECT_CALL(mockCompiler, compile(::testing::_, ::testing::_))
        .WillOnce([&](const std::string&, const std::vector<std::string>& options) {
            capturedOptions = options;
            auto kernel = std::make_unique<MockRunnableKernel>();
            EXPECT_CALL(*kernel, setBlockSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            EXPECT_CALL(*kernel, setGridSize(::testing::_, ::testing::_, ::testing::_)).Times(1);
            auto program = std::make_unique<MockCompiledProgram>();
            EXPECT_CALL(*program, getKernel(::testing::_))
                .WillOnce(::testing::Return(::testing::ByMove(std::move(kernel))));
            return program;
        });

    auto [fbb, plan] = createPlanFromGraph();
    auto deviceProps = createTestDeviceProps();

    plan.compile(mockCompiler, deviceProps);

    auto hasOption = [&](const std::string& opt) {
        return std::find(capturedOptions.begin(), capturedOptions.end(), opt)
               != capturedOptions.end();
    };

    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP32=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFP16=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_FPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_BFPMIX=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_SAVE_MEAN_VARIANCE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_RUNNING_RESULT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_C=3"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_HW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_INHW=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NHW=50176"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_CHW=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NCHW=150528"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2_FINAL=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP0=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP1=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_GRP2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NGRPS=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NGRPS2=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_N_ELEMENTS=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_MAXN=65"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_USE_AMDGCN=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXN=768"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDS_SIZE=1024"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VEC_SIZE=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LOOP_UNROLL_MAXHW=2500"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NODPP=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_LDSGCN_SIZE=16"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_USESAVED=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VECTORIZE=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_STASH_METHOD=0"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_VARIANT=1"));
    EXPECT_TRUE(hasOption("-DHIP_PLUGIN_BN_NRN_OP_ID=3"));
}

} // namespace hip_kernel_provider::batchnorm::test
