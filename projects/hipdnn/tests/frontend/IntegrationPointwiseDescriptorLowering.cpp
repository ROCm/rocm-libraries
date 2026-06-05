// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using PointwiseModeSdk = hipdnn_flatbuffers_sdk::data_objects::PointwiseMode;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationPointwiseDescriptorLowering : public IntegrationTestFixture
{
};

// Binary pointwise (ADD) round-trip: lower and verify deserialized graph
TEST_F(IntegrationPointwiseDescriptorLowering, PointwiseGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestPointwiseGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("add_op");
    pwAttrs.set_mode(PointwiseMode::ADD);

    auto out0 = graph->pointwise(in0, in1, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors (in0, in1, out0) --
    ASSERT_EQ(graphT.tensors.size(), 3u);

    auto tensorMap = buildTensorMap(graphT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    auto* in0T = tensorMap[K_PW_TENSOR_IN0_UID];
    EXPECT_EQ(in0T->name, "IN0");
    EXPECT_EQ(in0T->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(in0T->dims, toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(in0T->strides, toVec(K_PW_TENSOR_STRIDES));

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);
    auto* in1T = tensorMap[K_PW_TENSOR_IN1_UID];
    EXPECT_EQ(in1T->name, "IN1");
    EXPECT_EQ(in1T->data_type, DataTypeSdk::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
    auto* out0T = tensorMap[K_PW_TENSOR_OUT0_UID];
    EXPECT_EQ(out0T->name, "OUT0");

    // -- Verify pointwise operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::PointwiseAttributes);

    auto* pwNode = node->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    EXPECT_EQ(pwNode->in_1_tensor_uid, K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::ADD);
    EXPECT_EQ(node->name, "add_op");
}

// Unary pointwise (RELU_FWD) round-trip with single input
TEST_F(IntegrationPointwiseDescriptorLowering, UnaryPointwiseRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestUnaryPointwiseGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("relu_op");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // Unary: 2 tensors (in0, out0)
    ASSERT_EQ(graphT.tensors.size(), 2u);
    ASSERT_EQ(graphT.nodes.size(), 1u);

    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::RELU_FWD);
}

// RELU_FWD with activation scalar parameters round-trip
TEST_F(IntegrationPointwiseDescriptorLowering, ScalarAttributesPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestScalarAttrsGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    constexpr float K_LOWER_CLIP = -1.0F;
    constexpr float K_UPPER_CLIP = 6.0F;
    constexpr float K_LOWER_CLIP_SLOPE = 0.01F;

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("leaky_relu_op");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);
    pwAttrs.set_relu_lower_clip(K_LOWER_CLIP);
    pwAttrs.set_relu_upper_clip(K_UPPER_CLIP);
    pwAttrs.set_relu_lower_clip_slope(K_LOWER_CLIP_SLOPE);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::RELU_FWD);

    ASSERT_TRUE(pwNode->relu_lower_clip.has_value());
    EXPECT_FLOAT_EQ(pwNode->relu_lower_clip.value(), K_LOWER_CLIP);

    ASSERT_TRUE(pwNode->relu_upper_clip.has_value());
    EXPECT_FLOAT_EQ(pwNode->relu_upper_clip.value(), K_UPPER_CLIP);

    ASSERT_TRUE(pwNode->relu_lower_clip_slope.has_value());
    EXPECT_FLOAT_EQ(pwNode->relu_lower_clip_slope.value(), K_LOWER_CLIP_SLOPE);
}

// Ternary pointwise (BINARY_SELECT) round-trip with 3 inputs
TEST_F(IntegrationPointwiseDescriptorLowering, TernaryPointwiseRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestTernaryPointwiseGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in2 = std::make_shared<TensorAttributes>();
    in2->set_uid(K_PW_TENSOR_IN2_UID).set_name("IN2").set_data_type(DataType::FLOAT);
    in2->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("select_op");
    pwAttrs.set_mode(PointwiseMode::BINARY_SELECT);

    auto out0 = graph->pointwise(in0, in1, in2, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // Ternary: 4 tensors (in0, in1, in2, out0)
    ASSERT_EQ(graphT.tensors.size(), 4u);
    ASSERT_EQ(graphT.nodes.size(), 1u);

    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->in_0_tensor_uid, K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->out_0_tensor_uid, K_PW_TENSOR_OUT0_UID);
    EXPECT_EQ(pwNode->in_1_tensor_uid, K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->in_2_tensor_uid, K_PW_TENSOR_IN2_UID);
    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::BINARY_SELECT);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationPointwiseDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidPointwiseGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_mode(PointwiseMode::MUL);

    auto out0 = graph->pointwise(in0, in1, pwAttrs);
    out0->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 3u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 3u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The pointwise operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(pwNode->in_0_tensor_uid) > 0)
        << "IN_0 tensor UID " << pwNode->in_0_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    ASSERT_TRUE(pwNode->in_1_tensor_uid.has_value())
        << "IN_1 tensor UID should be set"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(pwNode->in_1_tensor_uid.value()) > 0)
        << "IN_1 tensor UID " << pwNode->in_1_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(pwNode->out_0_tensor_uid) > 0)
        << "OUT_0 tensor UID " << pwNode->out_0_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All three tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids
        = {pwNode->in_0_tensor_uid, pwNode->in_1_tensor_uid.value(), pwNode->out_0_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 3u)
        << "Pointwise node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

// Additional activation scalars (swish_beta, elu_alpha, softplus_beta) round-trip
TEST_F(IntegrationPointwiseDescriptorLowering, AdditionalScalarAttributesPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestAdditionalScalarAttrsGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    constexpr float K_SWISH_BETA = 1.5F;
    constexpr float K_ELU_ALPHA = 0.25F;
    constexpr float K_SOFTPLUS_BETA = 2.0F;

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("activation_op");
    pwAttrs.set_mode(PointwiseMode::SWISH_FWD);
    pwAttrs.set_swish_beta(K_SWISH_BETA);
    pwAttrs.set_elu_alpha(K_ELU_ALPHA);
    pwAttrs.set_softplus_beta(K_SOFTPLUS_BETA);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::SWISH_FWD);

    ASSERT_TRUE(pwNode->swish_beta.has_value());
    EXPECT_FLOAT_EQ(pwNode->swish_beta.value(), K_SWISH_BETA);

    ASSERT_TRUE(pwNode->elu_alpha.has_value());
    EXPECT_FLOAT_EQ(pwNode->elu_alpha.value(), K_ELU_ALPHA);

    ASSERT_TRUE(pwNode->softplus_beta.has_value());
    EXPECT_FLOAT_EQ(pwNode->softplus_beta.value(), K_SOFTPLUS_BETA);
}

// GEN_INDEX mode with axis attribute round-trip
TEST_F(IntegrationPointwiseDescriptorLowering, AxisAttributePreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestAxisAttrGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    constexpr int64_t K_AXIS = 2;

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("gen_index_op");
    pwAttrs.set_mode(PointwiseMode::GEN_INDEX);
    pwAttrs.set_axis(K_AXIS);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* pwNode = graphT.nodes[0]->attributes.AsPointwiseAttributes();
    ASSERT_NE(pwNode, nullptr);

    EXPECT_EQ(pwNode->operation, PointwiseModeSdk::GEN_INDEX);

    ASSERT_TRUE(pwNode->axis_tensor_uid.has_value());
    EXPECT_EQ(pwNode->axis_tensor_uid.value(), K_AXIS);
}

} // namespace
