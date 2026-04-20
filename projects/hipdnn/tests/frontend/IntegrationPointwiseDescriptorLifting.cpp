// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <vector>

#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/node/PointwiseNode.hpp>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/constants/PointwiseConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LiftingTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using namespace hipdnn_tests::constants;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::liftGraph;
using hipdnn_tests::liftGraphWithoutFinalization;
using hipdnn_tests::TestableGraphLifting;

namespace
{
class IntegrationPointwiseDescriptorLifting : public IntegrationTestFixture
{
};

// Builds a binary pointwise (ADD) graph, lowers via build_operation_graph(handle),
// lifts back with fromBackendDescriptor(), and verifies operation mode, tensors,
// and graph-level data types.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseBinaryAddRoundTripViaCApi)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseAddLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

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

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify graph-level data types
    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_intermediate_data_type(), DataType::FLOAT);
    EXPECT_EQ(liftedGraph->get_io_data_type(), DataType::FLOAT);

    // Verify tensors by UID (in0, in1, out0)
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 3u);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_data_type(), DataType::FLOAT);

    // Verify the lifted graph has 1 pointwise sub-node
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::ADD);
    EXPECT_EQ(pwNode->attributes.get_name(), "add_op");
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Builds a unary pointwise (RELU_FWD) graph with scalar attributes,
// lowers, lifts, and verifies mode + activation parameters.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseUnaryReluScalarsPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseReluLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

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

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::RELU_FWD);

    ASSERT_TRUE(pwNode->attributes.get_relu_lower_clip().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_lower_clip().value(), K_LOWER_CLIP);

    ASSERT_TRUE(pwNode->attributes.get_relu_upper_clip().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_upper_clip().value(), K_UPPER_CLIP);

    ASSERT_TRUE(pwNode->attributes.get_relu_lower_clip_slope().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_relu_lower_clip_slope().value(), K_LOWER_CLIP_SLOPE);

    // Verify tensor UIDs
    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 2u);
    EXPECT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
}

// Builds a ternary pointwise (BINARY_SELECT) graph, lowers, lifts,
// and verifies all three inputs and the output are correctly reconstructed.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseTernarySelectRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseTernaryLiftingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

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

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    ASSERT_EQ(tensorMap.size(), 4u);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN1_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN1_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_IN2_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN2_UID]->get_data_type(), DataType::FLOAT);

    ASSERT_NE(tensorMap.count(K_PW_TENSOR_OUT0_UID), 0u);
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_dim(), toVec(K_PW_TENSOR_DIMS));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_stride(), toVec(K_PW_TENSOR_STRIDES));
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID]->get_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::BINARY_SELECT);
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_input_2()->get_uid(), K_PW_TENSOR_IN2_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Builds a pointwise graph, serializes to binary, creates a backend descriptor
// from bytes (no handle), calls fromBackendDescriptor(), and verifies the
// pointwise operation survives the FlatBuffer-direct deserialization path.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseLiftWithoutFinalization)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseFlatBufferLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    auto in1 = std::make_shared<TensorAttributes>();
    in1->set_uid(K_PW_TENSOR_IN1_UID).set_name("IN1").set_data_type(DataType::FLOAT);
    in1->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("mul_op");
    pwAttrs.set_mode(PointwiseMode::MUL);

    auto out0 = graph->pointwise(in0, in1, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraphWithoutFinalization(*graph);
    ASSERT_NE(liftedGraph, nullptr);

    EXPECT_EQ(liftedGraph->get_compute_data_type(), DataType::FLOAT);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::MUL);
    EXPECT_EQ(pwNode->attributes.get_input_0()->get_uid(), K_PW_TENSOR_IN0_UID);
    EXPECT_EQ(pwNode->attributes.get_input_1()->get_uid(), K_PW_TENSOR_IN1_UID);
    EXPECT_EQ(pwNode->attributes.get_output_0()->get_uid(), K_PW_TENSOR_OUT0_UID);
}

// Verifies that tensor objects are shared between the tensor map and the
// pointwise node attributes after lifting.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseTensorSharingPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseTensorSharingTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("relu_op");
    pwAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto tensorMap = liftedGraph->getTensorsByUid();
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr);

    // Verify tensor objects are shared (same pointer) between map and node
    EXPECT_EQ(tensorMap[K_PW_TENSOR_IN0_UID].get(), pwNode->attributes.get_input_0().get());
    EXPECT_EQ(tensorMap[K_PW_TENSOR_OUT0_UID].get(), pwNode->attributes.get_output_0().get());
}

// Builds a unary pointwise (SWISH_FWD) graph with swish_beta scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseSwishBetaPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseSwishBetaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("swish_op");
    pwAttrs.set_mode(PointwiseMode::SWISH_FWD);
    pwAttrs.set_swish_beta(0.5F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::SWISH_FWD);
    EXPECT_TRUE(pwNode->attributes.get_swish_beta().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_swish_beta().value(), 0.5F);
}

// Builds a unary pointwise (ELU_FWD) graph with elu_alpha scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseEluAlphaPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseEluAlphaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("elu_op");
    pwAttrs.set_mode(PointwiseMode::ELU_FWD);
    pwAttrs.set_elu_alpha(1.0F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::ELU_FWD);
    EXPECT_TRUE(pwNode->attributes.get_elu_alpha().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_elu_alpha().value(), 1.0F);
}

// Builds a unary pointwise (SOFTPLUS_FWD) graph with softplus_beta scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseSoftplusBetaPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseSoftplusBetaLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("softplus_op");
    pwAttrs.set_mode(PointwiseMode::SOFTPLUS_FWD);
    pwAttrs.set_softplus_beta(2.0F);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::SOFTPLUS_FWD);
    EXPECT_TRUE(pwNode->attributes.get_softplus_beta().has_value());
    EXPECT_FLOAT_EQ(pwNode->attributes.get_softplus_beta().value(), 2.0F);
}

// Builds a unary pointwise (GEN_INDEX) graph with axis scalar,
// lowers, lifts, and verifies the scalar survives the round trip.
TEST_F(IntegrationPointwiseDescriptorLifting, PointwiseGenIndexAxisPreserved)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("PointwiseGenIndexAxisLiftTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    auto in0 = std::make_shared<TensorAttributes>();
    in0->set_uid(K_PW_TENSOR_IN0_UID).set_name("IN0").set_data_type(DataType::FLOAT);
    in0->set_dim(toVec(K_PW_TENSOR_DIMS)).set_stride(toVec(K_PW_TENSOR_STRIDES));

    PointwiseAttributes pwAttrs;
    pwAttrs.set_name("gen_index_op");
    pwAttrs.set_mode(PointwiseMode::GEN_INDEX);
    pwAttrs.set_axis(2);

    auto out0 = graph->pointwise(in0, pwAttrs);
    out0->set_uid(K_PW_TENSOR_OUT0_UID).set_output(true).set_name("OUT0");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 1u);

    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[0].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected a PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)

    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::GEN_INDEX);
    EXPECT_TRUE(pwNode->attributes.get_axis().has_value());
    EXPECT_EQ(pwNode->attributes.get_axis().value(), 2);
}

// Builds a conv_fprop + pointwise RELU fusion graph, lowers via the C-API,
// lifts back, and verifies both operation nodes and the shared virtual tensor.
TEST_F(IntegrationPointwiseDescriptorLifting, ConvFpropReluFusionRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLifting>();
    graph->set_name("ConvReluFusionTest")
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_io_data_type(DataType::FLOAT);

    // Conv fprop inputs
    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_FPROP_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_FPROP_TENSOR_X_DIMS)).set_stride(toVec(K_FPROP_TENSOR_X_STRIDES));

    auto w = std::make_shared<TensorAttributes>();
    w->set_uid(K_FPROP_TENSOR_W_UID).set_name("W").set_data_type(DataType::FLOAT);
    w->set_dim(toVec(K_FPROP_TENSOR_W_DIMS)).set_stride(toVec(K_FPROP_TENSOR_W_STRIDES));

    ConvFpropAttributes convAttrs;
    convAttrs.set_name("conv_fprop_op");
    convAttrs.set_pre_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_post_padding(toVec(K_FPROP_CONV_PADDING));
    convAttrs.set_stride(toVec(K_FPROP_CONV_STRIDE));
    convAttrs.set_dilation(toVec(K_FPROP_CONV_DILATION));
    convAttrs.set_convolution_mode(ConvolutionMode::CROSS_CORRELATION);

    // Conv output y is a virtual intermediate — no UID, not an output
    auto y = graph->conv_fprop(x, w, convAttrs);

    // Pointwise RELU on the conv output
    PointwiseAttributes reluAttrs;
    reluAttrs.set_name("relu_activation");
    reluAttrs.set_mode(PointwiseMode::RELU_FWD);

    auto reluOut = graph->pointwise(y, reluAttrs);
    reluOut->set_uid(K_PW_RELU_OUT_UID).set_output(true).set_name("relu_out");

    auto liftedGraph = liftGraph(*graph, _handle);
    ASSERT_NE(liftedGraph, nullptr);

    // Verify 2 operation nodes
    auto& subNodes = liftedGraph->getSubNodes();
    ASSERT_EQ(subNodes.size(), 2u)
        << "Expected 2 operation nodes (conv + relu)"; // NOLINT(readability-implicit-bool-conversion)

    // First node: ConvolutionFpropNode
    auto* convNode = dynamic_cast<ConvolutionFpropNode*>(subNodes[0].get());
    ASSERT_NE(convNode, nullptr) << "Expected first node to be ConvolutionFpropNode";
    EXPECT_EQ(convNode->attributes.get_pre_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_post_padding(), toVec(K_FPROP_CONV_PADDING));
    EXPECT_EQ(convNode->attributes.get_stride(), toVec(K_FPROP_CONV_STRIDE));
    EXPECT_EQ(convNode->attributes.get_dilation(), toVec(K_FPROP_CONV_DILATION));
    EXPECT_EQ(convNode->attributes.get_convolution_mode(), ConvolutionMode::CROSS_CORRELATION);
    EXPECT_EQ(convNode->attributes.get_name(), "conv_fprop_op");

    // Second node: PointwiseNode with RELU_FWD
    auto* pwNode = dynamic_cast<PointwiseNode*>(subNodes[1].get());
    ASSERT_NE(pwNode, nullptr)
        << "Expected second node to be PointwiseNode"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_EQ(pwNode->attributes.get_mode(), PointwiseMode::RELU_FWD);
    EXPECT_EQ(pwNode->attributes.get_name(), "relu_activation");

    // Verify tensor sharing: conv output and relu input share the same TensorAttributes
    auto convY = convNode->attributes.get_y();
    auto reluIn0 = pwNode->attributes.get_input_0();
    EXPECT_EQ(convY.get(), reluIn0.get())
        << "Conv output and relu input should share the same "
           "TensorAttributes object"; // NOLINT(readability-implicit-bool-conversion)

    // Verify tensor map contains external tensors (X, W, relu_out) plus the virtual intermediate
    auto tensorMap = liftedGraph->getTensorsByUid();
    EXPECT_NE(tensorMap.count(K_FPROP_TENSOR_X_UID), 0u) << "X tensor not found";
    EXPECT_NE(tensorMap.count(K_FPROP_TENSOR_W_UID), 0u) << "W tensor not found";
    EXPECT_NE(tensorMap.count(K_PW_RELU_OUT_UID), 0u) << "relu_out tensor not found";
}

} // namespace
