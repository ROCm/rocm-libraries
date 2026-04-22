// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/LayernormConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using NormFwdPhaseSdk = hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationLayerNormDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a layernorm graph via the frontend API (training phase, with mean/inv_variance),
// lowers it to the backend via build_operation_graph_via_descriptors, retrieves the
// serialized graph, and verifies all tensor and operation attributes match.
TEST_F(IntegrationLayerNormDescriptorLowering, LayerNormGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestLayerNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_LAYERNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_LAYERNORM_TENSOR_X_DIMS)).set_stride(toVec(K_LAYERNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_LAYERNORM_TENSOR_SCALE_UID).set_name("SCALE").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_LAYERNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_LAYERNORM_TENSOR_BIAS_UID).set_name("BIAS").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_LAYERNORM_TENSOR_BIAS_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(K_LAYERNORM_TENSOR_EPSILON_UID)
        .set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES))
        .set_value(1e-5f);

    LayernormAttributes layernormAttrs;
    layernormAttrs.set_name("layernorm_op");
    layernormAttrs.set_forward_phase(NormFwdPhase::TRAINING);
    layernormAttrs.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph->layernorm(x, scale, bias, layernormAttrs);
    y->set_uid(K_LAYERNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    ASSERT_NE(mean, nullptr); // always set in TRAINING mode
    ASSERT_NE(invVariance, nullptr); // always set in TRAINING mode
    mean->set_uid(K_LAYERNORM_TENSOR_MEAN_UID).set_output(true).set_name("MEAN");
    invVariance->set_uid(K_LAYERNORM_TENSOR_INV_VARIANCE_UID)
        .set_output(true)
        .set_name("INV_VARIANCE");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors -- training phase yields 7 tensors (x, scale, bias, epsilon, y, mean, inv_variance)
    ASSERT_EQ(graphT.tensors.size(), 7u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_LAYERNORM_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_LAYERNORM_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_LAYERNORM_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify SCALE tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_LAYERNORM_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "SCALE");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(scaleT->dims, toVec(K_LAYERNORM_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));

    // Verify BIAS tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_LAYERNORM_TENSOR_BIAS_UID];
    EXPECT_EQ(biasT->name, "BIAS");
    EXPECT_EQ(biasT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(biasT->dims, toVec(K_LAYERNORM_TENSOR_BIAS_DIMS));
    EXPECT_EQ(biasT->strides, toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));

    // Verify EPSILON tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_EPSILON_UID), 0u);
    auto* epsilonT = tensorMap[K_LAYERNORM_TENSOR_EPSILON_UID];
    EXPECT_EQ(epsilonT->name, "EPSILON");
    EXPECT_EQ(epsilonT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(epsilonT->dims, toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS));
    EXPECT_EQ(epsilonT->strides, toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES));

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_LAYERNORM_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(yT->dims, toVec(K_LAYERNORM_TENSOR_Y_DIMS));
    EXPECT_EQ(yT->strides, toVec(K_LAYERNORM_TENSOR_Y_STRIDES));
    EXPECT_FALSE(yT->virtual_);

    // Verify MEAN tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_MEAN_UID), 0u);
    auto* meanT = tensorMap[K_LAYERNORM_TENSOR_MEAN_UID];
    EXPECT_EQ(meanT->name, "MEAN");
    EXPECT_EQ(meanT->uid, K_LAYERNORM_TENSOR_MEAN_UID);
    EXPECT_EQ(meanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(meanT->dims, toVec(K_LAYERNORM_TENSOR_MEAN_DIMS));
    EXPECT_EQ(meanT->strides, toVec(K_LAYERNORM_TENSOR_MEAN_STRIDES));

    // Verify INV_VARIANCE tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_INV_VARIANCE_UID), 0u);
    auto* invVarianceT = tensorMap[K_LAYERNORM_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(invVarianceT->name, "INV_VARIANCE");
    EXPECT_EQ(invVarianceT->uid, K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(invVarianceT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(invVarianceT->dims, toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS));
    EXPECT_EQ(invVarianceT->strides, toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES));

    // -- Verify layernorm operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->name, "layernorm_op");
    EXPECT_EQ(node->attributes.type, NodeAttrType::LayernormAttributes);

    auto* layernorm = node->attributes.AsLayernormAttributes();
    ASSERT_NE(layernorm, nullptr);

    EXPECT_EQ(layernorm->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(layernorm->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(layernorm->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(layernorm->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(layernorm->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_TRUE(layernorm->mean_tensor_uid.has_value());
    EXPECT_EQ(layernorm->mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    EXPECT_TRUE(layernorm->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(layernorm->inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(layernorm->forward_phase, NormFwdPhaseSdk::TRAINING);
    // X_DIMS={2,64,32,32}, SCALE_DIMS={1,64,32,32}: 3 trailing dims are normalized
    EXPECT_EQ(layernorm->normalized_dim_count, 3);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationLayerNormDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidLayerNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_LAYERNORM_TENSOR_X_DIMS)).set_stride(toVec(K_LAYERNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("SCALE").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_LAYERNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("BIAS").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_LAYERNORM_TENSOR_BIAS_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES))
        .set_value(1e-5f);

    LayernormAttributes layernormAttrs;
    layernormAttrs.set_forward_phase(NormFwdPhase::TRAINING);
    layernormAttrs.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph->layernorm(x, scale, bias, layernormAttrs);
    y->set_output(true);
    ASSERT_NE(mean, nullptr); // always set in TRAINING mode
    ASSERT_NE(invVariance, nullptr); // always set in TRAINING mode
    mean->set_output(true);
    invVariance->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // Training phase produces 7 tensors (x, scale, bias, epsilon, y, mean, inv_variance)
    ASSERT_EQ(graphT.tensors.size(), 7u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 7u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The layernorm operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* layernorm = graphT.nodes[0]->attributes.AsLayernormAttributes();
    ASSERT_NE(layernorm, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(layernorm->x_tensor_uid) > 0)
        << "X tensor UID " << layernorm->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(layernorm->scale_tensor_uid) > 0)
        << "Scale tensor UID " << layernorm->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(layernorm->bias_tensor_uid) > 0)
        << "Bias tensor UID " << layernorm->bias_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(layernorm->epsilon_tensor_uid) > 0)
        << "Epsilon tensor UID " << layernorm->epsilon_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(layernorm->y_tensor_uid) > 0)
        << "Y tensor UID " << layernorm->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(layernorm->mean_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(layernorm->mean_tensor_uid.value()) > 0)
        << "Mean tensor UID "
        << layernorm->mean_tensor_uid.value() // NOLINT(readability-implicit-bool-conversion)
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(layernorm->inv_variance_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(layernorm->inv_variance_tensor_uid.value()) > 0)
        << "InvVariance tensor UID "
        << layernorm->inv_variance_tensor_uid
               .value() // NOLINT(readability-implicit-bool-conversion)
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {layernorm->x_tensor_uid,
                                                  layernorm->scale_tensor_uid,
                                                  layernorm->bias_tensor_uid,
                                                  layernorm->epsilon_tensor_uid,
                                                  layernorm->y_tensor_uid,
                                                  layernorm->mean_tensor_uid.value(),
                                                  layernorm->inv_variance_tensor_uid.value()};
    EXPECT_EQ(nodeUids.size(), 7u)
        << "LayerNorm node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

// Inference mode: mean and inv_variance should not appear in the serialized graph.
TEST_F(IntegrationLayerNormDescriptorLowering, InferenceModeOmitsMeanAndInvVariance)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("InferenceLayerNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_LAYERNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_LAYERNORM_TENSOR_X_DIMS)).set_stride(toVec(K_LAYERNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_LAYERNORM_TENSOR_SCALE_UID).set_name("SCALE").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_LAYERNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_LAYERNORM_TENSOR_BIAS_UID).set_name("BIAS").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_LAYERNORM_TENSOR_BIAS_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(K_LAYERNORM_TENSOR_EPSILON_UID)
        .set_name("EPSILON")
        .set_data_type(DataType::FLOAT)
        .set_dim(toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS))
        .set_stride(toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES))
        .set_value(1e-5f);

    LayernormAttributes layernormAttrs;
    layernormAttrs.set_name("layernorm_op");
    layernormAttrs.set_forward_phase(NormFwdPhase::INFERENCE);
    layernormAttrs.set_epsilon(epsilon);

    auto [y, mean, invVariance] = graph->layernorm(x, scale, bias, layernormAttrs);
    y->set_uid(K_LAYERNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    // mean and invVariance should be nullptr in INFERENCE mode
    EXPECT_EQ(mean, nullptr);
    EXPECT_EQ(invVariance, nullptr);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors -- inference phase yields 5 tensors (x, scale, bias, epsilon, y)
    // No mean or inv_variance tensors
    ASSERT_EQ(graphT.tensors.size(), 5u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_LAYERNORM_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_LAYERNORM_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_LAYERNORM_TENSOR_X_STRIDES));

    // Verify SCALE tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_SCALE_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORM_TENSOR_SCALE_UID]->name, "SCALE");

    // Verify BIAS tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_BIAS_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORM_TENSOR_BIAS_UID]->name, "BIAS");

    // Verify EPSILON tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_EPSILON_UID), 0u);
    EXPECT_EQ(tensorMap[K_LAYERNORM_TENSOR_EPSILON_UID]->name, "EPSILON");

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_LAYERNORM_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_LAYERNORM_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(yT->virtual_);

    // -- Verify layernorm operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->name, "layernorm_op");
    EXPECT_EQ(node->attributes.type, NodeAttrType::LayernormAttributes);

    auto* layernorm = node->attributes.AsLayernormAttributes();
    ASSERT_NE(layernorm, nullptr);

    EXPECT_EQ(layernorm->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(layernorm->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(layernorm->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(layernorm->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(layernorm->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    // mean and inv_variance should not be set in inference mode
    EXPECT_FALSE(layernorm->mean_tensor_uid.has_value());
    EXPECT_FALSE(layernorm->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(layernorm->forward_phase, NormFwdPhaseSdk::INFERENCE);
}

} // namespace
