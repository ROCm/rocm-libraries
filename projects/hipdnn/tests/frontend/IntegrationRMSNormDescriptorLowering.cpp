// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;
using NormFwdPhaseSdk = hipdnn_data_sdk::data_objects::NormFwdPhase;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationRMSNormDescriptorLowering : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const std::array<const char*, 1> paths
            = {hipdnn_tests::plugin_constants::testGoodPluginPath().c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnDestroy(_handle);
        }
    }

    hipdnnHandle_t _handle = nullptr;
};

// Builds an RMSNorm graph via the frontend API (TRAINING mode with all tensors),
// lowers it to the backend via build_operation_graph_via_descriptors, retrieves
// the serialized graph, and verifies all tensor and operation attributes match
// the values set in the frontend.
TEST_F(IntegrationRMSNormDescriptorLowering, RMSNormGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestRMSNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RMSNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORM_TENSOR_X_DIMS)).set_stride(toVec(K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_RMSNORM_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_RMSNORM_TENSOR_EPSILON_UID).set_name("Epsilon");

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_op");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto outputs = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    const auto& y = outputs[0];
    const auto& invRms = outputs[1];

    y->set_uid(K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    ASSERT_NE(invRms, nullptr); // should be created in TRAINING mode
    invRms->set_uid(K_RMSNORM_TENSOR_INV_RMS_UID).set_output(true).set_name("InvRms");

    // -- Validate and lower --
    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // -- Retrieve serialized graph --
    auto rawDesc = graph->get_raw_graph_descriptor();
    ASSERT_NE(rawDesc, nullptr);

    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    // -- Deserialize into GraphT --
    auto graphFb = hipdnn_data_sdk::data_objects::GetGraph(serializedData.data());
    ASSERT_NE(graphFb, nullptr);
    hipdnn_data_sdk::data_objects::GraphT graphT;
    graphFb->UnPackTo(&graphT);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    // X, Scale, Epsilon, Y, InvRms = 5 tensors (bias intentionally omitted from this test)
    ASSERT_EQ(graphT.tensors.size(), 5u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_RMSNORM_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_RMSNORM_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_RMSNORM_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_RMSNORM_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);

    // Verify Epsilon tensor
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_EPSILON_UID), 0u);
    auto* epsilonT = tensorMap[K_RMSNORM_TENSOR_EPSILON_UID];
    EXPECT_EQ(epsilonT->name, "Epsilon");
    EXPECT_EQ(epsilonT->data_type, DataTypeSdk::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_RMSNORM_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(yT->virtual_);

    // Verify InvRms tensor
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_INV_RMS_UID), 0u);
    auto* invRmsT = tensorMap[K_RMSNORM_TENSOR_INV_RMS_UID];
    EXPECT_EQ(invRmsT->name, "InvRms");
    EXPECT_EQ(invRmsT->data_type, DataTypeSdk::FLOAT);

    // -- Verify rmsnorm operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::RMSNormAttributes);

    auto* rmsnorm = node->attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnorm, nullptr);

    EXPECT_EQ(rmsnorm->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsnorm->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsnorm->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsnorm->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_TRUE(rmsnorm->inv_rms_tensor_uid.has_value());
    EXPECT_EQ(*rmsnorm->inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(rmsnorm->forward_phase, NormFwdPhaseSdk::TRAINING);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationRMSNormDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidRMSNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORM_TENSOR_X_DIMS)).set_stride(toVec(K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_name("Epsilon");

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto outputs = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    outputs[0]->set_output(true);
    if(outputs[1])
    {
        outputs[1]->set_output(true);
    }

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve serialized graph
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    ASSERT_GT(serializedSize, 0u);

    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // All tensors should have been auto-assigned unique UIDs:
    // X, Scale, Epsilon, Y, InvRms = 5 tensors
    ASSERT_EQ(graphT.tensors.size(), 5u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 5u) << "Tensor UIDs are not unique";

    // The rmsnorm operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* rmsnorm = graphT.nodes[0]->attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnorm, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(rmsnorm->x_tensor_uid) > 0)
        << "X tensor UID " << rmsnorm->x_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(rmsnorm->scale_tensor_uid) > 0)
        << "Scale tensor UID " << rmsnorm->scale_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(rmsnorm->epsilon_tensor_uid) > 0)
        << "Epsilon tensor UID " << rmsnorm->epsilon_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(rmsnorm->y_tensor_uid) > 0)
        << "Y tensor UID " << rmsnorm->y_tensor_uid << " not found in graph tensors";
    if(rmsnorm->inv_rms_tensor_uid.has_value())
    {
        EXPECT_TRUE(uids.count(*rmsnorm->inv_rms_tensor_uid) > 0)
            << "InvRms tensor UID " << *rmsnorm->inv_rms_tensor_uid
            << " not found in graph tensors";
    }

    // All required tensor UIDs referenced by the node should be distinct
    std::unordered_set<int64_t> nodeUids = {rmsnorm->x_tensor_uid,
                                            rmsnorm->scale_tensor_uid,
                                            rmsnorm->epsilon_tensor_uid,
                                            rmsnorm->y_tensor_uid};
    if(rmsnorm->inv_rms_tensor_uid.has_value())
    {
        nodeUids.insert(*rmsnorm->inv_rms_tensor_uid);
    }
    EXPECT_EQ(nodeUids.size(), 5u) << "RMSNorm node tensor UIDs are not distinct";
}

// Roundtrip with optional bias tensor set, verifying it appears in the serialized graph.
TEST_F(IntegrationRMSNormDescriptorLowering, RMSNormWithBiasRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("BiasRMSNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RMSNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORM_TENSOR_X_DIMS)).set_stride(toVec(K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_RMSNORM_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_RMSNORM_TENSOR_EPSILON_UID).set_name("Epsilon");

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_RMSNORM_TENSOR_BIAS_UID).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_RMSNORM_TENSOR_BIAS_DIMS))
        .set_stride(toVec(K_RMSNORM_TENSOR_BIAS_STRIDES));

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_op");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_bias(bias);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::TRAINING);

    auto outputs = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    const auto& y = outputs[0];
    const auto& invRms = outputs[1];
    y->set_uid(K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    ASSERT_NE(invRms, nullptr);
    invRms->set_uid(K_RMSNORM_TENSOR_INV_RMS_UID).set_output(true).set_name("InvRms");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve and deserialize
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // X, Scale, Epsilon, Bias, Y, InvRms = 6 tensors
    ASSERT_EQ(graphT.tensors.size(), 6u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Bias tensor should be present
    ASSERT_NE(tensorMap.count(K_RMSNORM_TENSOR_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_RMSNORM_TENSOR_BIAS_UID];
    EXPECT_EQ(biasT->name, "Bias");
    EXPECT_EQ(biasT->data_type, DataTypeSdk::FLOAT);

    // Node should reference bias
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* rmsnorm = graphT.nodes[0]->attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnorm, nullptr);
    ASSERT_TRUE(rmsnorm->bias_tensor_uid.has_value());
    EXPECT_EQ(*rmsnorm->bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_TRUE(rmsnorm->inv_rms_tensor_uid.has_value());
    EXPECT_EQ(*rmsnorm->inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(rmsnorm->forward_phase, NormFwdPhaseSdk::TRAINING);
}

// Inference mode: inv_rms should not appear in the serialized graph.
TEST_F(IntegrationRMSNormDescriptorLowering, InferenceModeOmitsInvRms)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("InferenceRMSNormGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RMSNORM_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORM_TENSOR_X_DIMS)).set_stride(toVec(K_RMSNORM_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_RMSNORM_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORM_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>(1e-5f);
    epsilon->set_uid(K_RMSNORM_TENSOR_EPSILON_UID).set_name("Epsilon");

    RMSNormAttributes rmsnormAttrs;
    rmsnormAttrs.set_name("rmsnorm_op");
    rmsnormAttrs.set_epsilon(epsilon);
    rmsnormAttrs.set_forward_phase(NormFwdPhase::INFERENCE);

    auto outputs = graph->rmsnorm(x, scale, std::move(rmsnormAttrs));
    const auto& y = outputs[0];
    y->set_uid(K_RMSNORM_TENSOR_Y_UID).set_output(true).set_name("Y");
    // outputs[1] (inv_rms) should be nullptr in INFERENCE mode

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve and deserialize
    auto rawDesc = graph->get_raw_graph_descriptor();
    size_t serializedSize = 0;
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(rawDesc, 0, &serializedSize, nullptr),
              HIPDNN_STATUS_SUCCESS);
    std::vector<uint8_t> serializedData(serializedSize);
    ASSERT_EQ(hipdnnBackendGetSerializedBinaryGraph_ext(
                  rawDesc, serializedSize, &serializedSize, serializedData.data()),
              HIPDNN_STATUS_SUCCESS);

    hipdnn_data_sdk::data_objects::GraphT graphT;
    hipdnn_data_sdk::data_objects::GetGraph(serializedData.data())->UnPackTo(&graphT);

    // X, Scale, Epsilon, Y = 4 tensors (no inv_rms, no bias)
    ASSERT_EQ(graphT.tensors.size(), 4u);

    // Node should not reference inv_rms or bias
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* rmsnorm = graphT.nodes[0]->attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnorm, nullptr);
    EXPECT_FALSE(rmsnorm->inv_rms_tensor_uid.has_value());
    EXPECT_FALSE(rmsnorm->bias_tensor_uid.has_value());
    EXPECT_EQ(rmsnorm->forward_phase, NormFwdPhaseSdk::INFERENCE);
}

} // namespace
