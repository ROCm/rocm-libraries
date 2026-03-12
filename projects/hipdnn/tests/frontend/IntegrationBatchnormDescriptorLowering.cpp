// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_data_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_data_sdk::data_objects::NodeAttributes;

namespace
{

// Exposes protected Graph methods for testing
class TestableGraph : public Graph
{
public:
    using Graph::build_operation_graph_via_descriptors;
    using Graph::get_raw_graph_descriptor;
};

// -- Test constants for BatchnormGraphRoundTrip --

constexpr int64_t K_TENSOR_X_UID = 500;
constexpr int64_t K_TENSOR_SCALE_UID = 501;
constexpr int64_t K_TENSOR_BIAS_UID = 502;
constexpr int64_t K_TENSOR_EPSILON_UID = 503;

constexpr std::array<int64_t, 4> K_TENSOR_DATA_DIMS = {2, 64, 16, 16};
constexpr std::array<int64_t, 4> K_TENSOR_DATA_STRIDES = {16384, 256, 16, 1};
constexpr std::array<int64_t, 4> K_TENSOR_PARAM_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_TENSOR_PARAM_STRIDES = {64, 1, 1, 1};
constexpr std::array<int64_t, 1> K_TENSOR_SCALAR_DIMS = {1};
constexpr std::array<int64_t, 1> K_TENSOR_SCALAR_STRIDES = {1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBatchnormDescriptorLowering : public ::testing::Test
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

TEST_F(IntegrationBatchnormDescriptorLowering, BatchnormGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestBnFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_TENSOR_BIAS_UID).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(K_TENSOR_EPSILON_UID).set_name("Epsilon").set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_TENSOR_SCALAR_DIMS)).set_stride(toVec(K_TENSOR_SCALAR_STRIDES));

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("bn_fwd_op");
    bnAttrs.set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_uid(504).set_output(true).set_name("Y");
    meanOut->set_uid(505).set_output(true).set_name("Mean");
    invVarOut->set_uid(506).set_output(true).set_name("InvVariance");

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
    // x, scale, bias, epsilon, y, mean, invVariance = 7 tensors
    ASSERT_EQ(graphT.tensors.size(), 7u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_TENSOR_DATA_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_TENSOR_DATA_STRIDES));

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(scaleT->dims, toVec(K_TENSOR_PARAM_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_TENSOR_PARAM_STRIDES));

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_TENSOR_BIAS_UID];
    EXPECT_EQ(biasT->name, "Bias");

    // Verify output tensors exist
    ASSERT_NE(tensorMap.count(504), 0u);
    EXPECT_EQ(tensorMap[504]->name, "Y");
    ASSERT_NE(tensorMap.count(505), 0u);
    EXPECT_EQ(tensorMap[505]->name, "Mean");
    ASSERT_NE(tensorMap.count(506), 0u);
    EXPECT_EQ(tensorMap[506]->name, "InvVariance");

    // -- Verify batchnorm forward operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BatchnormAttributes);

    auto* bnFwd = node->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    EXPECT_EQ(bnFwd->x_tensor_uid, K_TENSOR_X_UID);
    EXPECT_EQ(bnFwd->scale_tensor_uid, K_TENSOR_SCALE_UID);
    EXPECT_EQ(bnFwd->bias_tensor_uid, K_TENSOR_BIAS_UID);
    EXPECT_EQ(bnFwd->epsilon_tensor_uid, K_TENSOR_EPSILON_UID);
    EXPECT_EQ(bnFwd->y_tensor_uid, 504);

    // Verify mean and inv_variance are set
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->mean_tensor_uid.value(), 505);
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->inv_variance_tensor_uid.value(), 506);

    // Running stats should not be set (not provided)
    EXPECT_FALSE(bnFwd->prev_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->prev_running_variance_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->momentum_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_variance_tensor_uid.has_value());

    // No peer stats
    EXPECT_EQ(bnFwd->peer_stats_tensor_uid.size(), 0u);
}

TEST_F(IntegrationBatchnormDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidBnFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_TENSOR_SCALAR_DIMS)).set_stride(toVec(K_TENSOR_SCALAR_STRIDES));

    BatchnormAttributes bnAttrs;
    bnAttrs.set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_output(true);
    meanOut->set_output(true);
    invVarOut->set_output(true);

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

    // x, scale, bias, epsilon, y, mean, invVariance = 7 tensors
    ASSERT_EQ(graphT.tensors.size(), 7u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 7u) << "Tensor UIDs are not unique";

    // The batchnorm forward operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnFwd = graphT.nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bnFwd->x_tensor_uid) > 0)
        << "X tensor UID " << bnFwd->x_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bnFwd->scale_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->bias_tensor_uid) > 0)
        << "Bias tensor UID " << bnFwd->bias_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->epsilon_tensor_uid) > 0)
        << "Epsilon tensor UID " << bnFwd->epsilon_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->y_tensor_uid) > 0)
        << "Y tensor UID " << bnFwd->y_tensor_uid << " not found in graph tensors";

    // Mean and inv_variance should be set with valid UIDs
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(bnFwd->mean_tensor_uid.value()) > 0)
        << "Mean tensor UID " << bnFwd->mean_tensor_uid.value() << " not found in graph tensors";
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(bnFwd->inv_variance_tensor_uid.value()) > 0)
        << "InvVariance tensor UID " << bnFwd->inv_variance_tensor_uid.value()
        << " not found in graph tensors";

    // All seven tensor UIDs referenced by the node should be distinct
    std::unordered_set<int64_t> nodeUids = {bnFwd->x_tensor_uid,
                                            bnFwd->scale_tensor_uid,
                                            bnFwd->bias_tensor_uid,
                                            bnFwd->epsilon_tensor_uid,
                                            bnFwd->y_tensor_uid,
                                            bnFwd->mean_tensor_uid.value(),
                                            bnFwd->inv_variance_tensor_uid.value()};
    EXPECT_EQ(nodeUids.size(), 7u) << "Batchnorm forward node tensor UIDs are not distinct";
}

TEST_F(IntegrationBatchnormDescriptorLowering, MinimalRequiredOnlyRoundTrip)
{
    // Tests the absolute minimum: x, scale, bias, epsilon as inputs -> y, mean, invVar as outputs
    // No running stats, no peer stats
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("MinimalBnFwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(600).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(601).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(602).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(603).set_name("Epsilon").set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_TENSOR_SCALAR_DIMS)).set_stride(toVec(K_TENSOR_SCALAR_STRIDES));

    BatchnormAttributes bnAttrs;
    bnAttrs.set_name("minimal_bn").set_epsilon(epsilon);

    auto [y, meanOut, invVarOut, nextRunMean, nextRunVar]
        = graph->batchnorm(x, scale, bias, bnAttrs);
    y->set_uid(604).set_output(true).set_name("Y");
    meanOut->set_uid(605).set_output(true).set_name("Mean");
    invVarOut->set_uid(606).set_output(true).set_name("InvVariance");

    auto result = graph->validate();
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    result = graph->build_operation_graph_via_descriptors(_handle);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

    // Retrieve and verify
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

    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnFwd = graphT.nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(bnFwd, nullptr);

    // Required tensors
    EXPECT_EQ(bnFwd->x_tensor_uid, 600);
    EXPECT_EQ(bnFwd->scale_tensor_uid, 601);
    EXPECT_EQ(bnFwd->bias_tensor_uid, 602);
    EXPECT_EQ(bnFwd->epsilon_tensor_uid, 603);
    EXPECT_EQ(bnFwd->y_tensor_uid, 604);

    // Optional outputs set by the graph's batchnorm() method
    ASSERT_TRUE(bnFwd->mean_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->mean_tensor_uid.value(), 605);
    ASSERT_TRUE(bnFwd->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(bnFwd->inv_variance_tensor_uid.value(), 606);

    // No running stats
    EXPECT_FALSE(bnFwd->prev_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->prev_running_variance_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->momentum_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_mean_tensor_uid.has_value());
    EXPECT_FALSE(bnFwd->next_running_variance_tensor_uid.has_value());

    // No peer stats
    EXPECT_EQ(bnFwd->peer_stats_tensor_uid.size(), 0u);
}

} // namespace
