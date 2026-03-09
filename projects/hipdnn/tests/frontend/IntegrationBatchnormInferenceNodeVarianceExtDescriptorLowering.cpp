// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_variance_ext_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/BnInfVarExtConstants.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
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

// -- Test constants for AutoAssignedUidsPreservedInRoundTrip --
constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {2, 32, 8, 8};
constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {2048, 64, 8, 1};

constexpr std::array<int64_t, 4> K_AUTO_PARAM_DIMS = {1, 32, 1, 1};
constexpr std::array<int64_t, 4> K_AUTO_PARAM_STRIDES = {32, 1, 1, 1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBnInfVarExtDescriptorLowering : public ::testing::Test
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

// Builds a batchnorm_inference_variance_ext graph via the frontend API, lowers it
// to the backend via build_operation_graph_via_descriptors, retrieves the serialized
// graph, and verifies all tensor and operation attributes match.
TEST_F(IntegrationBnInfVarExtDescriptorLowering, BnInfVarExtGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("TestBnInfVarExtGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_BN_INF_VAR_EXT_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_BN_INF_VAR_EXT_X_DIMS)).set_stride(toVec(K_BN_INF_VAR_EXT_X_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(K_BN_INF_VAR_EXT_MEAN_UID).set_name("Mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_BN_INF_VAR_EXT_MEAN_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));

    auto variance = std::make_shared<TensorAttributes>();
    variance->set_uid(K_BN_INF_VAR_EXT_VARIANCE_UID)
        .set_name("Variance")
        .set_data_type(DataType::FLOAT);
    variance->set_dim(toVec(K_BN_INF_VAR_EXT_VARIANCE_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_VARIANCE_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_BN_INF_VAR_EXT_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_BN_INF_VAR_EXT_SCALE_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_SCALE_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_BN_INF_VAR_EXT_BIAS_UID).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_BN_INF_VAR_EXT_BIAS_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_BIAS_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_uid(K_BN_INF_VAR_EXT_EPSILON_UID)
        .set_name("Epsilon")
        .set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    epsilon->set_value(1e-5f);

    BatchnormInferenceAttributesVarianceExt bnAttrs;
    bnAttrs.set_name("bn_inf_var_ext_op");

    auto y = graph->batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, std::move(bnAttrs));
    y->set_uid(K_BN_INF_VAR_EXT_Y_UID).set_output(true).set_name("Y");

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
    ASSERT_EQ(graphT.tensors.size(), 7u);

    std::unordered_map<int64_t, const hipdnn_data_sdk::data_objects::TensorAttributesT*> tensorMap;
    for(const auto& t : graphT.tensors)
    {
        tensorMap[t->uid] = t.get();
    }

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_X_UID), 0u);
    auto* xT = tensorMap[K_BN_INF_VAR_EXT_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_BN_INF_VAR_EXT_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_BN_INF_VAR_EXT_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify Mean tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_MEAN_UID), 0u);
    auto* meanT = tensorMap[K_BN_INF_VAR_EXT_MEAN_UID];
    EXPECT_EQ(meanT->name, "Mean");
    EXPECT_EQ(meanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(meanT->dims, toVec(K_BN_INF_VAR_EXT_MEAN_DIMS));
    EXPECT_EQ(meanT->strides, toVec(K_BN_INF_VAR_EXT_MEAN_STRIDES));

    // Verify Variance tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_VARIANCE_UID), 0u);
    auto* varianceT = tensorMap[K_BN_INF_VAR_EXT_VARIANCE_UID];
    EXPECT_EQ(varianceT->name, "Variance");
    EXPECT_EQ(varianceT->data_type, DataTypeSdk::FLOAT);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_BN_INF_VAR_EXT_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_BN_INF_VAR_EXT_BIAS_UID];
    EXPECT_EQ(biasT->name, "Bias");
    EXPECT_EQ(biasT->data_type, DataTypeSdk::FLOAT);

    // Verify Epsilon tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_EPSILON_UID), 0u);
    auto* epsilonT = tensorMap[K_BN_INF_VAR_EXT_EPSILON_UID];
    EXPECT_EQ(epsilonT->name, "Epsilon");
    EXPECT_EQ(epsilonT->data_type, DataTypeSdk::FLOAT);

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_BN_INF_VAR_EXT_Y_UID), 0u);
    auto* yT = tensorMap[K_BN_INF_VAR_EXT_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(yT->virtual_);

    // -- Verify batchnorm inference variance ext operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BatchnormInferenceAttributesVarianceExt);

    auto* bnFwd = node->attributes.AsBatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(bnFwd, nullptr);

    EXPECT_EQ(bnFwd->x_tensor_uid, K_BN_INF_VAR_EXT_X_UID);
    EXPECT_EQ(bnFwd->mean_tensor_uid, K_BN_INF_VAR_EXT_MEAN_UID);
    EXPECT_EQ(bnFwd->variance_tensor_uid, K_BN_INF_VAR_EXT_VARIANCE_UID);
    EXPECT_EQ(bnFwd->scale_tensor_uid, K_BN_INF_VAR_EXT_SCALE_UID);
    EXPECT_EQ(bnFwd->bias_tensor_uid, K_BN_INF_VAR_EXT_BIAS_UID);
    EXPECT_EQ(bnFwd->y_tensor_uid, K_BN_INF_VAR_EXT_Y_UID);
    EXPECT_EQ(bnFwd->epsilon_tensor_uid, K_BN_INF_VAR_EXT_EPSILON_UID);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationBnInfVarExtDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraph>();
    graph->set_name("AutoUidBnGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_name("Mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_AUTO_PARAM_DIMS)).set_stride(toVec(K_AUTO_PARAM_STRIDES));

    auto variance = std::make_shared<TensorAttributes>();
    variance->set_name("Variance").set_data_type(DataType::FLOAT);
    variance->set_dim(toVec(K_AUTO_PARAM_DIMS)).set_stride(toVec(K_AUTO_PARAM_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_AUTO_PARAM_DIMS)).set_stride(toVec(K_AUTO_PARAM_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_AUTO_PARAM_DIMS)).set_stride(toVec(K_AUTO_PARAM_STRIDES));

    auto epsilon = std::make_shared<TensorAttributes>();
    epsilon->set_name("Epsilon").set_data_type(DataType::FLOAT);
    epsilon->set_dim(toVec(K_BN_INF_VAR_EXT_EPSILON_DIMS))
        .set_stride(toVec(K_BN_INF_VAR_EXT_EPSILON_STRIDES));
    epsilon->set_value(1e-5f);

    BatchnormInferenceAttributesVarianceExt bnAttrs;

    auto y = graph->batchnorm_inference_variance_ext(
        x, mean, variance, scale, bias, epsilon, std::move(bnAttrs));
    y->set_output(true);

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

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 7u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 7u) << "Tensor UIDs are not unique";

    // The batchnorm operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnFwd = graphT.nodes[0]->attributes.AsBatchnormInferenceAttributesVarianceExt();
    ASSERT_NE(bnFwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bnFwd->x_tensor_uid) > 0)
        << "X tensor UID " << bnFwd->x_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->mean_tensor_uid) > 0)
        << "Mean tensor UID " << bnFwd->mean_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->variance_tensor_uid) > 0)
        << "Variance tensor UID " << bnFwd->variance_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bnFwd->scale_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->bias_tensor_uid) > 0)
        << "Bias tensor UID " << bnFwd->bias_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->y_tensor_uid) > 0)
        << "Y tensor UID " << bnFwd->y_tensor_uid << " not found in graph tensors";
    EXPECT_TRUE(uids.count(bnFwd->epsilon_tensor_uid) > 0)
        << "Epsilon tensor UID " << bnFwd->epsilon_tensor_uid << " not found in graph tensors";

    // All seven tensor UIDs referenced by the node should be distinct
    std::unordered_set<int64_t> nodeUids = {bnFwd->x_tensor_uid,
                                            bnFwd->mean_tensor_uid,
                                            bnFwd->variance_tensor_uid,
                                            bnFwd->scale_tensor_uid,
                                            bnFwd->bias_tensor_uid,
                                            bnFwd->y_tensor_uid,
                                            bnFwd->epsilon_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 7u) << "BN node tensor UIDs are not distinct";
}

} // namespace
