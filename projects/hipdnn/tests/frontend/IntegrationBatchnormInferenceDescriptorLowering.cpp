// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include "test_plugins/TestPluginConstants.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;

namespace
{

// -- Test constants for BnInferenceGraphRoundTrip --

constexpr int64_t K_TENSOR_X_UID = 70;
constexpr int64_t K_TENSOR_MEAN_UID = 71;
constexpr int64_t K_TENSOR_INV_VARIANCE_UID = 72;
constexpr int64_t K_TENSOR_SCALE_UID = 73;
constexpr int64_t K_TENSOR_BIAS_UID = 74;
constexpr int64_t K_TENSOR_Y_UID = 75;

constexpr std::array<int64_t, 4> K_TENSOR_X_DIMS = {2, 64, 16, 16};
constexpr std::array<int64_t, 4> K_TENSOR_X_STRIDES = {16384, 256, 16, 1};
constexpr std::array<int64_t, 4> K_TENSOR_CHANNEL_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_TENSOR_CHANNEL_STRIDES = {64, 1, 1, 1};

// -- Test constants for AutoAssignedUidsPreservedInRoundTrip --

constexpr std::array<int64_t, 4> K_AUTO_X_DIMS = {1, 32, 8, 8};
constexpr std::array<int64_t, 4> K_AUTO_X_STRIDES = {2048, 64, 8, 1};
constexpr std::array<int64_t, 4> K_AUTO_CHANNEL_DIMS = {1, 32, 1, 1};
constexpr std::array<int64_t, 4> K_AUTO_CHANNEL_STRIDES = {32, 1, 1, 1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBatchnormInferenceDescriptorLowering : public IntegrationTestFixture
{
};

// Builds a batchnorm_inference graph via the frontend API, lowers it to the backend
// via build_operation_graph_via_descriptors, retrieves the serialized graph,
// and verifies all tensor and operation attributes match the values set
// in the frontend.
TEST_F(IntegrationBatchnormInferenceDescriptorLowering, BnInferenceGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBnInferenceGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_X_DIMS)).set_stride(toVec(K_TENSOR_X_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(K_TENSOR_MEAN_UID).set_name("Mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_TENSOR_CHANNEL_DIMS)).set_stride(toVec(K_TENSOR_CHANNEL_STRIDES));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_uid(K_TENSOR_INV_VARIANCE_UID)
        .set_name("InvVariance")
        .set_data_type(DataType::FLOAT);
    invVariance->set_dim(toVec(K_TENSOR_CHANNEL_DIMS)).set_stride(toVec(K_TENSOR_CHANNEL_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_CHANNEL_DIMS)).set_stride(toVec(K_TENSOR_CHANNEL_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_uid(K_TENSOR_BIAS_UID).set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_TENSOR_CHANNEL_DIMS)).set_stride(toVec(K_TENSOR_CHANNEL_STRIDES));

    BatchnormInferenceAttributes bnAttrs;
    bnAttrs.set_name("bn_inference_op");

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs);
    y->set_uid(K_TENSOR_Y_UID).set_output(true).set_name("Y");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    ASSERT_EQ(graphT.tensors.size(), 6u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(xT->dims, toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_TENSOR_X_STRIDES));
    EXPECT_FALSE(xT->virtual_);

    // Verify Mean tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_MEAN_UID), 0u);
    auto* meanT = tensorMap[K_TENSOR_MEAN_UID];
    EXPECT_EQ(meanT->name, "Mean");
    EXPECT_EQ(meanT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(meanT->dims, toVec(K_TENSOR_CHANNEL_DIMS));
    EXPECT_EQ(meanT->strides, toVec(K_TENSOR_CHANNEL_STRIDES));

    // Verify InvVariance tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_INV_VARIANCE_UID), 0u);
    auto* invVarT = tensorMap[K_TENSOR_INV_VARIANCE_UID];
    EXPECT_EQ(invVarT->name, "InvVariance");
    EXPECT_EQ(invVarT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(invVarT->dims, toVec(K_TENSOR_CHANNEL_DIMS));
    EXPECT_EQ(invVarT->strides, toVec(K_TENSOR_CHANNEL_STRIDES));

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(scaleT->dims, toVec(K_TENSOR_CHANNEL_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_TENSOR_CHANNEL_STRIDES));

    // Verify Bias tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_BIAS_UID), 0u);
    auto* biasT = tensorMap[K_TENSOR_BIAS_UID];
    EXPECT_EQ(biasT->name, "Bias");
    EXPECT_EQ(biasT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(biasT->dims, toVec(K_TENSOR_CHANNEL_DIMS));
    EXPECT_EQ(biasT->strides, toVec(K_TENSOR_CHANNEL_STRIDES));

    // Verify Y tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_Y_UID), 0u);
    auto* yT = tensorMap[K_TENSOR_Y_UID];
    EXPECT_EQ(yT->name, "Y");
    EXPECT_EQ(yT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(yT->virtual_);

    // -- Verify batchnorm inference operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BatchnormInferenceAttributes);

    auto* bnInf = node->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(bnInf, nullptr);

    EXPECT_EQ(bnInf->x_tensor_uid, K_TENSOR_X_UID);
    EXPECT_EQ(bnInf->mean_tensor_uid, K_TENSOR_MEAN_UID);
    EXPECT_EQ(bnInf->inv_variance_tensor_uid, K_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(bnInf->scale_tensor_uid, K_TENSOR_SCALE_UID);
    EXPECT_EQ(bnInf->bias_tensor_uid, K_TENSOR_BIAS_UID);
    EXPECT_EQ(bnInf->y_tensor_uid, K_TENSOR_Y_UID);
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationBatchnormInferenceDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidBnInfGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_AUTO_X_DIMS)).set_stride(toVec(K_AUTO_X_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_name("Mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_AUTO_CHANNEL_DIMS)).set_stride(toVec(K_AUTO_CHANNEL_STRIDES));

    auto invVariance = std::make_shared<TensorAttributes>();
    invVariance->set_name("InvVariance").set_data_type(DataType::FLOAT);
    invVariance->set_dim(toVec(K_AUTO_CHANNEL_DIMS)).set_stride(toVec(K_AUTO_CHANNEL_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_AUTO_CHANNEL_DIMS)).set_stride(toVec(K_AUTO_CHANNEL_STRIDES));

    auto bias = std::make_shared<TensorAttributes>();
    bias->set_name("Bias").set_data_type(DataType::FLOAT);
    bias->set_dim(toVec(K_AUTO_CHANNEL_DIMS)).set_stride(toVec(K_AUTO_CHANNEL_STRIDES));

    const BatchnormInferenceAttributes bnAttrs;

    auto y = graph->batchnorm_inference(x, mean, invVariance, scale, bias, bnAttrs);
    y->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 6u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 6u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The batchnorm inference operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnInf = graphT.nodes[0]->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(bnInf, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bnInf->x_tensor_uid) > 0)
        << "X tensor UID " << bnInf->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnInf->mean_tensor_uid) > 0)
        << "Mean tensor UID " << bnInf->mean_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnInf->inv_variance_tensor_uid) > 0)
        << "InvVariance tensor UID "
        << bnInf->inv_variance_tensor_uid // NOLINT(readability-implicit-bool-conversion)
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnInf->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bnInf->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnInf->bias_tensor_uid) > 0)
        << "Bias tensor UID " << bnInf->bias_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnInf->y_tensor_uid) > 0)
        << "Y tensor UID " << bnInf->y_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All six tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {bnInf->x_tensor_uid,
                                                  bnInf->mean_tensor_uid,
                                                  bnInf->inv_variance_tensor_uid,
                                                  bnInf->scale_tensor_uid,
                                                  bnInf->bias_tensor_uid,
                                                  bnInf->y_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 6u)
        << "Batchnorm inference node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
