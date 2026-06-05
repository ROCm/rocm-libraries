// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
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

// -- Test constants for BatchnormBackwardGraphRoundTrip --

constexpr int64_t K_TENSOR_DY_UID = 70;
constexpr int64_t K_TENSOR_X_UID = 71;
constexpr int64_t K_TENSOR_SCALE_UID = 72;
constexpr int64_t K_TENSOR_MEAN_UID = 73;
constexpr int64_t K_TENSOR_INV_VAR_UID = 74;

constexpr std::array<int64_t, 4> K_TENSOR_DATA_DIMS = {2, 64, 16, 16};
constexpr std::array<int64_t, 4> K_TENSOR_DATA_STRIDES = {16384, 256, 16, 1};
constexpr std::array<int64_t, 4> K_TENSOR_PARAM_DIMS = {1, 64, 1, 1};
constexpr std::array<int64_t, 4> K_TENSOR_PARAM_STRIDES = {64, 1, 1, 1};

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationBatchnormBackwardDescriptorLowering : public IntegrationTestFixture
{
};

TEST_F(IntegrationBatchnormBackwardDescriptorLowering, BatchnormBackwardGraphRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestBnBwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_TENSOR_SCALE_UID).set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto mean = std::make_shared<TensorAttributes>();
    mean->set_uid(K_TENSOR_MEAN_UID).set_name("Mean").set_data_type(DataType::FLOAT);
    mean->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    auto invVar = std::make_shared<TensorAttributes>();
    invVar->set_uid(K_TENSOR_INV_VAR_UID).set_name("InvVariance").set_data_type(DataType::FLOAT);
    invVar->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    BatchnormBackwardAttributes bnBwdAttrs;
    bnBwdAttrs.set_name("bn_bwd_op");
    bnBwdAttrs.set_saved_mean_and_inv_variance(mean, invVar);

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
    dx->set_uid(80).set_output(true).set_name("DX");
    dscale->set_uid(81).set_output(true).set_name("DScale");
    dbias->set_uid(82).set_output(true).set_name("DBias");

    // -- Validate, lower, serialize, and deserialize --
    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // -- Verify tensors --
    // dy, x, scale, mean, invVar, dx, dscale, dbias = 8 tensors
    ASSERT_EQ(graphT.tensors.size(), 8u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify DY tensor
    ASSERT_NE(tensorMap.count(K_TENSOR_DY_UID), 0u);
    auto* dyT = tensorMap[K_TENSOR_DY_UID];
    EXPECT_EQ(dyT->name, "DY");
    EXPECT_EQ(dyT->data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(dyT->dims, toVec(K_TENSOR_DATA_DIMS));
    EXPECT_EQ(dyT->strides, toVec(K_TENSOR_DATA_STRIDES));

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

    // Verify output tensors exist
    ASSERT_NE(tensorMap.count(80), 0u);
    EXPECT_EQ(tensorMap[80]->name, "DX");
    ASSERT_NE(tensorMap.count(81), 0u);
    EXPECT_EQ(tensorMap[81]->name, "DScale");
    ASSERT_NE(tensorMap.count(82), 0u);
    EXPECT_EQ(tensorMap[82]->name, "DBias");

    // -- Verify batchnorm backward operation node --
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::BatchnormBackwardAttributes);

    auto* bnBwd = node->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(bnBwd, nullptr);

    EXPECT_EQ(bnBwd->dy_tensor_uid, K_TENSOR_DY_UID);
    EXPECT_EQ(bnBwd->x_tensor_uid, K_TENSOR_X_UID);
    EXPECT_EQ(bnBwd->scale_tensor_uid, K_TENSOR_SCALE_UID);
    EXPECT_EQ(bnBwd->dx_tensor_uid, 80);
    EXPECT_EQ(bnBwd->dscale_tensor_uid, 81);
    EXPECT_EQ(bnBwd->dbias_tensor_uid, 82);

    // Verify mean and inv_variance are set
    ASSERT_TRUE(bnBwd->mean_tensor_uid.has_value());
    EXPECT_EQ(bnBwd->mean_tensor_uid.value(), K_TENSOR_MEAN_UID);
    ASSERT_TRUE(bnBwd->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(bnBwd->inv_variance_tensor_uid.value(), K_TENSOR_INV_VAR_UID);
}

TEST_F(IntegrationBatchnormBackwardDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidBnBwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_TENSOR_DATA_DIMS)).set_stride(toVec(K_TENSOR_DATA_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_TENSOR_PARAM_DIMS)).set_stride(toVec(K_TENSOR_PARAM_STRIDES));

    const BatchnormBackwardAttributes bnBwdAttrs;

    auto [dx, dscale, dbias] = graph->batchnorm_backward(dy, x, scale, bnBwdAttrs);
    dx->set_output(true);
    dscale->set_output(true);
    dbias->set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // dy, x, scale, dx, dscale, dbias = 6 tensors (no mean/invVar)
    ASSERT_EQ(graphT.tensors.size(), 6u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 6u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The batchnorm backward operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* bnBwd = graphT.nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(bnBwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(bnBwd->dy_tensor_uid) > 0)
        << "DY tensor UID " << bnBwd->dy_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnBwd->x_tensor_uid) > 0)
        << "X tensor UID " << bnBwd->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnBwd->scale_tensor_uid) > 0)
        << "Scale tensor UID " << bnBwd->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnBwd->dx_tensor_uid) > 0)
        << "DX tensor UID " << bnBwd->dx_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnBwd->dscale_tensor_uid) > 0)
        << "DScale tensor UID " << bnBwd->dscale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(bnBwd->dbias_tensor_uid) > 0)
        << "DBias tensor UID " << bnBwd->dbias_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All six tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {bnBwd->dy_tensor_uid,
                                                  bnBwd->x_tensor_uid,
                                                  bnBwd->scale_tensor_uid,
                                                  bnBwd->dx_tensor_uid,
                                                  bnBwd->dscale_tensor_uid,
                                                  bnBwd->dbias_tensor_uid};
    EXPECT_EQ(nodeUids.size(), 6u)
        << "BatchnormBackward node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}

} // namespace
