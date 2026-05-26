// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_frontend.hpp>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/IntegrationTestFixture.hpp>
#include <hipdnn_test_sdk/utilities/LoweringTestHelpers.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/TestableGraph.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace hipdnn_tests::constants;
using hipdnn_tests::buildTensorMap;
using hipdnn_tests::IntegrationTestFixture;
using hipdnn_tests::lowerAndDeserialize;
using hipdnn_tests::TestableGraphLowering;
using hipdnn_tests::toVec;
using DataTypeSdk = hipdnn_flatbuffers_sdk::data_objects::DataType;
using NodeAttrType = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes;

namespace
{

// Lowers a frontend graph via build_operation_graph_via_descriptors, then
// retrieves the serialized graph and deserializes it for verification.
class IntegrationRMSNormBackwardDescriptorLowering : public IntegrationTestFixture
{
};

// Lowering round-trip: builds a graph, lowers via descriptors, and verifies
// the deserialized FlatBuffer attributes match.
TEST_F(IntegrationRMSNormBackwardDescriptorLowering, RMSNormBackwardRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestRMSNormBackwardGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_RMSNORMBACKWARD_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RMSNORMBACKWARD_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_uid(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID)
        .set_name("inv_rms")
        .set_data_type(DataType::FLOAT);
    invRms->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));

    RMSNormBackwardAttributes attrs;
    attrs.set_name("rmsnorm_bwd_op");

    auto outputs = graph->rmsnorm_backward(dy, x, scale, invRms, std::move(attrs));
    const auto& dx = outputs[0];
    const auto& dScale = outputs[1];
    const auto& dBias = outputs[2];

    dx->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("DX");
    dScale->set_uid(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
    EXPECT_EQ(dBias, nullptr);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // Verify tensors: dy, x, scale, inv_rms, dx, dscale = 6 tensors
    ASSERT_EQ(graphT.tensors.size(), 6u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify DY tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    auto* dyT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID];
    EXPECT_EQ(dyT->name, "DY");
    EXPECT_EQ(dyT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(dyT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(dyT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dyT->virtual_);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(xT->virtual_);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(scaleT->virtual_);

    // Verify inv rms tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID), 0u);
    auto* invRMST = tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID];
    EXPECT_EQ(invRMST->name, "inv_rms");
    EXPECT_EQ(invRMST->dims, toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(invRMST->strides, toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(invRMST->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(invRMST->virtual_);

    // Verify DX tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    auto* dxT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID];
    EXPECT_EQ(dxT->name, "DX");
    EXPECT_EQ(dxT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(dxT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(dxT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dxT->virtual_);

    // Verify Dscale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    auto* dscaleT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID];
    EXPECT_EQ(dscaleT->name, "DScale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(dscaleT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(dscaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dscaleT->virtual_);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::RMSNormBackwardAttributes);

    auto* opNode = node->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(opNode->x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(opNode->scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(opNode->inv_rms_tensor_uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(opNode->dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "rmsnorm_bwd_op");
}

// Roundtrip with optional dbias tensor set, verifying it appears in the serialized graph.
TEST_F(IntegrationRMSNormBackwardDescriptorLowering, RMSNormBackwardWithDBiasRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("TestRMSNormBackwardGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_uid(K_RMSNORMBACKWARD_TENSOR_DY_UID).set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_uid(K_RMSNORMBACKWARD_TENSOR_X_UID).set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_uid(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)
        .set_name("Scale")
        .set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_uid(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID)
        .set_name("inv_rms")
        .set_data_type(DataType::FLOAT);
    invRms->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));

    RMSNormBackwardAttributes attrs;
    attrs.set_name("rmsnorm_bwd_op");
    attrs.set_compute_dbias(true);

    auto outputs = graph->rmsnorm_backward(dy, x, scale, invRms, std::move(attrs));
    const auto& dx = outputs[0];
    const auto& dScale = outputs[1];
    const auto& dBias = outputs[2];

    dx->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true).set_name("DX");
    dScale->set_uid(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID).set_output(true).set_name("DScale");
    dBias->set_uid(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID).set_output(true).set_name("DBias");

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // -- Verify graph-level attributes --
    EXPECT_EQ(graphT.compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.intermediate_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(graphT.io_data_type, DataTypeSdk::FLOAT);

    // Verify tensors: dy, x, scale, inv_rms, dx, dscale, dbias = 7 tensors
    ASSERT_EQ(graphT.tensors.size(), 7u);

    auto tensorMap = buildTensorMap(graphT);

    // Verify DY tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DY_UID), 0u);
    auto* dyT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DY_UID];
    EXPECT_EQ(dyT->name, "DY");
    EXPECT_EQ(dyT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS));
    EXPECT_EQ(dyT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    EXPECT_EQ(dyT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dyT->virtual_);

    // Verify X tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_X_UID), 0u);
    auto* xT = tensorMap[K_RMSNORMBACKWARD_TENSOR_X_UID];
    EXPECT_EQ(xT->name, "X");
    EXPECT_EQ(xT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS));
    EXPECT_EQ(xT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    EXPECT_EQ(xT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(xT->virtual_);

    // Verify Scale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_SCALE_UID), 0u);
    auto* scaleT = tensorMap[K_RMSNORMBACKWARD_TENSOR_SCALE_UID];
    EXPECT_EQ(scaleT->name, "Scale");
    EXPECT_EQ(scaleT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS));
    EXPECT_EQ(scaleT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    EXPECT_EQ(scaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(scaleT->virtual_);

    // Verify inv rms tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID), 0u);
    auto* invRMST = tensorMap[K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID];
    EXPECT_EQ(invRMST->name, "inv_rms");
    EXPECT_EQ(invRMST->dims, toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS));
    EXPECT_EQ(invRMST->strides, toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    EXPECT_EQ(invRMST->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(invRMST->virtual_);

    // Verify DX tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DX_UID), 0u);
    auto* dxT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DX_UID];
    EXPECT_EQ(dxT->name, "DX");
    EXPECT_EQ(dxT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS));
    EXPECT_EQ(dxT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    EXPECT_EQ(dxT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dxT->virtual_);

    // Verify Dscale tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID), 0u);
    auto* dscaleT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID];
    EXPECT_EQ(dscaleT->name, "DScale");
    EXPECT_EQ(tensorMap[K_RMSNORMBACKWARD_TENSOR_DSCALE_UID]->dims,
              toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS));
    EXPECT_EQ(dscaleT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    EXPECT_EQ(dscaleT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dscaleT->virtual_);

    // Verify Dbias tensor
    ASSERT_NE(tensorMap.count(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID), 0u);
    auto* dbiasT = tensorMap[K_RMSNORMBACKWARD_TENSOR_DBIAS_UID];
    EXPECT_EQ(dbiasT->name, "DBias");
    EXPECT_EQ(dbiasT->dims, toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS));
    EXPECT_EQ(dbiasT->strides, toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    EXPECT_EQ(dbiasT->data_type, DataTypeSdk::FLOAT);
    EXPECT_FALSE(dbiasT->virtual_);

    // Verify operation node
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto& node = graphT.nodes[0];
    EXPECT_EQ(node->compute_data_type, DataTypeSdk::FLOAT);
    EXPECT_EQ(node->attributes.type, NodeAttrType::RMSNormBackwardAttributes);

    auto* opNode = node->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(opNode, nullptr);

    // Verify required tensor UIDs
    EXPECT_EQ(opNode->dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(opNode->x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(opNode->scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(opNode->inv_rms_tensor_uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(opNode->dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(opNode->dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(opNode->dbias_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify operation name preserved through lowering
    EXPECT_EQ(node->name, "rmsnorm_bwd_op");
}

// Verifies that tensor UIDs auto-assigned by the frontend are preserved
// through the lowering round-trip.
TEST_F(IntegrationRMSNormBackwardDescriptorLowering, AutoAssignedUidsPreservedInRoundTrip)
{
    auto graph = std::make_shared<TestableGraphLowering>();
    graph->set_name("AutoUidRMSNormBwdGraph")
        .set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto dy = std::make_shared<TensorAttributes>();
    dy->set_name("DY").set_data_type(DataType::FLOAT);
    dy->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto x = std::make_shared<TensorAttributes>();
    x->set_name("X").set_data_type(DataType::FLOAT);
    x->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));

    auto scale = std::make_shared<TensorAttributes>();
    scale->set_name("Scale").set_data_type(DataType::FLOAT);
    scale->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));

    auto invRms = std::make_shared<TensorAttributes>();
    invRms->set_uid(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID)
        .set_name("inv_rms")
        .set_data_type(DataType::FLOAT);
    invRms->set_dim(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS))
        .set_stride(toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));

    RMSNormBackwardAttributes attrs;
    attrs.set_name("rmsnorm_bwd_op");
    attrs.set_compute_dbias(true);

    auto outputs = graph->rmsnorm_backward(dy, x, scale, invRms, std::move(attrs));
    const auto& dx = outputs[0];
    const auto& dScale = outputs[1];
    const auto& dBias = outputs[2];

    dx->set_uid(K_RMSNORMBACKWARD_TENSOR_DX_UID).set_output(true);
    dScale->set_uid(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID).set_output(true);
    dBias->set_uid(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID).set_output(true);

    auto graphT = lowerAndDeserialize(*graph, _handle);

    // All tensors should have been auto-assigned unique UIDs
    ASSERT_EQ(graphT.tensors.size(), 7u);
    std::unordered_set<int64_t> uids;
    for(const auto& t : graphT.tensors)
    {
        uids.insert(t->uid);
    }
    EXPECT_EQ(uids.size(), 7u)
        << "Tensor UIDs are not unique"; // NOLINT(readability-implicit-bool-conversion)

    // The batchnorm inference operation should reference the auto-assigned UIDs
    ASSERT_EQ(graphT.nodes.size(), 1u);
    auto* rmsBwd = graphT.nodes[0]->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(rmsBwd, nullptr);

    // Tensor UIDs in the node should match tensors in the graph
    EXPECT_TRUE(uids.count(rmsBwd->dy_tensor_uid) > 0)
        << "DY tensor UID " << rmsBwd->dy_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(rmsBwd->x_tensor_uid) > 0)
        << "X tensor UID " << rmsBwd->x_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(rmsBwd->scale_tensor_uid) > 0)
        << "Scale tensor UID " << rmsBwd->scale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(rmsBwd->inv_rms_tensor_uid) > 0)
        << "INV RMS tensor UID " << rmsBwd->inv_rms_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(rmsBwd->dx_tensor_uid) > 0)
        << "DX tensor UID " << rmsBwd->dx_tensor_uid // NOLINT(readability-implicit-bool-conversion)
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(uids.count(rmsBwd->dscale_tensor_uid) > 0)
        << "DScale tensor UID " << rmsBwd->dscale_tensor_uid
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)
    EXPECT_TRUE(rmsBwd->dbias_tensor_uid.has_value());
    EXPECT_TRUE(uids.count(rmsBwd->dbias_tensor_uid.value()) > 0)
        << "DBias tensor UID " << rmsBwd->dbias_tensor_uid.value()
        << " not found in graph tensors"; // NOLINT(readability-implicit-bool-conversion)

    // All six tensor UIDs referenced by the node should be distinct
    const std::unordered_set<int64_t> nodeUids = {rmsBwd->dy_tensor_uid,
                                                  rmsBwd->x_tensor_uid,
                                                  rmsBwd->scale_tensor_uid,
                                                  rmsBwd->inv_rms_tensor_uid,
                                                  rmsBwd->dx_tensor_uid,
                                                  rmsBwd->dscale_tensor_uid,
                                                  rmsBwd->dbias_tensor_uid.value()};
    EXPECT_EQ(nodeUids.size(), 7u)
        << "RMSNorm backward node tensor UIDs are not distinct"; // NOLINT(readability-implicit-bool-conversion)
}
} // namespace
