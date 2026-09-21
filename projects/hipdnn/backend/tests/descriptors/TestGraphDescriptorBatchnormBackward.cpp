// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormBackwardOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;
namespace
{

// Helper: create a finalized BatchnormBackwardOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormBackwardOp(HipdnnBackendDescriptor* dyDesc,
                                       HipdnnBackendDescriptor* xDesc,
                                       HipdnnBackendDescriptor* scaleDesc,
                                       HipdnnBackendDescriptor* dxDesc,
                                       HipdnnBackendDescriptor* dscaleDesc,
                                       HipdnnBackendDescriptor* dbiasDesc,
                                       HipdnnBackendDescriptor* meanDesc,
                                       HipdnnBackendDescriptor* invVarianceDesc,
                                       hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                       std::vector<HipdnnBackendDescriptor*> peerStatsDescs = {},
                                       const std::string& name = "")
{
    auto wrapper = createDescriptor<BatchnormBackwardOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormBackwardOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dscaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dbiasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&meanDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&invVarianceDesc));
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!peerStatsDescs.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           static_cast<int64_t>(peerStatsDescs.size()),
                           static_cast<const void*>(peerStatsDescs.data()));
    }

    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.c_str());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorBatchnormBackward : public ::testing::Test
{
public:
    std::shared_ptr<GraphDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<GraphDescriptor>();
    }

    void setHandle() const
    {
        auto desc = getDescriptor();
        hipdnnHandle_t handle = &_mockHandle;
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                           HIPDNN_TYPE_HANDLE,
                           1,
                           static_cast<const void*>(&handle));
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    mutable MockHandle _mockHandle;

    void SetUp() override
    {
        _wrapper = createDescriptor<GraphDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

TEST_F(TestGraphDescriptorBatchnormBackward, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DY_UID, toVec(K_BN_BWD_TENSOR_DY_DIMS), toVec(K_BN_BWD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_X_UID, toVec(K_BN_BWD_TENSOR_X_DIMS), toVec(K_BN_BWD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_SCALE_UID,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DX_UID, toVec(K_BN_BWD_TENSOR_DX_DIMS), toVec(K_BN_BWD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                                            toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                            toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DBIAS_UID,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_BWD_TENSOR_MEAN_UID);
    auto invVarianceDesc = createFinalizedTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto opDesc = createFinalizedBatchnormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 8);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormBackwardAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(attrs->dscale_tensor_uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(attrs->dbias_tensor_uid, K_BN_BWD_TENSOR_DBIAS_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_BN_BWD_TENSOR_MEAN_UID);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, K_BN_BWD_TENSOR_INV_VARIANCE_UID);

    // Verify peer_stats tensor array is empty (not set in basic test)
    EXPECT_EQ(attrs->peer_stats_tensor_uid.size(), 0u);
}

TEST_F(TestGraphDescriptorBatchnormBackward, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DY_UID, toVec(K_BN_BWD_TENSOR_DY_DIMS), toVec(K_BN_BWD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_X_UID, toVec(K_BN_BWD_TENSOR_X_DIMS), toVec(K_BN_BWD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_SCALE_UID,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DX_UID, toVec(K_BN_BWD_TENSOR_DX_DIMS), toVec(K_BN_BWD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                                            toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                            toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DBIAS_UID,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_BWD_TENSOR_MEAN_UID);
    auto invVarianceDesc = createFinalizedTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto opDesc = createFinalizedBatchnormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorBatchnormBackward, BuildWithPeerStatsTensorArray)
{
    auto dyDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DY_UID, toVec(K_BN_BWD_TENSOR_DY_DIMS), toVec(K_BN_BWD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_X_UID, toVec(K_BN_BWD_TENSOR_X_DIMS), toVec(K_BN_BWD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_SCALE_UID,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DX_UID, toVec(K_BN_BWD_TENSOR_DX_DIMS), toVec(K_BN_BWD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                                            toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                            toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DBIAS_UID,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_BWD_TENSOR_MEAN_UID);
    auto invVarianceDesc = createFinalizedTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto peerStatsDesc0 = createFinalizedTensor(K_BN_BWD_TENSOR_PEER_STAT_0_UID);
    auto peerStatsDesc1 = createFinalizedTensor(K_BN_BWD_TENSOR_PEER_STAT_1_UID);

    const std::vector<HipdnnBackendDescriptor*> peerStatsDescs
        = {peerStatsDesc0.get(), peerStatsDesc1.get()};
    auto opDesc = createFinalizedBatchnormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     HIPDNN_DATA_FLOAT,
                                                     peerStatsDescs);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify peer_stats tensor array UIDs
    ASSERT_EQ(attrs->peer_stats_tensor_uid.size(), 2u);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[0], K_BN_BWD_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[1], K_BN_BWD_TENSOR_PEER_STAT_1_UID);
}

TEST_F(TestGraphDescriptorBatchnormBackward, OperationNamePreservedInSerialization)
{
    auto dyDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DY_UID, toVec(K_BN_BWD_TENSOR_DY_DIMS), toVec(K_BN_BWD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_X_UID, toVec(K_BN_BWD_TENSOR_X_DIMS), toVec(K_BN_BWD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_SCALE_UID,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DX_UID, toVec(K_BN_BWD_TENSOR_DX_DIMS), toVec(K_BN_BWD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                                            toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                            toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DBIAS_UID,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_BWD_TENSOR_MEAN_UID);
    auto invVarianceDesc = createFinalizedTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto opDesc = createFinalizedBatchnormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     HIPDNN_DATA_FLOAT,
                                                     {},
                                                     "bn_bwd_train");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "bn_bwd_train");
}

TEST_F(TestGraphDescriptorBatchnormBackward, OperationNameRoundTripThroughLifting)
{
    auto dyDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DY_UID, toVec(K_BN_BWD_TENSOR_DY_DIMS), toVec(K_BN_BWD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_X_UID, toVec(K_BN_BWD_TENSOR_X_DIMS), toVec(K_BN_BWD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_SCALE_UID,
                                           toVec(K_BN_BWD_TENSOR_SCALE_DIMS),
                                           toVec(K_BN_BWD_TENSOR_SCALE_STRIDES));
    auto dxDesc = createFinalizedTensor(
        K_BN_BWD_TENSOR_DX_UID, toVec(K_BN_BWD_TENSOR_DX_DIMS), toVec(K_BN_BWD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DSCALE_UID,
                                            toVec(K_BN_BWD_TENSOR_DSCALE_DIMS),
                                            toVec(K_BN_BWD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_BN_BWD_TENSOR_DBIAS_UID,
                                           toVec(K_BN_BWD_TENSOR_DBIAS_DIMS),
                                           toVec(K_BN_BWD_TENSOR_DBIAS_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_BWD_TENSOR_MEAN_UID);
    auto invVarianceDesc = createFinalizedTensor(K_BN_BWD_TENSOR_INV_VARIANCE_UID);
    auto opDesc = createFinalizedBatchnormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     HIPDNN_DATA_FLOAT,
                                                     {},
                                                     "bn_bwd_train");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    // Serialize the graph
    auto serialized = desc->getSerializedGraph();
    std::vector<uint8_t> bytes(static_cast<const uint8_t*>(serialized.ptr),
                               static_cast<const uint8_t*>(serialized.ptr) + serialized.size);

    // Deserialize into a new GraphDescriptor (lifting path)
    auto liftedWrapper = createDescriptor<GraphDescriptor>();
    auto liftedDesc = liftedWrapper->asDescriptor<GraphDescriptor>();
    liftedDesc->deserializeGraph(bytes.data(), bytes.size());

    hipdnnHandle_t handle = &_mockHandle;
    liftedDesc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE,
                             HIPDNN_TYPE_HANDLE,
                             1,
                             static_cast<const void*>(&handle));
    liftedDesc->finalize();

    // Re-serialize and verify name survived the round-trip
    auto reSerialized = liftedDesc->getSerializedGraph();
    auto graphT = UnPackGraph(reSerialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "bn_bwd_train");

    // Verify all tensor UIDs survived
    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->dy_tensor_uid, K_BN_BWD_TENSOR_DY_UID);
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_BWD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_BWD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_BN_BWD_TENSOR_DX_UID);
    EXPECT_EQ(attrs->dscale_tensor_uid, K_BN_BWD_TENSOR_DSCALE_UID);
    EXPECT_EQ(attrs->dbias_tensor_uid, K_BN_BWD_TENSOR_DBIAS_UID);

    // Verify optional tensor UIDs survived
    ASSERT_TRUE(attrs->mean_tensor_uid.has_value());
    EXPECT_EQ(attrs->mean_tensor_uid.value(), K_BN_BWD_TENSOR_MEAN_UID);
    ASSERT_TRUE(attrs->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(attrs->inv_variance_tensor_uid.value(), K_BN_BWD_TENSOR_INV_VARIANCE_UID);
}

} // namespace
