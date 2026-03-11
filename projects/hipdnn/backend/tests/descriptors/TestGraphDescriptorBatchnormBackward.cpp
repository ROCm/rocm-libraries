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
#include <hipdnn_data_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
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
                                       std::vector<HipdnnBackendDescriptor*> peerStatsDescs = {})
{
    auto wrapper = createDescriptor<BatchnormBackwardOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormBackwardOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &dbiasDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarianceDesc);
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!peerStatsDescs.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           static_cast<int64_t>(peerStatsDescs.size()),
                           peerStatsDescs.data());
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
        desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_HANDLE, HIPDNN_TYPE_HANDLE, 1, &handle);
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
    auto dyDesc = createFinalizedTensor(60, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto xDesc = createFinalizedTensor(61, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto scaleDesc = createFinalizedTensor(62, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dxDesc = createFinalizedTensor(63, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto dscaleDesc = createFinalizedTensor(64, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dbiasDesc = createFinalizedTensor(65, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto meanDesc = createFinalizedTensor(7);
    auto invVarianceDesc = createFinalizedTensor(8);
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
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    // Verify the built graph
    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 8);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormBackwardAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, 60);
    EXPECT_EQ(attrs->x_tensor_uid, 61);
    EXPECT_EQ(attrs->scale_tensor_uid, 62);
    EXPECT_EQ(attrs->dx_tensor_uid, 63);
    EXPECT_EQ(attrs->dscale_tensor_uid, 64);
    EXPECT_EQ(attrs->dbias_tensor_uid, 65);
    EXPECT_EQ(attrs->mean_tensor_uid, 7);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, 8);

    // Verify peer_stats tensor array is empty (not set in basic test)
    EXPECT_EQ(attrs->peer_stats_tensor_uid.size(), 0u);
}

TEST_F(TestGraphDescriptorBatchnormBackward, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(60, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto xDesc = createFinalizedTensor(61, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto scaleDesc = createFinalizedTensor(62, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dxDesc = createFinalizedTensor(63, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto dscaleDesc = createFinalizedTensor(64, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dbiasDesc = createFinalizedTensor(65, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto meanDesc = createFinalizedTensor(7);
    auto invVarianceDesc = createFinalizedTensor(8);
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
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorBatchnormBackward, BuildWithPeerStatsTensorArray)
{
    auto dyDesc = createFinalizedTensor(60, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto xDesc = createFinalizedTensor(61, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto scaleDesc = createFinalizedTensor(62, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dxDesc = createFinalizedTensor(63, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto dscaleDesc = createFinalizedTensor(64, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto dbiasDesc = createFinalizedTensor(65, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto meanDesc = createFinalizedTensor(7);
    auto invVarianceDesc = createFinalizedTensor(8);
    auto peerStatsDesc0 = createFinalizedTensor(110);
    auto peerStatsDesc1 = createFinalizedTensor(111);

    std::vector<HipdnnBackendDescriptor*> peerStatsDescs
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
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify peer_stats tensor array UIDs
    ASSERT_EQ(attrs->peer_stats_tensor_uid.size(), 2u);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[0], 110);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[1], 111);
}

} // namespace
