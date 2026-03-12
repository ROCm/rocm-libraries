// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized BatchnormOperationDescriptor with all tensors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormOp(HipdnnBackendDescriptor* xDesc,
                               HipdnnBackendDescriptor* scaleDesc,
                               HipdnnBackendDescriptor* biasDesc,
                               HipdnnBackendDescriptor* epsilonDesc,
                               HipdnnBackendDescriptor* yDesc,
                               HipdnnBackendDescriptor* meanDesc = nullptr,
                               HipdnnBackendDescriptor* invVarianceDesc = nullptr,
                               HipdnnBackendDescriptor* prevRunningMeanDesc = nullptr,
                               HipdnnBackendDescriptor* prevRunningVarianceDesc = nullptr,
                               HipdnnBackendDescriptor* momentumDesc = nullptr,
                               HipdnnBackendDescriptor* nextRunningMeanDesc = nullptr,
                               HipdnnBackendDescriptor* nextRunningVarianceDesc = nullptr,
                               std::vector<HipdnnBackendDescriptor*> peerStatsDescs = {},
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<BatchnormOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &scaleDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &biasDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc);

    if(meanDesc != nullptr)
    {
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &meanDesc);
    }
    if(invVarianceDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &invVarianceDesc);
    }
    if(prevRunningMeanDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &prevRunningMeanDesc);
    }
    if(prevRunningVarianceDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &prevRunningVarianceDesc);
    }
    if(momentumDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &momentumDesc);
    }
    if(nextRunningMeanDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &nextRunningMeanDesc);
    }
    if(nextRunningVarianceDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &nextRunningVarianceDesc);
    }

    desc->setAttribute(HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    if(!peerStatsDescs.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           static_cast<int64_t>(peerStatsDescs.size()),
                           peerStatsDescs.data());
    }

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorBatchnorm : public ::testing::Test
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

TEST_F(TestGraphDescriptorBatchnorm, BuildFromSingleOperationWithOptionals)
{
    auto xDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_X_UID,
                                       toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_SCALE_UID,
                                           toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_BIAS_UID,
                                          toVec(K_BATCHNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_EPSILON_UID,
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_Y_UID,
                                       toVec(K_BATCHNORM_TENSOR_Y_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_Y_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_MEAN_UID,
                                          toVec(K_BATCHNORM_TENSOR_MEAN_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS),
                                                 toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES));

    auto opDesc = createFinalizedBatchnormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
                                             meanDesc.get(),
                                             invVarianceDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graph = GetGraph(serialized.ptr);
    auto graphT = graph->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1u);
    ASSERT_EQ(graphT->tensors.size(), 7u);

    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_EQ(attrs->x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);
    ASSERT_TRUE(attrs->mean_tensor_uid.has_value());
    EXPECT_EQ(attrs->mean_tensor_uid.value(), K_BATCHNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(attrs->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(attrs->inv_variance_tensor_uid.value(), K_BATCHNORM_TENSOR_INV_VARIANCE_UID);

    EXPECT_EQ(attrs->peer_stats_tensor_uid.size(), 0u);
}

TEST_F(TestGraphDescriptorBatchnorm, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_X_UID,
                                       toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_SCALE_UID,
                                           toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_BIAS_UID,
                                          toVec(K_BATCHNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_EPSILON_UID,
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_Y_UID,
                                       toVec(K_BATCHNORM_TENSOR_Y_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBatchnormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             {},
                                             HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data());
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorBatchnorm, BuildWithPeerStatsTensorArray)
{
    auto xDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_X_UID,
                                       toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_SCALE_UID,
                                           toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_BIAS_UID,
                                          toVec(K_BATCHNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_EPSILON_UID,
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_Y_UID,
                                       toVec(K_BATCHNORM_TENSOR_Y_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_Y_STRIDES));
    auto peerStatsDesc0 = createFinalizedTensor(K_BATCHNORM_TENSOR_PEER_STAT_0_UID,
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS),
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
    auto peerStatsDesc1 = createFinalizedTensor(K_BATCHNORM_TENSOR_PEER_STAT_1_UID,
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS),
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));

    std::vector<HipdnnBackendDescriptor*> peerStatsDescs
        = {peerStatsDesc0.get(), peerStatsDesc1.get()};
    auto opDesc = createFinalizedBatchnormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             nullptr,
                                             peerStatsDescs);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATIONGRAPH_OPS, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, ops.data()));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = GetGraph(serialized.ptr)->UnPack();

    ASSERT_EQ(graphT->nodes.size(), 1u);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormAttributes();
    ASSERT_NE(attrs, nullptr);

    ASSERT_EQ(attrs->peer_stats_tensor_uid.size(), 2u);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[0], K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(attrs->peer_stats_tensor_uid[1], K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
}

} // namespace
