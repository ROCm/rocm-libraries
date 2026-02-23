// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceOperationDescriptor.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_test_sdk/constants/BnInferenceConstants.hpp>
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

// Helper: create a finalized BatchnormInferenceOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormInferenceOp(HipdnnBackendDescriptor* xDesc,
                                        HipdnnBackendDescriptor* meanDesc,
                                        HipdnnBackendDescriptor* invVarianceDesc,
                                        HipdnnBackendDescriptor* scaleDesc,
                                        HipdnnBackendDescriptor* biasDesc,
                                        HipdnnBackendDescriptor* yDesc,
                                        DataType computeType = DataType::FLOAT)
{
    auto wrapper = createDescriptor<BatchnormInferenceOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormInferenceOperationDescriptor>();

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &biasDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &yDesc);
    desc->setAttribute(HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorBatchnormInference : public ::testing::Test
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

TEST_F(TestGraphDescriptorBatchnormInference, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_TENSOR_X_UID, toVec(K_BN_INF_TENSOR_X_DIMS), toVec(K_BN_INF_TENSOR_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                          toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                                 toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                           toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                          toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_TENSOR_Y_UID, toVec(K_BN_INF_TENSOR_Y_DIMS), toVec(K_BN_INF_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestGraphDescriptorBatchnormInference, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(
        K_BN_INF_TENSOR_X_UID, toVec(K_BN_INF_TENSOR_X_DIMS), toVec(K_BN_INF_TENSOR_X_STRIDES));
    auto meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                          toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                                 toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                           toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          toVec(K_BN_INF_PER_CHANNEL_DIMS),
                                          toVec(K_BN_INF_PER_CHANNEL_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_BN_INF_TENSOR_Y_UID, toVec(K_BN_INF_TENSOR_Y_DIMS), toVec(K_BN_INF_TENSOR_Y_STRIDES));
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get(),
                                                      DataType::HALF);

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

} // namespace
