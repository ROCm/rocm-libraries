// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
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

// Helper: create a finalized BatchnormInferenceOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedBatchnormInferenceOp(HipdnnBackendDescriptor* xDesc,
                                        HipdnnBackendDescriptor* meanDesc,
                                        HipdnnBackendDescriptor* invVarianceDesc,
                                        HipdnnBackendDescriptor* scaleDesc,
                                        HipdnnBackendDescriptor* biasDesc,
                                        HipdnnBackendDescriptor* yDesc,
                                        hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<BatchnormInferenceOperationDescriptor>();
    auto desc = wrapper->asDescriptor<BatchnormInferenceOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&meanDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&invVarianceDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&biasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

TEST_F(TestGraphDescriptorBatchnormInference, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(70, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto meanDesc = createFinalizedTensor(71, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto invVarianceDesc = createFinalizedTensor(72, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto scaleDesc = createFinalizedTensor(73, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto biasDesc = createFinalizedTensor(74, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto yDesc = createFinalizedTensor(75, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get());

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
    ASSERT_EQ(graphT->tensors.size(), 6);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, 70);
    EXPECT_EQ(attrs->mean_tensor_uid, 71);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, 72);
    EXPECT_EQ(attrs->scale_tensor_uid, 73);
    EXPECT_EQ(attrs->bias_tensor_uid, 74);
    EXPECT_EQ(attrs->y_tensor_uid, 75);
}

TEST_F(TestGraphDescriptorBatchnormInference, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(70, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto meanDesc = createFinalizedTensor(71, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto invVarianceDesc = createFinalizedTensor(72, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto scaleDesc = createFinalizedTensor(73, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto biasDesc = createFinalizedTensor(74, {1, 64, 1, 1}, {64, 1, 1, 1});
    auto yDesc = createFinalizedTensor(75, {1, 64, 32, 32}, {65536, 1024, 32, 1});
    auto opDesc = createFinalizedBatchnormInferenceOp(xDesc.get(),
                                                      meanDesc.get(),
                                                      invVarianceDesc.get(),
                                                      scaleDesc.get(),
                                                      biasDesc.get(),
                                                      yDesc.get(),
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

} // namespace
