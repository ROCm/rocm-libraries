// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/LayernormOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <set>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized LayernormOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedLayernormOp(HipdnnBackendDescriptor* xDesc,
                               HipdnnBackendDescriptor* scaleDesc,
                               HipdnnBackendDescriptor* biasDesc,
                               HipdnnBackendDescriptor* epsilonDesc,
                               HipdnnBackendDescriptor* yDesc,
                               HipdnnBackendDescriptor* meanDesc,
                               HipdnnBackendDescriptor* invVarianceDesc,
                               hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT)
{
    auto wrapper = createDescriptor<LayernormOperationDescriptor>();
    auto desc = wrapper->asDescriptor<LayernormOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&biasDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&epsilonDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    if(meanDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&meanDesc));
    }
    if(invVarianceDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&invVarianceDesc));
    }
    desc->setAttribute(HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    auto forwardPhase = HIPDNN_NORM_FWD_TRAINING;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                       HIPDNN_TYPE_NORM_FWD_PHASE,
                       1,
                       &forwardPhase);
    int64_t normalizedDimCount = 3;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT,
                       HIPDNN_TYPE_INT64,
                       1,
                       &normalizedDimCount);

    desc->finalize();
    return wrapper;
}

class TestGraphDescriptorLayernorm : public ::testing::Test
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

TEST_F(TestGraphDescriptorLayernorm, BuildFromSingleOperation)
{
    auto xDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_X_UID,
                                       toVec(K_LAYERNORM_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_BIAS_UID,
                                          toVec(K_LAYERNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_Y_UID,
                                       toVec(K_LAYERNORM_TENSOR_Y_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_Y_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORM_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS),
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES));
    auto opDesc = createFinalizedLayernormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
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
    ASSERT_EQ(graphT->tensors.size(), 7);

    // Verify the node has correct attributes type
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::LayernormAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    // Verify compute_data_type and forward_phase on the node
    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::FLOAT);

    // Verify tensor UID references
    EXPECT_EQ(attrs->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_TRUE(attrs->mean_tensor_uid.has_value());
    EXPECT_EQ(attrs->mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    EXPECT_TRUE(attrs->inv_variance_tensor_uid.has_value());
    EXPECT_EQ(attrs->inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(attrs->forward_phase, NormFwdPhase::TRAINING);
    EXPECT_EQ(attrs->normalized_dim_count, 3);
}

TEST_F(TestGraphDescriptorLayernorm, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_X_UID,
                                       toVec(K_LAYERNORM_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_BIAS_UID,
                                          toVec(K_LAYERNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_Y_UID,
                                       toVec(K_LAYERNORM_TENSOR_Y_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_Y_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORM_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS),
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES));
    auto opDesc = createFinalizedLayernormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
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

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::HALF);
    ASSERT_EQ(node.attributes.type, NodeAttributes::LayernormAttributes);

    auto* attrs = node.attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    EXPECT_EQ(attrs->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_EQ(attrs->forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestGraphDescriptorLayernorm, BuildFromOperationWithoutOptionalTensors)
{
    auto xDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_X_UID,
                                       toVec(K_LAYERNORM_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));
    auto biasDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_BIAS_UID,
                                          toVec(K_LAYERNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_Y_UID,
                                       toVec(K_LAYERNORM_TENSOR_Y_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_Y_STRIDES));
    // Pass nullptr for optional mean and inv_variance
    auto opDesc = createFinalizedLayernormOp(xDesc.get(),
                                             scaleDesc.get(),
                                             biasDesc.get(),
                                             epsilonDesc.get(),
                                             yDesc.get(),
                                             nullptr,
                                             nullptr);

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
    ASSERT_EQ(graphT->tensors.size(), 5);

    auto* attrs = graphT->nodes[0]->attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);

    EXPECT_EQ(attrs->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    EXPECT_EQ(attrs->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    EXPECT_FALSE(attrs->mean_tensor_uid.has_value());
    EXPECT_FALSE(attrs->inv_variance_tensor_uid.has_value());
}

} // namespace
