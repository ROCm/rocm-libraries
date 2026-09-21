// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnNormFwdPhase.h"
#include "TensorDescriptorTestUtils.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/RMSNormOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <string>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedRMSNormOp(HipdnnBackendDescriptor* xDesc,
                             HipdnnBackendDescriptor* scaleDesc,
                             HipdnnBackendDescriptor* epsilonDesc,
                             HipdnnBackendDescriptor* yDesc,
                             HipdnnBackendDescriptor* biasDesc,
                             HipdnnBackendDescriptor* invRmsDesc,
                             hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                             hipdnnNormFwdPhase_t forwardPhase = HIPDNN_NORM_FWD_TRAINING,
                             const std::string& name = "")
{
    auto wrapper = createDescriptor<RMSNormOperationDescriptor>();
    auto desc = wrapper->asDescriptor<RMSNormOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&epsilonDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&yDesc));
    if(biasDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&biasDesc));
    }
    if(invRmsDesc != nullptr)
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           static_cast<const void*>(&invRmsDesc));
    }

    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT, HIPDNN_TYPE_NORM_FWD_PHASE, 1, &forwardPhase);
    desc->setAttribute(HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    if(!name.empty())
    {
        desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                           HIPDNN_TYPE_CHAR,
                           static_cast<int64_t>(name.size()),
                           name.data());
    }

    desc->finalize();
    return wrapper;
}

} // namespace

class TestGraphDescriptorRMSNorm : public ::testing::Test
{
public:
    static const TensorAttributesT* findTensorByUid(const GraphT& graphT, int64_t uid)
    {
        for(const auto& tensor : graphT.tensors)
        {
            if(tensor->uid == uid)
            {
                return tensor.get();
            }
        }
        return nullptr;
    }

    static void verifyTensor(const TensorAttributesT* tensor,
                             int64_t expectedUid,
                             const std::vector<int64_t>& expectedDims,
                             const std::vector<int64_t>& expectedStrides,
                             DataType expectedDataType,
                             bool expectedVirtual = false)
    {
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid << " not found";
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
    }

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

TEST_F(TestGraphDescriptorRMSNorm, BuildFromSingleRMSNormOperation)
{
    auto xDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_X_UID, toVec(K_RMSNORM_TENSOR_X_DIMS), toVec(K_RMSNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORM_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_RMSNORM_TENSOR_EPSILON_UID,
                                             toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_Y_UID, toVec(K_RMSNORM_TENSOR_Y_DIMS), toVec(K_RMSNORM_TENSOR_Y_STRIDES));
    auto biasDesc = createFinalizedTensor(K_RMSNORM_TENSOR_BIAS_UID,
                                          toVec(K_RMSNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_RMSNORM_TENSOR_BIAS_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORM_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES));

    auto rmsnormOp = createFinalizedRMSNormOp(xDesc.get(),
                                              scaleDesc.get(),
                                              epsilonDesc.get(),
                                              yDesc.get(),
                                              biasDesc.get(),
                                              invRmsDesc.get());

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {rmsnormOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(serialized.ptr), serialized.size);
    ASSERT_TRUE(verifier.VerifyBuffer<Graph>());

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 6);

    verifyTensor(findTensorByUid(*graphT, K_RMSNORM_TENSOR_X_UID),
                 K_RMSNORM_TENSOR_X_UID,
                 toVec(K_RMSNORM_TENSOR_X_DIMS),
                 toVec(K_RMSNORM_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORM_TENSOR_SCALE_UID),
                 K_RMSNORM_TENSOR_SCALE_UID,
                 toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                 toVec(K_RMSNORM_TENSOR_SCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORM_TENSOR_Y_UID),
                 K_RMSNORM_TENSOR_Y_UID,
                 toVec(K_RMSNORM_TENSOR_Y_DIMS),
                 toVec(K_RMSNORM_TENSOR_Y_STRIDES),
                 DataType::FLOAT);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node.attributes.type, NodeAttributes::RMSNormAttributes);

    auto* rmsnormAttrs = node.attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnormAttrs, nullptr);
    EXPECT_EQ(rmsnormAttrs->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsnormAttrs->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsnormAttrs->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsnormAttrs->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_TRUE(rmsnormAttrs->bias_tensor_uid.has_value());
    EXPECT_EQ(*rmsnormAttrs->bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_TRUE(rmsnormAttrs->inv_rms_tensor_uid.has_value());
    EXPECT_EQ(*rmsnormAttrs->inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    EXPECT_EQ(rmsnormAttrs->forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestGraphDescriptorRMSNorm, ComputeDataTypePreserved)
{
    auto xDesc = createFinalizedTensor(K_RMSNORM_TENSOR_X_UID,
                                       toVec(K_RMSNORM_TENSOR_X_DIMS),
                                       toVec(K_RMSNORM_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_HALF);
    auto scaleDesc = createFinalizedTensor(K_RMSNORM_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORM_TENSOR_SCALE_STRIDES),
                                           HIPDNN_DATA_HALF);
    auto epsilonDesc = createFinalizedTensor(K_RMSNORM_TENSOR_EPSILON_UID,
                                             toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES),
                                             HIPDNN_DATA_HALF);
    auto yDesc = createFinalizedTensor(K_RMSNORM_TENSOR_Y_UID,
                                       toVec(K_RMSNORM_TENSOR_Y_DIMS),
                                       toVec(K_RMSNORM_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_HALF);
    auto biasDesc = createFinalizedTensor(K_RMSNORM_TENSOR_BIAS_UID,
                                          toVec(K_RMSNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_RMSNORM_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_HALF);
    auto invRmsDesc = createFinalizedTensor(K_RMSNORM_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES),
                                            HIPDNN_DATA_HALF);

    auto rmsnormOp = createFinalizedRMSNormOp(xDesc.get(),
                                              scaleDesc.get(),
                                              epsilonDesc.get(),
                                              yDesc.get(),
                                              biasDesc.get(),
                                              invRmsDesc.get(),
                                              HIPDNN_DATA_HALF);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {rmsnormOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);

    const auto& node = *graphT->nodes[0];
    EXPECT_EQ(node.compute_data_type, DataType::HALF);
    ASSERT_EQ(node.attributes.type, NodeAttributes::RMSNormAttributes);

    auto* rmsnormAttrs = node.attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnormAttrs, nullptr);
    EXPECT_EQ(rmsnormAttrs->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsnormAttrs->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsnormAttrs->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsnormAttrs->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
}

// Verifies that an INFERENCE-mode operation (no bias, no inv_rms) serializes
// correctly with only 4 tensors and no optional uid fields set.
TEST_F(TestGraphDescriptorRMSNorm, BuildFromOperationWithoutOptionalTensors)
{
    auto xDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_X_UID, toVec(K_RMSNORM_TENSOR_X_DIMS), toVec(K_RMSNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORM_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_RMSNORM_TENSOR_EPSILON_UID,
                                             toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_Y_UID, toVec(K_RMSNORM_TENSOR_Y_DIMS), toVec(K_RMSNORM_TENSOR_Y_STRIDES));

    // Pass nullptr for optional bias and inv_rms (INFERENCE mode)
    auto rmsnormOp = createFinalizedRMSNormOp(xDesc.get(),
                                              scaleDesc.get(),
                                              epsilonDesc.get(),
                                              yDesc.get(),
                                              nullptr,
                                              nullptr,
                                              HIPDNN_DATA_FLOAT,
                                              HIPDNN_NORM_FWD_INFERENCE);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {rmsnormOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    ASSERT_NE(serialized.ptr, nullptr);
    ASSERT_GT(serialized.size, 0UL);

    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 4); // x, scale, epsilon, y only

    auto* rmsnormAttrs = graphT->nodes[0]->attributes.AsRMSNormAttributes();
    ASSERT_NE(rmsnormAttrs, nullptr);

    EXPECT_EQ(rmsnormAttrs->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    EXPECT_EQ(rmsnormAttrs->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    EXPECT_EQ(rmsnormAttrs->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    EXPECT_EQ(rmsnormAttrs->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    EXPECT_FALSE(rmsnormAttrs->bias_tensor_uid.has_value());
    EXPECT_FALSE(rmsnormAttrs->inv_rms_tensor_uid.has_value());
    EXPECT_EQ(rmsnormAttrs->forward_phase, NormFwdPhase::INFERENCE);
}

// Verifies that the operation name set via HIPDNN_ATTR_OPERATION_NAME_EXT is
// preserved in the serialized FlatBuffer graph node.
TEST_F(TestGraphDescriptorRMSNorm, NamePreservedInFlatBuffer)
{
    auto xDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_X_UID, toVec(K_RMSNORM_TENSOR_X_DIMS), toVec(K_RMSNORM_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORM_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_RMSNORM_TENSOR_EPSILON_UID,
                                             toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));
    auto yDesc = createFinalizedTensor(
        K_RMSNORM_TENSOR_Y_UID, toVec(K_RMSNORM_TENSOR_Y_DIMS), toVec(K_RMSNORM_TENSOR_Y_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORM_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES));

    const std::string opName = "my_rmsnorm_op";
    auto rmsnormOp = createFinalizedRMSNormOp(xDesc.get(),
                                              scaleDesc.get(),
                                              epsilonDesc.get(),
                                              yDesc.get(),
                                              nullptr,
                                              invRmsDesc.get(),
                                              HIPDNN_DATA_FLOAT,
                                              HIPDNN_NORM_FWD_TRAINING,
                                              opName);

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {rmsnormOp.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       static_cast<const void*>(ops.data())));
    ASSERT_NO_THROW(desc->finalize());

    auto serialized = desc->getSerializedGraph();
    auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->name, opName);
}
