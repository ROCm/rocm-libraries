// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/RMSNormBackwardOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
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

// Helper: create a finalized RMSNormBackwardOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedRMSNormBackwardOp(HipdnnBackendDescriptor* dyDesc,
                                     HipdnnBackendDescriptor* xDesc,
                                     HipdnnBackendDescriptor* scaleDesc,
                                     HipdnnBackendDescriptor* invRmsDesc,
                                     HipdnnBackendDescriptor* dxDesc,
                                     HipdnnBackendDescriptor* dscaleDesc,
                                     HipdnnBackendDescriptor* dbiasDesc,
                                     hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                     const std::string& name = "")
{
    auto wrapper = createDescriptor<RMSNormBackwardOperationDescriptor>();
    auto desc = wrapper->asDescriptor<RMSNormBackwardOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&invRmsDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dscaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dbiasDesc));
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

class TestGraphDescriptorRMSNormBackward : public ::testing::Test
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
        ASSERT_NE(tensor, nullptr) << "Tensor with UID " << expectedUid
                                   << " not found"; // NOLINT(readability-implicit-bool-conversion)
        EXPECT_EQ(tensor->uid, expectedUid);
        EXPECT_EQ(tensor->dims, expectedDims);
        EXPECT_EQ(tensor->strides, expectedStrides);
        EXPECT_EQ(tensor->data_type, expectedDataType);
        EXPECT_EQ(tensor->virtual_, expectedVirtual);
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

TEST_F(TestGraphDescriptorRMSNormBackward, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    auto dxDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get());

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

    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 7);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_DY_UID),
                 K_RMSNORMBACKWARD_TENSOR_DY_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_X_UID),
                 K_RMSNORMBACKWARD_TENSOR_X_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_SCALE_UID),
                 K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID),
                 K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_DX_UID),
                 K_RMSNORMBACKWARD_TENSOR_DX_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID),
                 K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID),
                 K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                 toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                 toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::RMSNormBackwardAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(attrs->x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->inv_rms_tensor_uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(attrs->dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(attrs->dbias_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorRMSNormBackward, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    auto dxDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get(),
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
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::HALF);
}

TEST_F(TestGraphDescriptorRMSNormBackward, OperationNamePreservedInSerialization)
{
    auto dyDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    auto dxDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get(),
                                                   HIPDNN_DATA_FLOAT,
                                                   "test_rmsnormbackward_name");

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {opDesc.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1u);
    EXPECT_EQ(graphT->nodes[0]->name, "test_rmsnormbackward_name");
}

TEST_F(TestGraphDescriptorRMSNormBackward, OperationNameRoundTripThroughLifting)
{
    auto dyDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto invRmsDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
    auto dxDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedRMSNormBackwardOp(dyDesc.get(),
                                                   xDesc.get(),
                                                   scaleDesc.get(),
                                                   invRmsDesc.get(),
                                                   dxDesc.get(),
                                                   dscaleDesc.get(),
                                                   dbiasDesc.get(),
                                                   HIPDNN_DATA_FLOAT,
                                                   "test_rmsnormbackward_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_rmsnormbackward_lifting");
}

} // namespace
