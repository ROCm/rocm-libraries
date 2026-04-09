// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "descriptors/GraphDescriptor.hpp"
#include "descriptors/LayernormBackwardOperationDescriptor.hpp"
#include "hipdnn_backend.h"
#include "mocks/MockHandle.hpp"

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <array>
#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

namespace
{

// Helper: create a finalized LayernormBackwardOperationDescriptor from tensor descriptors
inline std::unique_ptr<HipdnnBackendDescriptor>
    createFinalizedLayernormBackwardOp(HipdnnBackendDescriptor* dyDesc,
                                       HipdnnBackendDescriptor* xDesc,
                                       HipdnnBackendDescriptor* scaleDesc,
                                       HipdnnBackendDescriptor* meanDesc,
                                       HipdnnBackendDescriptor* invVarianceDesc,
                                       HipdnnBackendDescriptor* epsilonDesc,
                                       HipdnnBackendDescriptor* dxDesc,
                                       HipdnnBackendDescriptor* dscaleDesc,
                                       HipdnnBackendDescriptor* dbiasDesc,
                                       hipdnnDataType_t computeType = HIPDNN_DATA_FLOAT,
                                       const std::string& name = "")
{
    auto wrapper = createDescriptor<LayernormBackwardOperationDescriptor>();
    auto desc = wrapper->asDescriptor<LayernormBackwardOperationDescriptor>();

    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dyDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&xDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&scaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&meanDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&invVarianceDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&epsilonDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dxDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dscaleDesc));
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(&dbiasDesc));
    desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

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

class TestGraphDescriptorLayernormBackward : public ::testing::Test
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

    static void verifyLayernormBackwardNode(const NodeT& node,
                                            DataType expectedComputeType,
                                            int64_t expectedDyUid,
                                            int64_t expectedXUid,
                                            int64_t expectedScaleUid,
                                            int64_t expectedMeanUid,
                                            int64_t expectedInvVarianceUid,
                                            int64_t expectedEpsilonUid,
                                            int64_t expectedDxUid,
                                            int64_t expectedDscaleUid,
                                            int64_t expectedDbiasUid)
    {
        EXPECT_EQ(node.compute_data_type, expectedComputeType);
        ASSERT_EQ(node.attributes.type, NodeAttributes::LayernormBackwardAttributes);

        auto* attrs = node.attributes.AsLayernormBackwardAttributes();
        ASSERT_NE(attrs, nullptr);

        EXPECT_EQ(attrs->dy_tensor_uid, expectedDyUid);
        EXPECT_EQ(attrs->x_tensor_uid, expectedXUid);
        EXPECT_EQ(attrs->scale_tensor_uid, expectedScaleUid);
        EXPECT_EQ(attrs->mean_tensor_uid, expectedMeanUid);
        EXPECT_EQ(attrs->inv_variance_tensor_uid, expectedInvVarianceUid);
        EXPECT_EQ(attrs->epsilon_tensor_uid, expectedEpsilonUid);
        EXPECT_EQ(attrs->dx_tensor_uid, expectedDxUid);
        EXPECT_EQ(attrs->dscale_tensor_uid, expectedDscaleUid);
        EXPECT_EQ(attrs->dbias_tensor_uid, expectedDbiasUid);
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

TEST_F(TestGraphDescriptorLayernormBackward, BuildFromSingleOperation)
{
    auto dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc
        = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    auto dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedLayernormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     epsilonDesc.get(),
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
    ASSERT_EQ(graphT->tensors.size(), 9);

    // Verify tensor attributes
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DY_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_X_UID),
                 K_LAYERNORMBACKWARD_TENSOR_X_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID),
                 K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID),
                 K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DX_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES),
                 DataType::FLOAT);

    // Verify node attributes
    ASSERT_EQ(graphT->nodes[0]->attributes.type, NodeAttributes::LayernormBackwardAttributes);

    auto* attrs = graphT->nodes[0]->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(graphT->nodes[0]->compute_data_type, DataType::FLOAT);

    // Verify tensor UID references
    EXPECT_EQ(attrs->dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    EXPECT_EQ(attrs->x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    EXPECT_EQ(attrs->scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    EXPECT_EQ(attrs->mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    EXPECT_EQ(attrs->inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    EXPECT_EQ(attrs->epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    EXPECT_EQ(attrs->dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    EXPECT_EQ(attrs->dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    EXPECT_EQ(attrs->dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify default node name is empty
    EXPECT_TRUE(graphT->nodes[0]->name.empty());
}

TEST_F(TestGraphDescriptorLayernormBackward, ComputeDataTypePreserved)
{
    auto dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc
        = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    auto dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedLayernormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     epsilonDesc.get(),
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

TEST_F(TestGraphDescriptorLayernormBackward, LayernormBackwardAttributesPreserved)
{
    auto dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc
        = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    auto dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));

    // Create op with non-default parameters to test graph roundtrip
    auto wrapper = createDescriptor<LayernormBackwardOperationDescriptor>();
    auto opDesc = wrapper->asDescriptor<LayernormBackwardOperationDescriptor>();

    HipdnnBackendDescriptor* dyPtr = dyDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dyPtr));
    HipdnnBackendDescriptor* xPtr = xDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&xPtr));
    HipdnnBackendDescriptor* scalePtr = scaleDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&scalePtr));
    HipdnnBackendDescriptor* meanPtr = meanDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&meanPtr));
    HipdnnBackendDescriptor* invVariancePtr = invVarianceDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&invVariancePtr));
    HipdnnBackendDescriptor* epsilonPtr = epsilonDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&epsilonPtr));
    HipdnnBackendDescriptor* dxPtr = dxDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dxPtr));
    HipdnnBackendDescriptor* dscalePtr = dscaleDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dscalePtr));
    HipdnnBackendDescriptor* dbiasPtr = dbiasDesc.get();
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                         HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                         1,
                         static_cast<const void*>(&dbiasPtr));

    auto computeType = HIPDNN_DATA_FLOAT;
    opDesc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    // Set operation name
    const std::string opName = "test_layernormbackward";
    opDesc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                         HIPDNN_TYPE_CHAR,
                         static_cast<int64_t>(opName.size()),
                         opName.c_str());
    opDesc->finalize();

    auto desc = getDescriptor();
    setHandle();

    std::array<HipdnnBackendDescriptor*, 1> ops = {wrapper.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATIONGRAPH_OPS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       static_cast<const void*>(ops.data()));
    desc->finalize();

    auto serialized = desc->getSerializedGraph();
    const auto graphT = UnPackGraph(serialized.ptr);

    ASSERT_EQ(graphT->nodes.size(), 1);
    ASSERT_EQ(graphT->tensors.size(), 9);

    // Verify tensors
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DY_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_X_UID),
                 K_LAYERNORMBACKWARD_TENSOR_X_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID),
                 K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID),
                 K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DX_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES),
                 DataType::FLOAT);
    verifyTensor(findTensorByUid(*graphT, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID),
                 K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                 toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES),
                 DataType::FLOAT);

    // Verify node with non-default attribute values
    verifyLayernormBackwardNode(*graphT->nodes[0],
                                DataType::FLOAT,
                                K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);

    // Verify operation name
    EXPECT_EQ(graphT->nodes[0]->name, "test_layernormbackward");
}

TEST_F(TestGraphDescriptorLayernormBackward, OperationNamePreservedInSerialization)
{
    auto dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc
        = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    auto dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedLayernormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     epsilonDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     HIPDNN_DATA_FLOAT,
                                                     "test_layernormbackward_name");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_layernormbackward_name");
}

TEST_F(TestGraphDescriptorLayernormBackward, OperationNameRoundTripThroughLifting)
{
    auto dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
    auto xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
    auto scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
    auto meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
    auto invVarianceDesc
        = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
    auto epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
    auto dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
    auto dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
    auto dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
    auto opDesc = createFinalizedLayernormBackwardOp(dyDesc.get(),
                                                     xDesc.get(),
                                                     scaleDesc.get(),
                                                     meanDesc.get(),
                                                     invVarianceDesc.get(),
                                                     epsilonDesc.get(),
                                                     dxDesc.get(),
                                                     dscaleDesc.get(),
                                                     dbiasDesc.get(),
                                                     HIPDNN_DATA_FLOAT,
                                                     "test_layernormbackward_lifting");

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
    EXPECT_EQ(graphT->nodes[0]->name, "test_layernormbackward_lifting");
}

} // namespace
