// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/LayernormBackwardOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestLayernormBackwardOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<LayernormBackwardOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<LayernormBackwardOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dyDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_meanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invVarianceDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_epsilonDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dxDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dscaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dbiasDesc);
    }

    void setLayernormBackwardParams() const
    {
        auto desc = getDescriptor();
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setLayernormBackwardParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dyDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _meanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _epsilonDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dscaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dbiasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<LayernormBackwardOperationDescriptor>();
        _dyDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DY_STRIDES));
        _xDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORMBACKWARD_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_SCALE_STRIDES));
        _meanDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORMBACKWARD_TENSOR_MEAN_STRIDES));
        _invVarianceDesc
            = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID,
                                    toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_DIMS),
                                    toVec(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_STRIDES));
        _epsilonDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORMBACKWARD_TENSOR_EPSILON_STRIDES));
        _dxDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_LAYERNORMBACKWARD_TENSOR_DX_STRIDES));
        _dscaleDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_LAYERNORMBACKWARD_TENSOR_DSCALE_STRIDES));
        _dbiasDesc = createFinalizedTensor(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_LAYERNORMBACKWARD_TENSOR_DBIAS_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _dyDesc.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _meanDesc.reset();
        _invVarianceDesc.reset();
        _epsilonDesc.reset();
        _dxDesc.reset();
        _dscaleDesc.reset();
        _dbiasDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_LAYERNORM_BACKWARD_DESCRIPTOR);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutDyTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dbiasDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dbiasDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutScaleTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dbiasDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutDxTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dscaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dbiasDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutDscaleTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dbiasDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutDbiasTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_meanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_epsilonDesc);
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_dscaleDesc);
    setLayernormBackwardParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setLayernormBackwardParams();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorDy)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dyDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_meanDesc));

    ASSERT_EQ(desc->getData().mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorInvVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invVarianceDesc));

    ASSERT_EQ(desc->getData().inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorEpsilon)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_epsilonDesc));

    ASSERT_EQ(desc->getData().epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorDx)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_dxDesc));

    ASSERT_EQ(desc->getData().dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorDscale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dscaleDesc));

    ASSERT_EQ(desc->getData().dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorDescriptorDbias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dbiasDesc));

    ASSERT_EQ(desc->getData().dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY, HIPDNN_TYPE_INT64, 1, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_dyDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_dyDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestLayernormBackwardOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedDy = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedDy)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedDy, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardDy(retrievedDy);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Query Mode Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorDyQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_X,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_SCALE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorMeanQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_MEAN,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorInvVarianceQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_INV_VARIANCE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorEpsilonQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_EPSILON,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorDxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DX,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorDscaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DSCALE,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorDbiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DBIAS,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BACKWARD_DY,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDyDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    ASSERT_EQ(desc->getInvVarianceDesc()->getData().uid,
              K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    ASSERT_EQ(desc->getDxDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(desc->getDscaleDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(desc->getDbiasDesc()->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("LayernormBackwardOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("dy_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_DY_UID)),
              std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_X_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("mean_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_MEAN_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("inv_variance_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID)),
        std::string::npos);
    ASSERT_NE(str.find("epsilon_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dx_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_DX_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dscale_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dbias_uid=" + std::to_string(K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9);
    ASSERT_EQ(tensors[0]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    ASSERT_EQ(tensors[6]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(tensors[7]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(tensors[8]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestLayernormBackwardOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::LayernormBackwardAttributes);

    auto* attrs = node->attributes.AsLayernormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->dy_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(attrs->x_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->mean_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_MEAN_UID);
    ASSERT_EQ(attrs->inv_variance_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(attrs->epsilon_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_EPSILON_UID);
    ASSERT_EQ(attrs->dx_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(attrs->dscale_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(attrs->dbias_tensor_uid, K_LAYERNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestLayernormBackwardOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestLayernormBackwardOperationDescriptor,
       GetTensorDescriptorsOrderIsDyXScaleMeanInvVarianceEpsilonDxDscaleDbias)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9);
    // Verify ordering: [DY, X, SCALE, MEAN, INV_VARIANCE, EPSILON, DX, DSCALE, DBIAS] matches UIDs [10, 11, 12, 13, 14, 15, 16, 17, 18]
    EXPECT_EQ(tensors[0], desc->getDyDesc());
    EXPECT_EQ(tensors[1], desc->getXDesc());
    EXPECT_EQ(tensors[2], desc->getScaleDesc());
    EXPECT_EQ(tensors[3], desc->getMeanDesc());
    EXPECT_EQ(tensors[4], desc->getInvVarianceDesc());
    EXPECT_EQ(tensors[5], desc->getEpsilonDesc());
    EXPECT_EQ(tensors[6], desc->getDxDesc());
    EXPECT_EQ(tensors[7], desc->getDscaleDesc());
    EXPECT_EQ(tensors[8], desc->getDbiasDesc());
}

TEST_F(TestLayernormBackwardOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 9);
    ASSERT_EQ(tensors[0]->getData().uid, K_LAYERNORMBACKWARD_TENSOR_DY_UID);
}

TEST_F(TestLayernormBackwardOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _dyDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_layernormbackward_op";

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(name.size()),
                                       name.c_str()));

    // Finalize and verify name round-trips
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_layernormbackward_op");
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
{
    auto desc = getDescriptor();
    const std::string name = "my_op";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    setRequiredAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, static_cast<int64_t>(name.size() + 1));
}

// =============================================================================
// Operation Type Tests
// =============================================================================

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_LAYERNORM_BACKWARD_EXT);
}

TEST_F(TestLayernormBackwardOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormBackwardOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_layernormbackward";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_layernormbackward");
}
