// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormBackwardOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/batchnorm_backward_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;

class TestBatchnormBackwardOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BatchnormBackwardOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BatchnormBackwardOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dyDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dxDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dscaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dbiasDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_meanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invVarianceDesc);
    }

    void setTensors(
        const std::unordered_map<hipdnnBackendAttributeName_t, const void*>& tensorMap) const
    {
        auto desc = getDescriptor();
        for(const auto& [attributeName, tensorDesc] : tensorMap)
        {
            desc->setAttribute(attributeName,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               static_cast<const void*>(&tensorDesc));
        }
    }

    void setRequiredAttributes() const
    {
        setTensors();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
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
    std::unique_ptr<HipdnnBackendDescriptor> _dxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dscaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dbiasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _meanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _peerStatsDesc0 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _peerStatsDesc1 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BatchnormBackwardOperationDescriptor>();
        _dyDesc = createFinalizedTensor(60, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _xDesc = createFinalizedTensor(61, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _scaleDesc = createFinalizedTensor(62, {1, 64, 1, 1}, {64, 1, 1, 1});
        _dxDesc = createFinalizedTensor(63, {1, 64, 32, 32}, {65536, 1024, 32, 1});
        _dscaleDesc = createFinalizedTensor(64, {1, 64, 1, 1}, {64, 1, 1, 1});
        _dbiasDesc = createFinalizedTensor(65, {1, 64, 1, 1}, {64, 1, 1, 1});
        _meanDesc = createFinalizedTensor(7);
        _invVarianceDesc = createFinalizedTensor(8);
        _peerStatsDesc0 = createFinalizedTensor(110);
        _peerStatsDesc1 = createFinalizedTensor(111);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _dyDesc.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _dxDesc.reset();
        _dscaleDesc.reset();
        _dbiasDesc.reset();
        _meanDesc.reset();
        _invVarianceDesc.reset();
        _peerStatsDesc0.reset();
        _peerStatsDesc1.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_BACKWARD_DESCRIPTOR_EXT);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutDyTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutScaleTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutDxTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutDscaleTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutDbiasTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeSucceedsWithoutMeanAndInvVariance)
{
    auto desc = getDescriptor();
    setTensors({{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()}});
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    ASSERT_NO_THROW(desc->finalize());
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getMeanDesc(), nullptr);
    ASSERT_EQ(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithMeanButNotInvVariance)
{
    auto desc = getDescriptor();
    setTensors({{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT, _meanDesc.get()}});
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithInvVarianceButNotMean)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, _dyDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT, _dxDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT, _dscaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT, _dbiasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT, _invVarianceDesc.get()}});
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorDy)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dyDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().dy_tensor_uid, 60);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, 61);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, 62);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorDx)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dxDesc));

    ASSERT_EQ(desc->getData().dx_tensor_uid, 63);
    ASSERT_NE(desc->getDxDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorDscale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dscaleDesc));

    ASSERT_EQ(desc->getData().dscale_tensor_uid, 64);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorDbias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dbiasDesc));

    ASSERT_EQ(desc->getData().dbias_tensor_uid, 65);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_meanDesc));

    ASSERT_EQ(desc->getData().mean_tensor_uid, 7);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorDescriptorInvVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invVarianceDesc));

    ASSERT_EQ(desc->getData().inv_variance_tensor_uid, 8);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT, HIPDNN_TYPE_INT64, 1, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_dyDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_dyDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawDy = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawDy)));
    std::unique_ptr<HipdnnBackendDescriptor> retrievedDy(rawDy);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedDy, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
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

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorDyQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorDxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DX_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorDscaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DSCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorDbiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DBIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorMeanQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorInvVarianceQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDyDesc()->getData().uid, 60);
    ASSERT_EQ(desc->getXDesc()->getData().uid, 61);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, 62);
    ASSERT_EQ(desc->getDxDesc()->getData().uid, 63);
    ASSERT_EQ(desc->getDscaleDesc()->getData().uid, 64);
    ASSERT_EQ(desc->getDbiasDesc()->getData().uid, 65);
    ASSERT_EQ(desc->getMeanDesc()->getData().uid, 7);
    ASSERT_EQ(desc->getInvVarianceDesc()->getData().uid, 8);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("BatchnormBackwardOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("dy_uid=60"), std::string::npos);
    ASSERT_NE(str.find("x_uid=61"), std::string::npos);
    ASSERT_NE(str.find("scale_uid=62"), std::string::npos);
    ASSERT_NE(str.find("dx_uid=63"), std::string::npos);
    ASSERT_NE(str.find("dscale_uid=64"), std::string::npos);
    ASSERT_NE(str.find("dbias_uid=65"), std::string::npos);
    ASSERT_NE(str.find("mean_uid=7"), std::string::npos);
    ASSERT_NE(str.find("inv_variance_uid=8"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    ASSERT_EQ(tensors[0]->getData().uid, 60);
    ASSERT_EQ(tensors[1]->getData().uid, 61);
    ASSERT_EQ(tensors[2]->getData().uid, 62);
    ASSERT_EQ(tensors[3]->getData().uid, 63);
    ASSERT_EQ(tensors[4]->getData().uid, 64);
    ASSERT_EQ(tensors[5]->getData().uid, 65);
    ASSERT_EQ(tensors[6]->getData().uid, 7);
    ASSERT_EQ(tensors[7]->getData().uid, 8);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BatchnormBackwardAttributes);

    auto* attrs = node->attributes.AsBatchnormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->dy_tensor_uid, 60);
    ASSERT_EQ(attrs->x_tensor_uid, 61);
    ASSERT_EQ(attrs->scale_tensor_uid, 62);
    ASSERT_EQ(attrs->dx_tensor_uid, 63);
    ASSERT_EQ(attrs->dscale_tensor_uid, 64);
    ASSERT_EQ(attrs->dbias_tensor_uid, 65);
    ASSERT_EQ(attrs->mean_tensor_uid, 7);
    ASSERT_EQ(attrs->inv_variance_tensor_uid, 8);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestBatchnormBackwardOperationDescriptor,
       GetTensorDescriptorsOrderIsDyXScaleDxDscaleDbiasMeanInvVariance)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    // Verify ordering: [DY, X, SCALE, DX, DSCALE, DBIAS, MEAN, INV_VARIANCE] matches UIDs [60, 61, 62, 63, 64, 65, 7, 8]
    EXPECT_EQ(tensors[0], desc->getDyDesc());
    EXPECT_EQ(tensors[1], desc->getXDesc());
    EXPECT_EQ(tensors[2], desc->getScaleDesc());
    EXPECT_EQ(tensors[3], desc->getDxDesc());
    EXPECT_EQ(tensors[4], desc->getDscaleDesc());
    EXPECT_EQ(tensors[5], desc->getDbiasDesc());
    EXPECT_EQ(tensors[6], desc->getMeanDesc());
    EXPECT_EQ(tensors[7], desc->getInvVarianceDesc());
}

TEST_F(TestBatchnormBackwardOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    ASSERT_EQ(tensors[0]->getData().uid, 60);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _dyDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Tensor Array Tests - PeerStats
// =============================================================================

TEST_F(TestBatchnormBackwardOperationDescriptor, SetPeerStatsTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> descs = {_peerStatsDesc0.get(), _peerStatsDesc1.get()};

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(descs.data())));

    auto& data = desc->getData();
    ASSERT_EQ(data.peer_stats_tensor_uid.size(), 2);
    EXPECT_EQ(data.peer_stats_tensor_uid[0], 110);
    EXPECT_EQ(data.peer_stats_tensor_uid[1], 111);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetPeerStatsTensorArrayFailsNotFinalized)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 1> descs = {_unfinalizedTensor.get()};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           descs.data()),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, SetPeerStatsTensorArrayFailsNullPointer)
{
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBatchnormBackwardOperationDescriptor, GetPeerStatsTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> descs = {_peerStatsDesc0.get(), _peerStatsDesc1.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(descs.data()));
    setRequiredAttributes();
    desc->finalize();

    std::array<HipdnnBackendDescriptor*, 2> retrieved = {};
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BACKWARD_PEER_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &elementCount,
                                       static_cast<void*>(retrieved.data())));
    std::unique_ptr<HipdnnBackendDescriptor> retrieved0(retrieved[0]);
    std::unique_ptr<HipdnnBackendDescriptor> retrieved1(retrieved[1]);

    ASSERT_EQ(elementCount, 2);
    ASSERT_NE(retrieved0, nullptr);
    ASSERT_NE(retrieved1, nullptr);
}
