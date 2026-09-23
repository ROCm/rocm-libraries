// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/RMSNormBackwardOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RMSNormBackwardConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestRMSNormBackwardOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<RMSNormBackwardOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<RMSNormBackwardOperationDescriptor>();
    }

    void setRequiredAttributesExcept(std::initializer_list<hipdnnBackendAttributeName_t> skip
                                     = {}) const
    {
        auto desc = getDescriptor();

        auto setIf = [&](hipdnnBackendAttributeName_t attr, auto& tensor) {
            if(std::find(skip.begin(), skip.end(), attr) == skip.end())
            {
                desc->setAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor);
            }
        };
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, _dyDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT, _invRmsDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT, _dxDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT, _dscaleDesc);
        // Compute data type
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT)
           == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void setAllTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dyDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invRmsDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dxDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dscaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_dbiasDesc);
    }

    void setRMSNormBackwardParams() const
    {
        auto desc = getDescriptor();
    }

    void setAllAttributes() const
    {
        setAllTensors();
        setRMSNormBackwardParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    }

    void makeFinalized() const
    {
        setAllAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dyDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invRmsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dxDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dscaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _dbiasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<RMSNormBackwardOperationDescriptor>();
        _dyDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DY_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DY_STRIDES));
        _xDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_X_UID,
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_DIMS),
                                       toVec(K_RMSNORMBACKWARD_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_SCALE_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_SCALE_STRIDES));
        _invRmsDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_INV_RMS_STRIDES));
        _dxDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DX_UID,
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_DIMS),
                                        toVec(K_RMSNORMBACKWARD_TENSOR_DX_STRIDES));
        _dscaleDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID,
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_DIMS),
                                            toVec(K_RMSNORMBACKWARD_TENSOR_DSCALE_STRIDES));
        _dbiasDesc = createFinalizedTensor(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID,
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_DIMS),
                                           toVec(K_RMSNORMBACKWARD_TENSOR_DBIAS_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _dyDesc.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _invRmsDesc.reset();
        _dxDesc.reset();
        _dscaleDesc.reset();
        _dbiasDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RMSNORM_BACKWARD_DESCRIPTOR_EXT);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

class TestRMSNormBackwardOperationDescriptorFinalizeFailsWithout
    : public TestRMSNormBackwardOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestRMSNormBackwardOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestRMSNormBackwardOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                                           HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                                           HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorDy)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dyDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_NE(desc->getDyDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorInvRms)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invRmsDesc));

    ASSERT_EQ(desc->getData().inv_rms_tensor_uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    ASSERT_NE(desc->getInvRmsDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorDx)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dxDesc));

    ASSERT_EQ(desc->getData().dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_NE(desc->getDxDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorDscale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dscaleDesc));

    ASSERT_EQ(desc->getData().dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorDescriptorDbias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_dbiasDesc));

    ASSERT_EQ(desc->getData().dbias_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT, HIPDNN_TYPE_INT64, 1, &_dyDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_dyDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_dyDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedDy = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
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

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
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

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorDyQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorInvRmsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_INV_RMS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorDxQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DX_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorDscaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DSCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorDbiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DBIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BACKWARD_DY_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getDyDesc(), nullptr);
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getInvRmsDesc(), nullptr);
    ASSERT_NE(desc->getDxDesc(), nullptr);
    ASSERT_NE(desc->getDscaleDesc(), nullptr);
    ASSERT_NE(desc->getDbiasDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getDyDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getInvRmsDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    ASSERT_EQ(desc->getDxDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(desc->getDscaleDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(desc->getDbiasDesc()->getData().uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("RMSNormBackwardOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("dy_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_DY_UID)),
              std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_X_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("inv_rms_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dx_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_DX_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dscale_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_DSCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("dbias_uid=" + std::to_string(K_RMSNORMBACKWARD_TENSOR_DBIAS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7);
    ASSERT_EQ(tensors[0]->getData().uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(tensors[6]->getData().uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setAllAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::RMSNormBackwardAttributes);

    auto* attrs = node->attributes.AsRMSNormBackwardAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->dy_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
    ASSERT_EQ(attrs->x_tensor_uid, K_RMSNORMBACKWARD_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->inv_rms_tensor_uid, K_RMSNORMBACKWARD_TENSOR_INV_RMS_UID);
    ASSERT_EQ(attrs->dx_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DX_UID);
    ASSERT_EQ(attrs->dscale_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DSCALE_UID);
    ASSERT_EQ(attrs->dbias_tensor_uid, K_RMSNORMBACKWARD_TENSOR_DBIAS_UID);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestRMSNormBackwardOperationDescriptor,
       GetTensorDescriptorsOrderIsDyXScaleInvRmsDxDscaleDbias)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7);
    // Verify ordering: [DY_EXT, X_EXT, SCALE_EXT, INV_RMS_EXT, DX_EXT, DSCALE_EXT, DBIAS_EXT] matches UIDs [70, 71, 72, 73, 74, 75, 76]
    EXPECT_EQ(tensors[0], desc->getDyDesc());
    EXPECT_EQ(tensors[1], desc->getXDesc());
    EXPECT_EQ(tensors[2], desc->getScaleDesc());
    EXPECT_EQ(tensors[3], desc->getInvRmsDesc());
    EXPECT_EQ(tensors[4], desc->getDxDesc());
    EXPECT_EQ(tensors[5], desc->getDscaleDesc());
    EXPECT_EQ(tensors[6], desc->getDbiasDesc());
}

TEST_F(TestRMSNormBackwardOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7);
    ASSERT_EQ(tensors[0]->getData().uid, K_RMSNORMBACKWARD_TENSOR_DY_UID);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _dyDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_rmsnormbackward_op";

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       static_cast<int64_t>(name.size()),
                                       name.c_str()));

    // Finalize and verify name round-trips
    setAllAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    ASSERT_EQ(count, static_cast<int64_t>(name.size() + 1));

    std::vector<char> buffer(static_cast<size_t>(count));
    int64_t actualCount = 0;
    desc->getAttribute(
        HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, count, &actualCount, buffer.data());
    EXPECT_STREQ(buffer.data(), "test_rmsnormbackward_op");
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
{
    auto desc = getDescriptor();
    const std::string name = "my_op";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(name.size()),
                       name.c_str());
    setAllAttributes();
    desc->finalize();

    int64_t count = 0;
    desc->getAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &count, nullptr);
    EXPECT_EQ(count, static_cast<int64_t>(name.size() + 1));
}

// =============================================================================
// Operation Type Tests
// =============================================================================

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_RMSNORM_BACKWARD_EXT);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormBackwardOperationDescriptor, BuildNodePreservesName)
{
    setAllAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_rmsnormbackward";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_BACKWARD_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_rmsnormbackward");
}
