// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/RMSNormOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/RMSNormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;

class TestRMSNormOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<RMSNormOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<RMSNormOperationDescriptor>();
    }

    void setAllAttributesExcept(std::initializer_list<hipdnnBackendAttributeName_t> skip = {}) const
    {
        auto desc = getDescriptor();
        auto setIf = [&](hipdnnBackendAttributeName_t attr, auto& tensor) {
            if(std::find(skip.begin(), skip.end(), attr) == skip.end())
            {
                desc->setAttribute(attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensor);
            }
        };
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT, _epsilonDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, _yDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, _biasDesc);
        setIf(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT, _invRmsDesc);
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT)
           == skip.end())
        {
            auto forwardPhase = HIPDNN_NORM_FWD_TRAINING;
            desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                               HIPDNN_TYPE_NORM_FWD_PHASE,
                               1,
                               &forwardPhase);
        }
    }

    void makeFinalized() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _epsilonDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _biasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invRmsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<RMSNormOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_RMSNORM_TENSOR_X_UID,
                                       hipdnn_tests::toVec(K_RMSNORM_TENSOR_X_DIMS),
                                       hipdnn_tests::toVec(K_RMSNORM_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_RMSNORM_TENSOR_SCALE_UID,
                                           hipdnn_tests::toVec(K_RMSNORM_TENSOR_SCALE_DIMS),
                                           hipdnn_tests::toVec(K_RMSNORM_TENSOR_SCALE_STRIDES));
        _epsilonDesc = createFinalizedTensor(K_RMSNORM_TENSOR_EPSILON_UID,
                                             hipdnn_tests::toVec(K_RMSNORM_TENSOR_EPSILON_DIMS),
                                             hipdnn_tests::toVec(K_RMSNORM_TENSOR_EPSILON_STRIDES));
        _yDesc = createFinalizedTensor(K_RMSNORM_TENSOR_Y_UID,
                                       hipdnn_tests::toVec(K_RMSNORM_TENSOR_Y_DIMS),
                                       hipdnn_tests::toVec(K_RMSNORM_TENSOR_Y_STRIDES));
        _biasDesc = createFinalizedTensor(K_RMSNORM_TENSOR_BIAS_UID,
                                          hipdnn_tests::toVec(K_RMSNORM_TENSOR_BIAS_DIMS),
                                          hipdnn_tests::toVec(K_RMSNORM_TENSOR_BIAS_STRIDES));
        _invRmsDesc = createFinalizedTensor(K_RMSNORM_TENSOR_INV_RMS_UID,
                                            hipdnn_tests::toVec(K_RMSNORM_TENSOR_INV_RMS_DIMS),
                                            hipdnn_tests::toVec(K_RMSNORM_TENSOR_INV_RMS_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _epsilonDesc.reset();
        _yDesc.reset();
        _biasDesc.reset();
        _invRmsDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_RMSNORM_DESCRIPTOR_EXT);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeSucceedsWithoutOptionalTensors)
{
    setAllAttributesExcept(
        {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT});
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutScaleTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutEpsilonTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setAllAttributesExcept({HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, FinalizeFailsWithoutForwardPhase)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorEpsilon)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_epsilonDesc));

    ASSERT_EQ(desc->getData().epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_biasDesc));

    ASSERT_TRUE(desc->getData().bias_tensor_uid.has_value());
    ASSERT_EQ(*desc->getData().bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorDescriptorInvRms)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invRmsDesc));

    ASSERT_TRUE(desc->getData().inv_rms_tensor_uid.has_value());
    ASSERT_EQ(*desc->getData().inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    ASSERT_NE(desc->getInvRmsDesc(), nullptr);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestRMSNormOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Forward Phase
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, SetForwardPhase)
{
    auto desc = getDescriptor();
    auto forwardPhase = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT, HIPDNN_TYPE_NORM_FWD_PHASE, 1, &forwardPhase));

    ASSERT_EQ(desc->getData().forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestRMSNormOperationDescriptor, SetForwardPhaseWrongElementCount)
{
    auto desc = getDescriptor();
    // Value is irrelevant — this test exercises the elementCount != 1 error path.
    auto forwardPhase = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                                  HIPDNN_TYPE_NORM_FWD_PHASE,
                                                  2,
                                                  &forwardPhase),
                               HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestRMSNormOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRMSNormOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawX)));
    const std::unique_ptr<HipdnnBackendDescriptor> retrievedX(rawX);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Forward Phase
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, GetAttributeForwardPhase)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnNormFwdPhase_t forwardPhase = {};
    int64_t forwardPhaseCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       1,
                                       &forwardPhaseCount,
                                       &forwardPhase));
    ASSERT_EQ(forwardPhaseCount, 1);
    EXPECT_EQ(forwardPhase, HIPDNN_NORM_FWD_TRAINING);
}

// =============================================================================
// GetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorEpsilonQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_EPSILON_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorBiasQueryReturnsOne)
{
    // makeFinalized() sets all optional tensors including bias — query returns 1.
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorBiasQueryReturnsZeroWhenAbsent)
{
    // Finalize without bias — query should report 0 elements available.
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT});
    getDescriptor()->finalize();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 0);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorInvRmsQueryReturnsOne)
{
    // makeFinalized() sets all optional tensors including inv_rms — query returns 1.
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorInvRmsQueryReturnsZeroWhenAbsent)
{
    // Finalize without inv_rms — query should report 0 elements available.
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT});
    getDescriptor()->finalize();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 0);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeForwardPhaseQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestRMSNormOperationDescriptor, GetAttributeForwardPhaseQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_RMSNORM_FWD_PHASE_EXT,
                                                  HIPDNN_TYPE_NORM_FWD_PHASE,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    ASSERT_NE(desc->getInvRmsDesc(), nullptr);

    ASSERT_EQ(desc->getXDesc()->getData().uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getEpsilonDesc()->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_EQ(desc->getBiasDesc()->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(desc->getInvRmsDesc()->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("RMSNormOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_RMSNORM_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_RMSNORM_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("epsilon_uid=" + std::to_string(K_RMSNORM_TENSOR_EPSILON_UID)),
              std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_RMSNORM_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("bias_uid=" + std::to_string(K_RMSNORM_TENSOR_BIAS_UID)), std::string::npos);
    ASSERT_NE(str.find("inv_rms_uid=" + std::to_string(K_RMSNORM_TENSOR_INV_RMS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestRMSNormOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_RMSNORM_TENSOR_INV_RMS_UID);
}

TEST_F(TestRMSNormOperationDescriptor, GetTensorDescriptorsWithoutOptionalTensors)
{
    setAllAttributesExcept(
        {HIPDNN_ATTR_OPERATION_RMSNORM_BIAS_EXT, HIPDNN_ATTR_OPERATION_RMSNORM_INV_RMS_EXT});
    getDescriptor()->finalize();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 4);
    ASSERT_EQ(tensors[0]->getData().uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_RMSNORM_TENSOR_Y_UID);
}

TEST_F(TestRMSNormOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::RMSNormAttributes);

    auto* attrs = node->attributes.AsRMSNormAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_RMSNORM_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_RMSNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->epsilon_tensor_uid, K_RMSNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_RMSNORM_TENSOR_Y_UID);
    ASSERT_TRUE(attrs->bias_tensor_uid.has_value());
    ASSERT_EQ(*attrs->bias_tensor_uid, K_RMSNORM_TENSOR_BIAS_UID);
    ASSERT_TRUE(attrs->inv_rms_tensor_uid.has_value());
    ASSERT_EQ(*attrs->inv_rms_tensor_uid, K_RMSNORM_TENSOR_INV_RMS_UID);
    ASSERT_EQ(attrs->forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestRMSNormOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_RMSNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestRMSNormOperationDescriptor, GetTensorDescriptorsOrderIsCorrect)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getScaleDesc());
    EXPECT_EQ(tensors[2], desc->getEpsilonDesc());
    EXPECT_EQ(tensors[3], desc->getYDesc());
    EXPECT_EQ(tensors[4], desc->getBiasDesc());
    EXPECT_EQ(tensors[5], desc->getInvRmsDesc());
}

TEST_F(TestRMSNormOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_RMSNORM_TENSOR_X_UID);
}

TEST_F(TestRMSNormOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
