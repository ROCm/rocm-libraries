// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

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

class TestBatchnormOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BatchnormOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BatchnormOperationDescriptor>();
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
        setIf(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, _biasDesc);
        setIf(HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, _epsilonDesc);
        setIf(HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, _yDesc);
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
    }

    void setOptionalMeanInvVariance() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_meanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invVarianceDesc);
    }

    void setRunningStats() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_prevRunningMeanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_prevRunningVarianceDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_momentumDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_nextRunningMeanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_nextRunningVarianceDesc);
    }

    // Sets all required attrs + mean + inv_variance, then finalizes
    void makeFinalized() const
    {
        setAllAttributesExcept();
        setOptionalMeanInvVariance();
        getDescriptor()->finalize();
    }

    // Sets all required attrs only, then finalizes
    void makeFinalizedMinimal() const
    {
        setAllAttributesExcept();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _biasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _epsilonDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _meanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _prevRunningMeanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _prevRunningVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _momentumDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _nextRunningMeanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _nextRunningVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _peerStatsDesc0 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _peerStatsDesc1 = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BatchnormOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_X_UID,
                                       toVec(K_BATCHNORM_TENSOR_X_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_SCALE_UID,
                                           toVec(K_BATCHNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_BATCHNORM_TENSOR_SCALE_STRIDES));
        _biasDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_BIAS_UID,
                                          toVec(K_BATCHNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_BIAS_STRIDES));
        _epsilonDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_EPSILON_UID,
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_BATCHNORM_TENSOR_EPSILON_STRIDES));
        _yDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_Y_UID,
                                       toVec(K_BATCHNORM_TENSOR_Y_DIMS),
                                       toVec(K_BATCHNORM_TENSOR_Y_STRIDES));
        _meanDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_MEAN_UID,
                                          toVec(K_BATCHNORM_TENSOR_MEAN_DIMS),
                                          toVec(K_BATCHNORM_TENSOR_MEAN_STRIDES));
        _invVarianceDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_DIMS),
                                                 toVec(K_BATCHNORM_TENSOR_INV_VARIANCE_STRIDES));
        _prevRunningMeanDesc
            = createFinalizedTensor(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID,
                                    toVec(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_DIMS),
                                    toVec(K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_STRIDES));
        _prevRunningVarianceDesc
            = createFinalizedTensor(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID,
                                    toVec(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_DIMS),
                                    toVec(K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_STRIDES));
        _momentumDesc = createFinalizedTensor(K_BATCHNORM_TENSOR_MOMENTUM_UID,
                                              toVec(K_BATCHNORM_TENSOR_MOMENTUM_DIMS),
                                              toVec(K_BATCHNORM_TENSOR_MOMENTUM_STRIDES));
        _nextRunningMeanDesc
            = createFinalizedTensor(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID,
                                    toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_DIMS),
                                    toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_STRIDES));
        _nextRunningVarianceDesc
            = createFinalizedTensor(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID,
                                    toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_DIMS),
                                    toVec(K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_STRIDES));
        _peerStatsDesc0 = createFinalizedTensor(K_BATCHNORM_TENSOR_PEER_STAT_0_UID,
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS),
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
        _peerStatsDesc1 = createFinalizedTensor(K_BATCHNORM_TENSOR_PEER_STAT_1_UID,
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_DIMS),
                                                toVec(K_BATCHNORM_TENSOR_PEER_STAT_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _biasDesc.reset();
        _epsilonDesc.reset();
        _yDesc.reset();
        _meanDesc.reset();
        _invVarianceDesc.reset();
        _prevRunningMeanDesc.reset();
        _prevRunningVarianceDesc.reset();
        _momentumDesc.reset();
        _nextRunningMeanDesc.reset();
        _nextRunningVarianceDesc.reset();
        _peerStatsDesc0.reset();
        _peerStatsDesc1.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_DESCRIPTOR_EXT);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestBatchnormOperationDescriptor, DoubleFinalizeSucceeds)
{
    makeFinalizedMinimal();
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeSucceedsWithAllOptionalTensors)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    setRunningStats();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// =============================================================================
// Parameterized Finalize-fails-without Tests
// =============================================================================

class TestBatchnormOperationDescriptorFinalizeFailsWithout
    : public TestBatchnormOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestBatchnormOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestBatchnormOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT,
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT,
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                                           HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT,
                                           HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT));

// =============================================================================
// Finalize Failure Tests - Optional Tensor Pairing (mean + inv_variance)
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithOnlyMean)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_meanDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithOnlyInvVariance)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeSucceedsWithoutMeanAndInvVariance)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_EQ(getDescriptor()->getMeanDesc(), nullptr);
    ASSERT_EQ(getDescriptor()->getInvVarianceDesc(), nullptr);
}

// =============================================================================
// Finalize Failure Tests - Running Stats All-or-None
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithPartialRunningStatsOnlyPrevMean)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_prevRunningMeanDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithPartialRunningStatsMissingMomentum)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_prevRunningMeanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_prevRunningVarianceDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_nextRunningMeanDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_nextRunningVarianceDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeSucceedsWithAllRunningStats)
{
    setAllAttributesExcept();
    setRunningStats();
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_biasDesc));

    ASSERT_EQ(desc->getData().bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorEpsilon)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_epsilonDesc));

    ASSERT_EQ(desc->getData().epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_meanDesc));

    ASSERT_TRUE(desc->getData().mean_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().mean_tensor_uid.value(), K_BATCHNORM_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorInvVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invVarianceDesc));

    ASSERT_TRUE(desc->getData().inv_variance_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().inv_variance_tensor_uid.value(), K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorPrevRunningMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_prevRunningMeanDesc));

    ASSERT_TRUE(desc->getData().prev_running_mean_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().prev_running_mean_tensor_uid.value(),
              K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorPrevRunningVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_prevRunningVarianceDesc));

    ASSERT_TRUE(desc->getData().prev_running_variance_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().prev_running_variance_tensor_uid.value(),
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorMomentum)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_momentumDesc));

    ASSERT_TRUE(desc->getData().momentum_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().momentum_tensor_uid.value(), K_BATCHNORM_TENSOR_MOMENTUM_UID);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorNextRunningMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_nextRunningMeanDesc));

    ASSERT_TRUE(desc->getData().next_running_mean_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().next_running_mean_tensor_uid.value(),
              K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorDescriptorNextRunningVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_nextRunningVarianceDesc));

    ASSERT_TRUE(desc->getData().next_running_variance_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().next_running_variance_tensor_uid.value(),
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
}

TEST_F(TestBatchnormOperationDescriptor, SetMathPrec)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));
    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorFailsNotFinalized)
{
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorFailsWrongType)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, SetTensorFailsNullPointer)
{
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBatchnormOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalizedMinimal();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormOperationDescriptor, SetAttributeUnsupported)
{
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - Tensor Descriptors (parameterized)
// =============================================================================

struct TensorAttrCase
{
    hipdnnBackendAttributeName_t attr;
    const char* name;
    int64_t expectedUid;
};

class TestBatchnormOperationDescriptorGetTensor
    : public TestBatchnormOperationDescriptor,
      public ::testing::WithParamInterface<TensorAttrCase>
{
};

TEST_P(TestBatchnormOperationDescriptorGetTensor, GetAttributeTensorDescriptorReturnsCorrectTensor)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    setRunningStats();
    getDescriptor()->finalize();

    auto desc = getDescriptor();
    const auto& tc = GetParam();

    // getAttribute packs the stored TensorDescriptor into a fresh HipdnnBackendDescriptor*.
    // Ownership is transferred to the caller - delete after use.
    HipdnnBackendDescriptor* retrieved = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        tc.attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &elementCount, static_cast<void*>(&retrieved)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrieved, nullptr);

    auto tensorImpl = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrieved, HIPDNN_STATUS_INTERNAL_ERROR, "Failed to unpack retrieved tensor descriptor");
    delete retrieved;

    ASSERT_NE(tensorImpl, nullptr);
    EXPECT_EQ(tensorImpl->getData().uid, tc.expectedUid);
}

TEST_P(TestBatchnormOperationDescriptorGetTensor, QueryModeReturnsOne)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    setRunningStats();
    getDescriptor()->finalize();

    auto desc = getDescriptor();
    const auto& tc = GetParam();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(tc.attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllTensors,
    TestBatchnormOperationDescriptorGetTensor,
    ::testing::Values(
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, "X", K_BATCHNORM_TENSOR_X_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, "Scale", K_BATCHNORM_TENSOR_SCALE_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, "Bias", K_BATCHNORM_TENSOR_BIAS_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, "Epsilon", K_BATCHNORM_TENSOR_EPSILON_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, "Y", K_BATCHNORM_TENSOR_Y_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, "Mean", K_BATCHNORM_TENSOR_MEAN_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                       "InvVariance",
                       K_BATCHNORM_TENSOR_INV_VARIANCE_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                       "PrevRunMean",
                       K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                       "PrevRunVar",
                       K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                       "Momentum",
                       K_BATCHNORM_TENSOR_MOMENTUM_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                       "NextRunMean",
                       K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                       "NextRunVar",
                       K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID}),
    [](const ::testing::TestParamInfo<TensorAttrCase>& info) { return info.param.name; });

// =============================================================================
// GetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetAttributeComputeType)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Query Mode (elementCount=0) Tests
// Tensor query mode is covered by TestBatchnormOperationDescriptorGetTensor::QueryModeReturnsOne
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetAttributeMathPrecQueryReturnsOne)
{
    makeFinalized();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    setAllAttributesExcept();
    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBatchnormOperationDescriptor, GetAttributeUnsupported)
{
    makeFinalizedMinimal();
    int64_t dummy = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->getAttribute(
            HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);

    ASSERT_EQ(desc->getXDesc()->getData().uid, K_BATCHNORM_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_BATCHNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getBiasDesc()->getData().uid, K_BATCHNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(desc->getEpsilonDesc()->getData().uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_BATCHNORM_TENSOR_Y_UID);
    ASSERT_EQ(desc->getMeanDesc()->getData().uid, K_BATCHNORM_TENSOR_MEAN_UID);
    ASSERT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizePreservesTensorReferencesWithRunningStats)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    setRunningStats();
    auto desc = getDescriptor();
    desc->finalize();

    ASSERT_NE(desc->getPrevRunningMeanDesc(), nullptr);
    ASSERT_NE(desc->getPrevRunningVarianceDesc(), nullptr);
    ASSERT_NE(desc->getMomentumDesc(), nullptr);
    ASSERT_NE(desc->getNextRunningMeanDesc(), nullptr);
    ASSERT_NE(desc->getNextRunningVarianceDesc(), nullptr);

    ASSERT_EQ(desc->getPrevRunningMeanDesc()->getData().uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID);
    ASSERT_EQ(desc->getPrevRunningVarianceDesc()->getData().uid,
              K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID);
    ASSERT_EQ(desc->getMomentumDesc()->getData().uid, K_BATCHNORM_TENSOR_MOMENTUM_UID);
    ASSERT_EQ(desc->getNextRunningMeanDesc()->getData().uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID);
    ASSERT_EQ(desc->getNextRunningVarianceDesc()->getData().uid,
              K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetTensorDescriptorsRequiredOnly)
{
    makeFinalizedMinimal();
    auto tensors = getDescriptor()->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 5u);
    EXPECT_EQ(tensors[0].get(), getDescriptor()->getXDesc().get());
    EXPECT_EQ(tensors[1].get(), getDescriptor()->getScaleDesc().get());
    EXPECT_EQ(tensors[2].get(), getDescriptor()->getBiasDesc().get());
    EXPECT_EQ(tensors[3].get(), getDescriptor()->getEpsilonDesc().get());
    EXPECT_EQ(tensors[4].get(), getDescriptor()->getYDesc().get());
}

TEST_F(TestBatchnormOperationDescriptor, GetTensorDescriptorsWithOptionals)
{
    makeFinalized();
    auto tensors = getDescriptor()->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7u);
    EXPECT_EQ(tensors[5].get(), getDescriptor()->getMeanDesc().get());
    EXPECT_EQ(tensors[6].get(), getDescriptor()->getInvVarianceDesc().get());
}

TEST_F(TestBatchnormOperationDescriptor, GetTensorDescriptorsWithAllOptionals)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    setRunningStats();
    auto desc = getDescriptor();
    desc->finalize();

    auto tensors = desc->getTensorDescriptors();
    // 5 required + 2 (mean, inv_var) + 5 (running stats)
    ASSERT_EQ(tensors.size(), 12u);
    EXPECT_EQ(tensors[0].get(), desc->getXDesc().get());
    EXPECT_EQ(tensors[1].get(), desc->getScaleDesc().get());
    EXPECT_EQ(tensors[2].get(), desc->getBiasDesc().get());
    EXPECT_EQ(tensors[3].get(), desc->getEpsilonDesc().get());
    EXPECT_EQ(tensors[4].get(), desc->getYDesc().get());
    EXPECT_EQ(tensors[5].get(), desc->getMeanDesc().get());
    EXPECT_EQ(tensors[6].get(), desc->getInvVarianceDesc().get());
    EXPECT_EQ(tensors[7].get(), desc->getPrevRunningMeanDesc().get());
    EXPECT_EQ(tensors[8].get(), desc->getPrevRunningVarianceDesc().get());
    EXPECT_EQ(tensors[9].get(), desc->getMomentumDesc().get());
    EXPECT_EQ(tensors[10].get(), desc->getNextRunningMeanDesc().get());
    EXPECT_EQ(tensors[11].get(), desc->getNextRunningVarianceDesc().get());
}

TEST_F(TestBatchnormOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    makeFinalized();
    auto node = getDescriptor()->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BatchnormAttributes);

    auto* attrs = node->attributes.AsBatchnormAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_BATCHNORM_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_BATCHNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->bias_tensor_uid, K_BATCHNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(attrs->epsilon_tensor_uid, K_BATCHNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_BATCHNORM_TENSOR_Y_UID);
    ASSERT_TRUE(attrs->mean_tensor_uid.has_value());
    ASSERT_EQ(attrs->mean_tensor_uid.value(), K_BATCHNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(attrs->inv_variance_tensor_uid.has_value());
    ASSERT_EQ(attrs->inv_variance_tensor_uid.value(), K_BATCHNORM_TENSOR_INV_VARIANCE_UID);
}

TEST_F(TestBatchnormOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestBatchnormOperationDescriptor, BuildNodeWithoutOptionalTensors)
{
    setAllAttributesExcept();
    getDescriptor()->finalize();

    auto node = getDescriptor()->buildNode();
    ASSERT_NE(node, nullptr);

    auto* attrs = node->attributes.AsBatchnormAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_FALSE(attrs->mean_tensor_uid.has_value());
    ASSERT_FALSE(attrs->inv_variance_tensor_uid.has_value());
    ASSERT_FALSE(attrs->prev_running_mean_tensor_uid.has_value());
    ASSERT_FALSE(attrs->prev_running_variance_tensor_uid.has_value());
    ASSERT_FALSE(attrs->momentum_tensor_uid.has_value());
    ASSERT_FALSE(attrs->next_running_mean_tensor_uid.has_value());
    ASSERT_FALSE(attrs->next_running_variance_tensor_uid.has_value());
}

TEST_F(TestBatchnormOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();
    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors[0]->getData().uid, K_BATCHNORM_TENSOR_X_UID);
}

TEST_F(TestBatchnormOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Tensor Array Tests - PeerStats
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, SetPeerStatsTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> descs = {_peerStatsDesc0.get(), _peerStatsDesc1.get()};
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       static_cast<const void*>(descs.data())));

    auto& data = desc->getData();
    ASSERT_EQ(data.peer_stats_tensor_uid.size(), 2u);
    EXPECT_EQ(data.peer_stats_tensor_uid[0], K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(data.peer_stats_tensor_uid[1], K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
}

TEST_F(TestBatchnormOperationDescriptor, GetPeerStatsTensorArray)
{
    auto desc = getDescriptor();
    std::array<HipdnnBackendDescriptor*, 2> descs = {_peerStatsDesc0.get(), _peerStatsDesc1.get()};
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       2,
                       static_cast<const void*>(descs.data()));
    setAllAttributesExcept();
    desc->finalize();

    std::array<HipdnnBackendDescriptor*, 2> retrieved = {};
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &elementCount,
                                       static_cast<void*>(retrieved.data())));

    ASSERT_EQ(elementCount, 2);
    ASSERT_NE(retrieved[0], nullptr);
    ASSERT_NE(retrieved[1], nullptr);

    auto peerDesc0 = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrieved[0], HIPDNN_STATUS_INTERNAL_ERROR, "Failed to unpack peer stats 0");
    delete retrieved[0];
    auto peerDesc1 = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrieved[1], HIPDNN_STATUS_INTERNAL_ERROR, "Failed to unpack peer stats 1");
    delete retrieved[1];

    ASSERT_NE(peerDesc0, nullptr);
    ASSERT_NE(peerDesc1, nullptr);
    EXPECT_EQ(peerDesc0->getData().uid, K_BATCHNORM_TENSOR_PEER_STAT_0_UID);
    EXPECT_EQ(peerDesc1->getData().uid, K_BATCHNORM_TENSOR_PEER_STAT_1_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    setOptionalMeanInvVariance();
    auto desc = getDescriptor();
    const std::string str = desc->toString();
    ASSERT_NE(str.find("BatchnormOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_BATCHNORM_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_BATCHNORM_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

} // namespace
