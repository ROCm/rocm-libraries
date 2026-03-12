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
#include <hipdnn_data_sdk/data_objects/batchnorm_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestBatchnormOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BatchnormOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BatchnormOperationDescriptor>();
    }

    void setRequiredTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_biasDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_epsilonDesc);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc);
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

    void setComputeType() const
    {
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    }

    void setRequiredAttributes() const
    {
        setRequiredTensors();
        setComputeType();
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        setOptionalMeanInvVariance();
        getDescriptor()->finalize();
    }

    void makeFinalizedMinimal() const
    {
        setRequiredAttributes();
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
    setRequiredAttributes();
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
    setRequiredAttributes();
    setOptionalMeanInvVariance();
    setRunningStats();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// =============================================================================
// Finalize Failure Tests - Required Tensors (parameterized)
// =============================================================================

struct FinalizeFailTestParam
{
    std::string name;
    std::vector<hipdnnBackendAttributeName_t> attrsToSkip;
};

class TestBatchnormFinalizeFailMissingRequired
    : public TestBatchnormOperationDescriptor,
      public ::testing::WithParamInterface<FinalizeFailTestParam>
{
};

TEST_P(TestBatchnormFinalizeFailMissingRequired, FinalizeFailsMissingRequiredTensor)
{
    const auto& param = GetParam();
    auto desc = getDescriptor();

    std::unordered_map<hipdnnBackendAttributeName_t, const void*> tensorMap = {
        {HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, _xDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, _scaleDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, _biasDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, _epsilonDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, _yDesc.get()},
    };

    for(auto skip : param.attrsToSkip)
    {
        tensorMap.erase(skip);
    }

    for(const auto& [attrName, tensorDesc] : tensorMap)
    {
        desc->setAttribute(attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &tensorDesc);
    }

    setComputeType();
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(
    MissingRequired,
    TestBatchnormFinalizeFailMissingRequired,
    ::testing::Values(
        FinalizeFailTestParam{"MissingX", {HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT}},
        FinalizeFailTestParam{"MissingScale", {HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT}},
        FinalizeFailTestParam{"MissingBias", {HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT}},
        FinalizeFailTestParam{"MissingEpsilon", {HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT}},
        FinalizeFailTestParam{"MissingY", {HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT}}),
    [](const ::testing::TestParamInfo<FinalizeFailTestParam>& info) { return info.param.name; });

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setRequiredTensors();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// Finalize Failure Tests - Optional Tensor Pairing (mean + inv_variance)
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithOnlyMean)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_meanDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithOnlyInvVariance)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeSucceedsWithoutMeanAndInvVariance)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_EQ(getDescriptor()->getMeanDesc(), nullptr);
    ASSERT_EQ(getDescriptor()->getInvVarianceDesc(), nullptr);
}

// =============================================================================
// Finalize Failure Tests - Running Stats All-or-None
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithPartialRunningStats_OnlyPrevMean)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_prevRunningMeanDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormOperationDescriptor, FinalizeFailsWithPartialRunningStats_MissingMomentum)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
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
    setRequiredAttributes();
    setRunningStats();
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// =============================================================================
// SetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, SetComputeDataType)
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

TEST_F(TestBatchnormOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalizedMinimal();
    ASSERT_THROW_HIPDNN_STATUS(
        getDescriptor()->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
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

struct GetTensorTestParam
{
    std::string name;
    hipdnnBackendAttributeName_t attrName;
    int64_t expectedUid;
};

class TestBatchnormGetTensor : public TestBatchnormOperationDescriptor,
                               public ::testing::WithParamInterface<GetTensorTestParam>
{
};

TEST_P(TestBatchnormGetTensor, GetAttributeTensorDescriptor)
{
    const auto& param = GetParam();
    setRequiredAttributes();
    setOptionalMeanInvVariance();
    setRunningStats();
    getDescriptor()->finalize();

    HipdnnBackendDescriptor* retrieved = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(getDescriptor()->getAttribute(
        param.attrName, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &elementCount, &retrieved));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrieved, nullptr);

    // Unpack and verify UID
    auto tensorDesc = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        &retrieved, HIPDNN_STATUS_BAD_PARAM, "test unpack");
    ASSERT_EQ(tensorDesc->getData().uid, param.expectedUid);
}

INSTANTIATE_TEST_SUITE_P(
    AllTensors,
    TestBatchnormGetTensor,
    ::testing::Values(
        GetTensorTestParam{"X", HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, K_BATCHNORM_TENSOR_X_UID},
        GetTensorTestParam{
            "Scale", HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, K_BATCHNORM_TENSOR_SCALE_UID},
        GetTensorTestParam{
            "Bias", HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, K_BATCHNORM_TENSOR_BIAS_UID},
        GetTensorTestParam{
            "Epsilon", HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, K_BATCHNORM_TENSOR_EPSILON_UID},
        GetTensorTestParam{"Y", HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, K_BATCHNORM_TENSOR_Y_UID},
        GetTensorTestParam{
            "Mean", HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, K_BATCHNORM_TENSOR_MEAN_UID},
        GetTensorTestParam{"InvVariance",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                           K_BATCHNORM_TENSOR_INV_VARIANCE_UID},
        GetTensorTestParam{"PrevRunMean",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                           K_BATCHNORM_TENSOR_PREV_RUNNING_MEAN_UID},
        GetTensorTestParam{"PrevRunVar",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                           K_BATCHNORM_TENSOR_PREV_RUNNING_VARIANCE_UID},
        GetTensorTestParam{"Momentum",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                           K_BATCHNORM_TENSOR_MOMENTUM_UID},
        GetTensorTestParam{"NextRunMean",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                           K_BATCHNORM_TENSOR_NEXT_RUNNING_MEAN_UID},
        GetTensorTestParam{"NextRunVar",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                           K_BATCHNORM_TENSOR_NEXT_RUNNING_VARIANCE_UID}),
    [](const ::testing::TestParamInfo<GetTensorTestParam>& info) { return info.param.name; });

// =============================================================================
// GetAttribute Tests - Compute Data Type
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetAttributeComputeType)
{
    setRequiredAttributes();
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
// GetAttribute Query Mode (elementCount=0) Tests (parameterized)
// =============================================================================

struct QueryModeTestParam
{
    std::string name;
    hipdnnBackendAttributeName_t attrName;
    hipdnnBackendAttributeType_t attrType;
};

class TestBatchnormQueryMode : public TestBatchnormOperationDescriptor,
                               public ::testing::WithParamInterface<QueryModeTestParam>
{
};

TEST_P(TestBatchnormQueryMode, QueryModeReturnsCorrectCount)
{
    const auto& param = GetParam();
    setRequiredAttributes();
    setOptionalMeanInvVariance();
    setRunningStats();
    getDescriptor()->finalize();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        getDescriptor()->getAttribute(param.attrName, param.attrType, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    Tensors,
    TestBatchnormQueryMode,
    ::testing::Values(
        QueryModeTestParam{
            "X", HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{
            "Scale", HIPDNN_ATTR_OPERATION_BATCHNORM_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{
            "Bias", HIPDNN_ATTR_OPERATION_BATCHNORM_BIAS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{
            "Epsilon", HIPDNN_ATTR_OPERATION_BATCHNORM_EPSILON_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{
            "Y", HIPDNN_ATTR_OPERATION_BATCHNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{
            "Mean", HIPDNN_ATTR_OPERATION_BATCHNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"InvVar",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"CompType", HIPDNN_ATTR_BATCHNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE},
        QueryModeTestParam{"PrevRunMean",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"PrevRunVar",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_PREV_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"Momentum",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_MOMENTUM_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"NextRunMean",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR},
        QueryModeTestParam{"NextRunVar",
                           HIPDNN_ATTR_OPERATION_BATCHNORM_NEXT_RUNNING_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR}),
    [](const ::testing::TestParamInfo<QueryModeTestParam>& info) { return info.param.name; });

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    setRequiredAttributes();
    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_X_EXT,
                                                             HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                             1,
                                                             nullptr,
                                                             &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
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
    setRequiredAttributes();
    setOptionalMeanInvVariance();
    setRunningStats();
    getDescriptor()->finalize();

    auto tensors = getDescriptor()->getTensorDescriptors();
    // 5 required + 2 (mean, inv_var) + 5 (running stats)
    ASSERT_EQ(tensors.size(), 12u);
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

TEST_F(TestBatchnormOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();
    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors[0]->getData().uid, K_BATCHNORM_TENSOR_X_UID);
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
                                       descs.data()));

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
                       descs.data());
    setRequiredAttributes();
    desc->finalize();

    std::array<HipdnnBackendDescriptor*, 2> retrieved = {};
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       2,
                                       &elementCount,
                                       retrieved.data()));

    ASSERT_EQ(elementCount, 2);
    ASSERT_NE(retrieved[0], nullptr);
    ASSERT_NE(retrieved[1], nullptr);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBatchnormOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    setOptionalMeanInvVariance();
    auto desc = getDescriptor();
    std::string str = desc->toString();
    ASSERT_NE(str.find("BatchnormOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=500"), std::string::npos);
    ASSERT_NE(str.find("scale_uid=501"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}
