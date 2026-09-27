// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/MoeGroupedMatmulOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/moe_grouped_matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/MoeGroupedMatmulConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <optional>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestMoeGroupedMatmulOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<MoeGroupedMatmulOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<MoeGroupedMatmulOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_tokenDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_weightDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_firstTokenOffsetDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_tokenIndexDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_tokenKsDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_outputDesc);
    }
    void setModeRuleAttributes(hipdnnMoeGroupedMatmulMode_t mode,
                               std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_tokenDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_weightDesc);
        }
        if(!excluded
           || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_firstTokenOffsetDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_outputDesc);
        }
        const auto computeType = HIPDNN_DATA_FLOAT;
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                           HIPDNN_TYPE_DATA_TYPE,
                           1,
                           &computeType);

        switch(mode)
        {
        case HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE:
        {
            const auto topK = static_cast<int32_t>(0);
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
            break;
        }
        case HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER:
        {
            if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC)
            {
                desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &_tokenIndexDesc);
            }
            const auto topK = static_cast<int32_t>(0);
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
            break;
        }
        case HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER:
        {
            if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC)
            {
                desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &_tokenIndexDesc);
            }
            if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC)
            {
                desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                   HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                   1,
                                   &_tokenKsDesc);
            }
            const auto topK = static_cast<int32_t>(2);
            desc->setAttribute(
                HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
            break;
        }
        default:
            break;
        }
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                           HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                           1,
                           &mode);
    }

    void setMoeGroupedMatmulParams() const
    {
        auto desc = getDescriptor();

        auto topK = static_cast<int32_t>(2);
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setMoeGroupedMatmulParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                      HIPDNN_TYPE_DATA_TYPE,
                                      1,
                                      &computeType);
        auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
        getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                      HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                      1,
                                      &mode);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _weightDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _firstTokenOffsetDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenIndexDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _tokenKsDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _outputDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<MoeGroupedMatmulOperationDescriptor>();
        _tokenDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID,
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_DIMS),
                                           toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_STRIDES),
                                           HIPDNN_DATA_FLOAT);
        _weightDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
        _firstTokenOffsetDesc
            = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID,
                                    toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_DIMS),
                                    toVec(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_STRIDES),
                                    HIPDNN_DATA_INT32);
        _tokenIndexDesc
            = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID,
                                    toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_DIMS),
                                    toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_STRIDES),
                                    HIPDNN_DATA_INT32);
        _tokenKsDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID,
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_DIMS),
                                             toVec(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_STRIDES),
                                             HIPDNN_DATA_INT32);
        _outputDesc = createFinalizedTensor(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID,
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_DIMS),
                                            toVec(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _tokenDesc.reset();
        _weightDesc.reset();
        _firstTokenOffsetDesc.reset();
        _tokenIndexDesc.reset();
        _tokenKsDesc.reset();
        _outputDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_MOE_GROUPED_MATMUL_DESCRIPTOR);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

class TestMoeGroupedMatmulOperationDescriptorFinalizeFailsWithout
    : public TestMoeGroupedMatmulOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestMoeGroupedMatmulOperationDescriptorFinalizeFailsWithout, RequiredTensor)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER, GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(
    RequiredTensors,
    TestMoeGroupedMatmulOperationDescriptorFinalizeFailsWithout,
    ::testing::Values(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                      HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC));

// ============================================================================
// Mode-rule descriptor contract
// ============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeSucceedsForNONEMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE);
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsTokenIndexInNONEMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE);
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_tokenIndexDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsTokenKsInNONEMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE);
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_tokenKsDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsNoncanonicalTopKInNONEMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_NONE);
    const auto topK = static_cast<int32_t>(1);
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeSucceedsForGATHERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER);
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsMissingTokenIndexInGATHERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER,
                          HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsTokenKsInGATHERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER);
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_tokenKsDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsNoncanonicalTopKInGATHERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_GATHER);
    const auto topK = static_cast<int32_t>(1);
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeSucceedsForSCATTERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER);
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsMissingTokenIndexInSCATTERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER,
                          HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsMissingTokenKsInSCATTERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER,
                          HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsBelowMinimumTopKInSCATTERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER);
    const auto topK = static_cast<int32_t>(0);
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeRejectsAboveMaximumTopKInSCATTERMode)
{
    setModeRuleAttributes(HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER);
    const auto topK = static_cast<int32_t>(3);
    getDescriptor()->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setMoeGroupedMatmulParams();
    auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
    getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                  HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                  1,
                                  &mode);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorToken)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_tokenDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    ASSERT_NE(desc->getTokenDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorWeight)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_weightDesc));

    ASSERT_EQ(desc->getData().weight_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorFirstTokenOffset)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_firstTokenOffsetDesc));

    ASSERT_EQ(desc->getData().first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorTokenIndex)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_tokenIndexDesc));

    ASSERT_EQ(desc->getData().token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorTokenKs)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_tokenKsDesc));

    ASSERT_EQ(desc->getData().token_ks_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorDescriptorOutput)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_outputDesc));

    ASSERT_EQ(desc->getData().output_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    ASSERT_NE(desc->getOutputDesc(), nullptr);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_unfinalizedTensor),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC, HIPDNN_TYPE_INT64, 1, &_tokenDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           2,
                           &_tokenDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetMoeGroupedMatmulMode)
{
    auto desc = getDescriptor();
    auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                       HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                       1,
                                       &mode));

    ASSERT_EQ(desc->getData().mode, MoeGroupedMatmulMode::SCATTER);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetMoeGroupedMatmulModeWrongElementCount)
{
    auto desc = getDescriptor();
    auto mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                                  HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                                  2,
                                                  &mode),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetTopK)
{
    auto desc = getDescriptor();
    auto topK = static_cast<int32_t>(2);

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topK));
    EXPECT_EQ(desc->getData().top_k, topK);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                           HIPDNN_TYPE_DATA_TYPE,
                           2,
                           &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_tokenDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedToken = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedToken)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedToken, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardToken(retrievedToken);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeMoeGroupedMatmulParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // mode
    hipdnnMoeGroupedMatmulMode_t mode = HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER;
    int64_t modeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                       HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                       1,
                                       &modeCount,
                                       &mode));
    ASSERT_EQ(modeCount, 1);
    EXPECT_EQ(mode, HIPDNN_MOE_GROUPED_MATMUL_MODE_SCATTER);

    // top k
    int32_t topK = 0;
    int64_t topKCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOP_K, HIPDNN_TYPE_INT32, 1, &topKCount, &topK));
    ASSERT_EQ(topKCount, 1);
    EXPECT_EQ(topK, static_cast<int32_t>(2));
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
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

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           &dummy),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorTokenQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorWeightQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_WEIGHT_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorFirstTokenOffsetQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_FIRST_TOKEN_OFFSET_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           &elementCount,
                           nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorTokenIndexQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_INDEX_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorTokenKsQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_KS_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorOutputQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_OUTPUT_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeMoeGroupedMatmulModeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                       HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_TOKEN_DESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor,
       GetAttributeMoeGroupedMatmulModeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MODE,
                                                  HIPDNN_TYPE_MOE_GROUPED_MATMUL_MODE,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getTokenDesc(), nullptr);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
    ASSERT_NE(desc->getFirstTokenOffsetDesc(), nullptr);
    ASSERT_NE(desc->getTokenIndexDesc(), nullptr);
    ASSERT_NE(desc->getTokenKsDesc(), nullptr);
    ASSERT_NE(desc->getOutputDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getTokenDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    ASSERT_EQ(desc->getWeightDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    ASSERT_EQ(desc->getFirstTokenOffsetDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(desc->getTokenIndexDesc()->getData().uid,
              K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(desc->getTokenKsDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    ASSERT_EQ(desc->getOutputDesc()->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("MoeGroupedMatmulOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("token_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID)),
              std::string::npos);
    ASSERT_NE(str.find("weight_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID)),
              std::string::npos);
    ASSERT_NE(str.find("first_token_offset_uid="
                       + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("token_index_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID)),
        std::string::npos);
    ASSERT_NE(str.find("token_ks_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("output_uid=" + std::to_string(K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::MoeGroupedMatmulAttributes);

    auto* attrs = node->attributes.AsMoeGroupedMatmulAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->token_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
    ASSERT_EQ(attrs->weight_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_WEIGHT_UID);
    ASSERT_EQ(attrs->first_token_offset_tensor_uid,
              K_MOE_GROUPED_MATMUL_TENSOR_FIRST_TOKEN_OFFSET_UID);
    ASSERT_EQ(attrs->token_index_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_INDEX_UID);
    ASSERT_EQ(attrs->token_ks_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_KS_UID);
    ASSERT_EQ(attrs->output_tensor_uid, K_MOE_GROUPED_MATMUL_TENSOR_OUTPUT_UID);
    EXPECT_EQ(attrs->mode, MoeGroupedMatmulMode::SCATTER);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor,
       GetTensorDescriptorsOrderIsTokenWeightFirstTokenOffsetTokenIndexTokenKsOutput)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    // Verify ordering: [TOKEN_DESC, WEIGHT_DESC, FIRST_TOKEN_OFFSET_DESC, TOKEN_INDEX_DESC, TOKEN_KS_DESC, OUTPUT_DESC] matches UIDs [1900, 1901, 1902, 1903, 1904, 1905]
    EXPECT_EQ(tensors[0], desc->getTokenDesc());
    EXPECT_EQ(tensors[1], desc->getWeightDesc());
    EXPECT_EQ(tensors[2], desc->getFirstTokenOffsetDesc());
    EXPECT_EQ(tensors[3], desc->getTokenIndexDesc());
    EXPECT_EQ(tensors[4], desc->getTokenKsDesc());
    EXPECT_EQ(tensors[5], desc->getOutputDesc());
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_MOE_GROUPED_MATMUL_TENSOR_TOKEN_UID);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _tokenDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestMoeGroupedMatmulOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_moegroupedmatmul_op";

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
    EXPECT_STREQ(buffer.data(), "test_moegroupedmatmul_op");
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_MOE_GROUPED_MATMUL_EXT);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestMoeGroupedMatmulOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_moegroupedmatmul";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_MOE_GROUPED_MATMUL_MATH_PREC, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_moegroupedmatmul");
}
