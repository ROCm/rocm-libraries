// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnOperationType.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/CausalConv1dOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/causal_conv1d_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/CausalConv1dConstants.hpp>
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

class TestCausalConv1dOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<CausalConv1dOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<CausalConv1dOperationDescriptor>();
    }

    void setTensors(std::optional<hipdnnBackendAttributeName_t> excluded = std::nullopt) const
    {
        auto desc = getDescriptor();
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_xDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_weightDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_cuSeqlensDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_yDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_biasDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_initialStateDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_stateIndicesDesc);
        }
        if(!excluded || *excluded != HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT)
        {
            desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                               HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                               1,
                               &_finalStateDesc);
        }
    }

    void setRequiredAttributes(std::optional<hipdnnBackendAttributeName_t> excluded
                               = std::nullopt) const
    {
        setTensors(excluded);
        if(!excluded || *excluded != HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT)
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            getDescriptor()->setAttribute(
                HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
        auto activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT;
        getDescriptor()->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                      HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                      1,
                                      &activation);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _weightDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _cuSeqlensDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _biasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _initialStateDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _stateIndicesDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _finalStateDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<CausalConv1dOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_X_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_X_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _weightDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID,
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_DIMS),
                                            toVec(K_CAUSAL_CONV1D_TENSOR_WEIGHT_STRIDES),
                                            HIPDNN_DATA_FLOAT);
        _cuSeqlensDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID,
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_DIMS),
                                               toVec(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_STRIDES),
                                               HIPDNN_DATA_INT32);
        _yDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_Y_UID,
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_DIMS),
                                       toVec(K_CAUSAL_CONV1D_TENSOR_Y_STRIDES),
                                       HIPDNN_DATA_FLOAT);
        _biasDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_BIAS_UID,
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_DIMS),
                                          toVec(K_CAUSAL_CONV1D_TENSOR_BIAS_STRIDES),
                                          HIPDNN_DATA_FLOAT);
        _initialStateDesc
            = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID,
                                    toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_DIMS),
                                    toVec(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_STRIDES),
                                    HIPDNN_DATA_FLOAT);
        _stateIndicesDesc
            = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID,
                                    toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_DIMS),
                                    toVec(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_STRIDES),
                                    HIPDNN_DATA_INT32);
        _finalStateDesc = createFinalizedTensor(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID,
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_DIMS),
                                                toVec(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_STRIDES),
                                                HIPDNN_DATA_FLOAT);
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _weightDesc.reset();
        _cuSeqlensDesc.reset();
        _yDesc.reset();
        _biasDesc.reset();
        _initialStateDesc.reset();
        _stateIndicesDesc.reset();
        _finalStateDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_CAUSAL_CONV1D_DESCRIPTOR_EXT);
}

TEST_F(TestCausalConv1dOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// The parameter is the one required attribute to leave unset. Every other required attribute is
// set, so each case proves the omitted attribute alone is what finalize() rejects.
class TestCausalConv1dOperationDescriptorFinalizeFailsWithout
    : public TestCausalConv1dOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestCausalConv1dOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setRequiredAttributes(GetParam());
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestCausalConv1dOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                           HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                                           HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                                           HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                                           HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT));

TEST_F(TestCausalConv1dOperationDescriptor, FinalizeFailsStateIndicesWithoutInitialState)
{
    setRequiredAttributes(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT);
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCausalConv1dOperationDescriptor, FinalizeSucceedsWithoutStateIndices)
{
    setRequiredAttributes(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT);
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorWeight)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_weightDesc));

    ASSERT_EQ(desc->getData().weight_tensor_uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorCuSeqlens)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_cuSeqlensDesc));

    ASSERT_EQ(desc->getData().cu_seqlens_tensor_uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_biasDesc));

    ASSERT_EQ(desc->getData().bias_tensor_uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorInitialState)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_initialStateDesc));

    ASSERT_EQ(desc->getData().initial_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorStateIndices)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_stateIndicesDesc));

    ASSERT_EQ(desc->getData().state_indices_tensor_uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorDescriptorFinalState)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_finalStateDesc));

    ASSERT_EQ(desc->getData().final_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, SetCausalConv1dActivation)
{
    auto desc = getDescriptor();
    auto activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       1,
                                       &activation));

    ASSERT_EQ(desc->getData().activation, CausalConv1dActivation::SILU);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetCausalConv1dActivationWrongElementCount)
{
    auto desc = getDescriptor();
    auto activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                           HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                           2,
                           &activation),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCausalConv1dOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> guardX(retrievedX);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeCausalConv1dParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // activation
    hipdnnCausalConv1dActivation_t activation = HIPDNN_CAUSAL_CONV1D_ACTIVATION_IDENTITY_EXT;
    int64_t activationCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       1,
                                       &activationCount,
                                       &activation));
    ASSERT_EQ(activationCount, 1);
    EXPECT_EQ(activation, HIPDNN_CAUSAL_CONV1D_ACTIVATION_SILU_EXT);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT,
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

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorWeightQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_WEIGHT_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorCuSeqlensQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_CU_SEQLENS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorBiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorInitialStateQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_INITIAL_STATE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorStateIndicesQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_STATE_INDICES_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorFinalStateQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_FINAL_STATE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeCausalConv1dActivationQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestCausalConv1dOperationDescriptor,
       GetAttributeCausalConv1dActivationQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_CAUSAL_CONV1D_ACTIVATION_EXT,
                           HIPDNN_TYPE_CAUSAL_CONV1D_ACTIVATION_EXT,
                           0,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getWeightDesc(), nullptr);
    ASSERT_NE(desc->getCuSeqlensDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    ASSERT_NE(desc->getInitialStateDesc(), nullptr);
    ASSERT_NE(desc->getStateIndicesDesc(), nullptr);
    ASSERT_NE(desc->getFinalStateDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    ASSERT_EQ(desc->getWeightDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    ASSERT_EQ(desc->getCuSeqlensDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    ASSERT_EQ(desc->getBiasDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_EQ(desc->getInitialStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(desc->getStateIndicesDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(desc->getFinalStateDesc()->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("CausalConv1dOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("weight_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID)),
              std::string::npos);
    ASSERT_NE(str.find("cu_seqlens_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("bias_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_BIAS_UID)),
              std::string::npos);
    ASSERT_NE(
        str.find("initial_state_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID)),
        std::string::npos);
    ASSERT_NE(
        str.find("state_indices_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID)),
        std::string::npos);
    ASSERT_NE(str.find("final_state_uid=" + std::to_string(K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    ASSERT_EQ(tensors[0]->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(tensors[6]->getData().uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(tensors[7]->getData().uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
}

TEST_F(TestCausalConv1dOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::CausalConv1dAttributes);

    auto* attrs = node->attributes.AsCausalConv1dAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
    ASSERT_EQ(attrs->weight_tensor_uid, K_CAUSAL_CONV1D_TENSOR_WEIGHT_UID);
    ASSERT_EQ(attrs->cu_seqlens_tensor_uid, K_CAUSAL_CONV1D_TENSOR_CU_SEQLENS_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_CAUSAL_CONV1D_TENSOR_Y_UID);
    ASSERT_EQ(attrs->bias_tensor_uid, K_CAUSAL_CONV1D_TENSOR_BIAS_UID);
    ASSERT_EQ(attrs->initial_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_INITIAL_STATE_UID);
    ASSERT_EQ(attrs->state_indices_tensor_uid, K_CAUSAL_CONV1D_TENSOR_STATE_INDICES_UID);
    ASSERT_EQ(attrs->final_state_tensor_uid, K_CAUSAL_CONV1D_TENSOR_FINAL_STATE_UID);
    EXPECT_EQ(attrs->activation, CausalConv1dActivation::SILU);
}

TEST_F(TestCausalConv1dOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestCausalConv1dOperationDescriptor,
       GetTensorDescriptorsOrderIsXWeightCuSeqlensYBiasInitialStateStateIndicesFinalState)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    // Verify ordering: [X_EXT, WEIGHT_EXT, CU_SEQLENS_EXT, Y_EXT, BIAS_EXT, INITIAL_STATE_EXT, STATE_INDICES_EXT, FINAL_STATE_EXT] matches UIDs [9600, 9601, 9602, 9603, 9604, 9605, 9606, 9607]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getWeightDesc());
    EXPECT_EQ(tensors[2], desc->getCuSeqlensDesc());
    EXPECT_EQ(tensors[3], desc->getYDesc());
    EXPECT_EQ(tensors[4], desc->getBiasDesc());
    EXPECT_EQ(tensors[5], desc->getInitialStateDesc());
    EXPECT_EQ(tensors[6], desc->getStateIndicesDesc());
    EXPECT_EQ(tensors[7], desc->getFinalStateDesc());
}

TEST_F(TestCausalConv1dOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 8);
    ASSERT_EQ(tensors[0]->getData().uid, K_CAUSAL_CONV1D_TENSOR_X_UID);
}

TEST_F(TestCausalConv1dOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

// =============================================================================
// Operation Name Tests
// =============================================================================

TEST_F(TestCausalConv1dOperationDescriptor, SetAttributeNameSuccess)
{
    auto desc = getDescriptor();
    const std::string name = "test_causalconv1d_op";

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
    EXPECT_STREQ(buffer.data(), "test_causalconv1d_op");
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeNameQueryReturnsSizeInclNull)
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

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeOperationTypeReturnsCorrectType)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnOperationType_ext_t opType = HIPDNN_OPERATION_TYPE_NOT_SET_EXT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 1, &elementCount, &opType));

    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(opType, HIPDNN_OPERATION_TYPE_CAUSAL_CONV1D_EXT);
}

TEST_F(TestCausalConv1dOperationDescriptor, GetAttributeOperationTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_OPERATION_TYPE_EXT, HIPDNN_TYPE_OPERATION_TYPE_EXT, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestCausalConv1dOperationDescriptor, BuildNodePreservesName)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string opName = "test_causalconv1d";
    desc->setAttribute(HIPDNN_ATTR_OPERATION_NAME_EXT,
                       HIPDNN_TYPE_CHAR,
                       static_cast<int64_t>(opName.size()),
                       opName.c_str());
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_CAUSAL_CONV1D_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->name, "test_causalconv1d");
}
