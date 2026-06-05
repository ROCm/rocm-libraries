// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "HipdnnNormFwdPhase.h"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/LayernormOperationDescriptor.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/layernorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/LayernormConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <algorithm>
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

class TestLayernormOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<LayernormOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<LayernormOperationDescriptor>();
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
        setIf(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, _biasDesc);
        setIf(HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, _epsilonDesc);
        setIf(HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, _yDesc);
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT) == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(
                HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
        }
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT)
           == skip.end())
        {
            auto forwardPhase = HIPDNN_NORM_FWD_INFERENCE;
            desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                               HIPDNN_TYPE_NORM_FWD_PHASE,
                               1,
                               &forwardPhase);
        }
    }

    void setRequiredAttributes() const
    {
        setAllAttributesExcept({});
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_meanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invVarianceDesc);
        desc->finalize();
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
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<LayernormOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_X_UID,
                                       toVec(K_LAYERNORM_TENSOR_X_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_SCALE_UID,
                                           toVec(K_LAYERNORM_TENSOR_SCALE_DIMS),
                                           toVec(K_LAYERNORM_TENSOR_SCALE_STRIDES));
        _biasDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_BIAS_UID,
                                          toVec(K_LAYERNORM_TENSOR_BIAS_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_BIAS_STRIDES));
        _epsilonDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_EPSILON_UID,
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_DIMS),
                                             toVec(K_LAYERNORM_TENSOR_EPSILON_STRIDES));
        _yDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_Y_UID,
                                       toVec(K_LAYERNORM_TENSOR_Y_DIMS),
                                       toVec(K_LAYERNORM_TENSOR_Y_STRIDES));
        _meanDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_MEAN_UID,
                                          toVec(K_LAYERNORM_TENSOR_MEAN_DIMS),
                                          toVec(K_LAYERNORM_TENSOR_MEAN_STRIDES));
        _invVarianceDesc = createFinalizedTensor(K_LAYERNORM_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_DIMS),
                                                 toVec(K_LAYERNORM_TENSOR_INV_VARIANCE_STRIDES));
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
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_LAYERNORM_DESCRIPTOR_EXT);
}

TEST_F(TestLayernormOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestLayernormOperationDescriptor, FinalizeSucceedsWithoutOptionalTensors)
{
    // Mean and inv_variance are optional - finalize should succeed without them
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
}

TEST_F(TestLayernormOperationDescriptor, DoubleFinalizeSucceeds)
{
    // LayernormOperationDescriptor::finalize() has no guard for already-finalized state.
    // All required fields remain set after the first finalize, so the second call
    // succeeds without throwing. This documents the current behavior.
    makeFinalized();
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->finalize());
}

// =============================================================================
// Parameterized Finalize-fails-without Tests
// =============================================================================

class TestLayernormOperationDescriptorFinalizeFailsWithout
    : public TestLayernormOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestLayernormOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestLayernormOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT,
                                           HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT,
                                           HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT));

TEST_F(TestLayernormOperationDescriptor, FinalizeFailsWithOnlyMean)
{
    setRequiredAttributes();
    auto desc = getDescriptor();
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_meanDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormOperationDescriptor, FinalizeFailsWithOnlyInvVariance)
{
    setRequiredAttributes();
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_invVarianceDesc);
    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_biasDesc));

    ASSERT_EQ(desc->getData().bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorEpsilon)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_epsilonDesc));

    ASSERT_EQ(desc->getData().epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    ASSERT_NE(desc->getEpsilonDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_meanDesc));

    ASSERT_TRUE(desc->getData().mean_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorDescriptorInvVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invVarianceDesc));

    ASSERT_TRUE(desc->getData().inv_variance_tensor_uid.has_value());
    ASSERT_EQ(desc->getData().inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 2, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, SetNormalizedDimCount)
{
    auto desc = getDescriptor();
    int64_t dimCount = 3;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT, HIPDNN_TYPE_INT64, 1, &dimCount));

    ASSERT_EQ(desc->getData().normalized_dim_count, 3);
}

TEST_F(TestLayernormOperationDescriptor, SetNormFwdPhase)
{
    auto desc = getDescriptor();
    auto forwardPhase = HIPDNN_NORM_FWD_INFERENCE;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       1,
                                       &forwardPhase));

    ASSERT_EQ(desc->getData().forward_phase, NormFwdPhase::INFERENCE);
}

TEST_F(TestLayernormOperationDescriptor, SetNormFwdPhaseTraining)
{
    auto desc = getDescriptor();
    auto forwardPhase = HIPDNN_NORM_FWD_TRAINING;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       1,
                                       &forwardPhase));

    ASSERT_EQ(desc->getData().forward_phase, NormFwdPhase::TRAINING);
}

TEST_F(TestLayernormOperationDescriptor, SetNormFwdPhaseWrongElementCount)
{
    auto desc = getDescriptor();
    hipdnnNormFwdPhase_t forwardPhase = HIPDNN_NORM_FWD_INFERENCE;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                                  HIPDNN_TYPE_NORM_FWD_PHASE,
                                                  2,
                                                  &forwardPhase),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestLayernormOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestLayernormOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestLayernormOperationDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
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

class TestLayernormOperationDescriptorGetTensor
    : public TestLayernormOperationDescriptor,
      public ::testing::WithParamInterface<TensorAttrCase>
{
};

TEST_P(TestLayernormOperationDescriptorGetTensor, GetAttributeTensorDescriptorReturnsCorrectTensor)
{
    makeFinalized();
    auto desc = getDescriptor();
    const auto& tc = GetParam();

    // getAttribute packs the stored TensorDescriptor into a fresh HipdnnBackendDescriptor*.
    // Ownership is transferred to the caller — delete after use.
    HipdnnBackendDescriptor* retrieved = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        tc.attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, &elementCount, static_cast<void*>(&retrieved)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrieved, nullptr);

    // Unpack the wrapper to inspect the underlying TensorDescriptor.
    auto tensorImpl = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrieved, HIPDNN_STATUS_INTERNAL_ERROR, "Failed to unpack retrieved tensor descriptor");
    delete retrieved;

    ASSERT_NE(tensorImpl, nullptr);
    EXPECT_EQ(tensorImpl->getData().uid, tc.expectedUid);
}

TEST_P(TestLayernormOperationDescriptorGetTensor, QueryModeReturnsOne)
{
    // makeFinalized() sets all tensors including optional mean/inv_variance.
    // Query mode (elementCount=0, arrayOfElements=nullptr) should report 1 for each.
    makeFinalized();
    auto desc = getDescriptor();
    const auto& tc = GetParam();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(
        desc->getAttribute(tc.attr, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllTensors,
    TestLayernormOperationDescriptorGetTensor,
    ::testing::Values(
        TensorAttrCase{HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT, "X", K_LAYERNORM_TENSOR_X_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT, "Scale", K_LAYERNORM_TENSOR_SCALE_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT, "Bias", K_LAYERNORM_TENSOR_BIAS_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT, "Epsilon", K_LAYERNORM_TENSOR_EPSILON_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT, "Y", K_LAYERNORM_TENSOR_Y_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT, "Mean", K_LAYERNORM_TENSOR_MEAN_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                       "InvVariance",
                       K_LAYERNORM_TENSOR_INV_VARIANCE_UID}),
    [](const ::testing::TestParamInfo<TensorAttrCase>& info) { return info.param.name; });

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, GetAttributeNormalizedDimCount)
{
    auto desc = getDescriptor();
    int64_t dimCount = 3;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT, HIPDNN_TYPE_INT64, 1, &dimCount);
    setRequiredAttributes();
    desc->finalize();

    int64_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(elementCount, 1);
    EXPECT_EQ(retrieved, 3);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeLayernormParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    hipdnnNormFwdPhase_t forwardPhase = {};
    int64_t forwardPhaseCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       1,
                                       &forwardPhaseCount,
                                       &forwardPhase));
    ASSERT_EQ(forwardPhaseCount, 1);
    EXPECT_EQ(forwardPhase, HIPDNN_NORM_FWD_INFERENCE);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorBiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorEpsilonQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_EPSILON_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorMeanQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorInvVarianceQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeNormFwdPhaseQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_FWD_PHASE_EXT,
                                       HIPDNN_TYPE_NORM_FWD_PHASE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestLayernormOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_LAYERNORM_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, FinalizePreservesTensorReferences)
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

    ASSERT_EQ(desc->getXDesc()->getData().uid, K_LAYERNORM_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_LAYERNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getBiasDesc()->getData().uid, K_LAYERNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(desc->getEpsilonDesc()->getData().uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_LAYERNORM_TENSOR_Y_UID);
    ASSERT_EQ(desc->getMeanDesc()->getData().uid, K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_EQ(desc->getInvVarianceDesc()->getData().uid, K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("LayernormOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_LAYERNORM_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_LAYERNORM_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("bias_uid=" + std::to_string(K_LAYERNORM_TENSOR_BIAS_UID)),
              std::string::npos);
    ASSERT_NE(str.find("epsilon_uid=" + std::to_string(K_LAYERNORM_TENSOR_EPSILON_UID)),
              std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_LAYERNORM_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
    // Optional tensors not set — must not appear in toString output
    ASSERT_EQ(str.find("mean_uid="), std::string::npos);
    ASSERT_EQ(str.find("inv_variance_uid="), std::string::npos);
}

TEST_F(TestLayernormOperationDescriptor, ToStringIncludesOptionalTensorsWhenSet)
{
    makeFinalized();
    const std::string str = getDescriptor()->toString();
    ASSERT_NE(str.find("mean_uid=" + std::to_string(K_LAYERNORM_TENSOR_MEAN_UID)),
              std::string::npos);
    ASSERT_NE(str.find("inv_variance_uid=" + std::to_string(K_LAYERNORM_TENSOR_INV_VARIANCE_UID)),
              std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestLayernormOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7u);
    // getTensorDescriptors() returns the same shared_ptr objects held internally —
    // pointer identity proves no accidental clone was made.
    ASSERT_EQ(tensors[0].get(), desc->getXDesc().get());
    ASSERT_EQ(tensors[1].get(), desc->getScaleDesc().get());
    ASSERT_EQ(tensors[2].get(), desc->getBiasDesc().get());
    ASSERT_EQ(tensors[3].get(), desc->getEpsilonDesc().get());
    ASSERT_EQ(tensors[4].get(), desc->getYDesc().get());
    ASSERT_EQ(tensors[5].get(), desc->getMeanDesc().get());
    ASSERT_EQ(tensors[6].get(), desc->getInvVarianceDesc().get());
}

TEST_F(TestLayernormOperationDescriptor, GetTensorDescriptorsWithoutOptional)
{
    setRequiredAttributes();
    getDescriptor()->finalize();

    auto desc = getDescriptor();
    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 5u);
    ASSERT_EQ(tensors[0].get(), desc->getXDesc().get());
    ASSERT_EQ(tensors[1].get(), desc->getScaleDesc().get());
    ASSERT_EQ(tensors[2].get(), desc->getBiasDesc().get());
    ASSERT_EQ(tensors[3].get(), desc->getEpsilonDesc().get());
    ASSERT_EQ(tensors[4].get(), desc->getYDesc().get());
}

TEST_F(TestLayernormOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    makeFinalized();

    auto desc = getDescriptor();
    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::LayernormAttributes);

    auto* attrs = node->attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_LAYERNORM_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_LAYERNORM_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->bias_tensor_uid, K_LAYERNORM_TENSOR_BIAS_UID);
    ASSERT_EQ(attrs->epsilon_tensor_uid, K_LAYERNORM_TENSOR_EPSILON_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_LAYERNORM_TENSOR_Y_UID);
    ASSERT_TRUE(attrs->mean_tensor_uid.has_value());
    ASSERT_EQ(attrs->mean_tensor_uid.value(), K_LAYERNORM_TENSOR_MEAN_UID);
    ASSERT_TRUE(attrs->inv_variance_tensor_uid.has_value());
    ASSERT_EQ(attrs->inv_variance_tensor_uid.value(), K_LAYERNORM_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(attrs->forward_phase, NormFwdPhase::INFERENCE);
    ASSERT_EQ(attrs->normalized_dim_count, 0); // not set via setAttribute in makeFinalized()
}

TEST_F(TestLayernormOperationDescriptor, BuildNodePreservesNormalizedDimCount)
{
    auto desc = getDescriptor();
    int64_t dimCount = 3;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_LAYERNORM_NORMALIZED_DIM_COUNT_EXT, HIPDNN_TYPE_INT64, 1, &dimCount);
    setRequiredAttributes();
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    auto* attrs = node->attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->normalized_dim_count, 3);
}

TEST_F(TestLayernormOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_LAYERNORM_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestLayernormOperationDescriptor, BuildNodeWithoutOptionalTensors)
{
    setRequiredAttributes();
    getDescriptor()->finalize();

    auto node = getDescriptor()->buildNode();
    ASSERT_NE(node, nullptr);

    auto* attrs = node->attributes.AsLayernormAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_FALSE(attrs->mean_tensor_uid.has_value());
    ASSERT_FALSE(attrs->inv_variance_tensor_uid.has_value());
}

TEST_F(TestLayernormOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 7u);
    ASSERT_EQ(tensors[0]->getData().uid, K_LAYERNORM_TENSOR_X_UID);
}

TEST_F(TestLayernormOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}

} // namespace
