// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BatchnormInferenceOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BatchnormInferenceConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <memory>
#include <string>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestBatchnormInferenceOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BatchnormInferenceOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BatchnormInferenceOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_meanDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_invVarianceDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_biasDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_yDesc);
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
            HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _meanDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _invVarianceDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _biasDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BatchnormInferenceOperationDescriptor>();
        _xDesc = createFinalizedTensor(
            K_BN_INF_TENSOR_X_UID, toVec(K_BN_INF_SPATIAL_DIMS), toVec(K_BN_INF_SPATIAL_STRIDES));
        _meanDesc = createFinalizedTensor(K_BN_INF_TENSOR_MEAN_UID,
                                          toVec(K_BN_INF_CHANNEL_DIMS),
                                          toVec(K_BN_INF_CHANNEL_STRIDES));
        _invVarianceDesc = createFinalizedTensor(K_BN_INF_TENSOR_INV_VARIANCE_UID,
                                                 toVec(K_BN_INF_CHANNEL_DIMS),
                                                 toVec(K_BN_INF_CHANNEL_STRIDES));
        _scaleDesc = createFinalizedTensor(K_BN_INF_TENSOR_SCALE_UID,
                                           toVec(K_BN_INF_CHANNEL_DIMS),
                                           toVec(K_BN_INF_CHANNEL_STRIDES));
        _biasDesc = createFinalizedTensor(K_BN_INF_TENSOR_BIAS_UID,
                                          toVec(K_BN_INF_CHANNEL_DIMS),
                                          toVec(K_BN_INF_CHANNEL_STRIDES));
        _yDesc = createFinalizedTensor(
            K_BN_INF_TENSOR_Y_UID, toVec(K_BN_INF_SPATIAL_DIMS), toVec(K_BN_INF_SPATIAL_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _meanDesc.reset();
        _invVarianceDesc.reset();
        _scaleDesc.reset();
        _biasDesc.reset();
        _yDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BATCHNORM_INFERENCE_DESCRIPTOR_EXT);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, _invVarianceDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, _biasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, _yDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutInvVarianceTensor)
{
    auto desc = getDescriptor();
    setTensors({{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, _xDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, _meanDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, _scaleDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, _biasDesc.get()},
                {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, _yDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutMeanTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, _invVarianceDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, _biasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, _yDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutScaleTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, _invVarianceDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, _biasDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, _yDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutBiasTensor)
{
    auto desc = getDescriptor();
    setTensors(
        {{HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, _xDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, _invVarianceDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, _meanDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, _scaleDesc.get()},
         {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT, _yDesc.get()}});

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    setTensors({
        {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, _xDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT, _invVarianceDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT, _meanDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT, _scaleDesc.get()},
        {HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT, _biasDesc.get()},
    });

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorMean)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_meanDesc));

    ASSERT_EQ(desc->getData().mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorInvVariance)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_invVarianceDesc));

    ASSERT_EQ(desc->getData().inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorBias)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_biasDesc));

    ASSERT_EQ(desc->getData().bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_xDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 2, &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_xDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* rawX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawX)));
    const std::unique_ptr<HipdnnBackendDescriptor> retrievedX(rawX);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT,
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

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorMeanQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_MEAN_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorInvVarianceQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_INV_VARIANCE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorBiasQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_BIAS_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BATCHNORM_INFERENCE_X_EXT,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getMeanDesc(), nullptr);
    ASSERT_NE(desc->getInvVarianceDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getBiasDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_BN_INF_TENSOR_X_UID);
    ASSERT_EQ(desc->getMeanDesc()->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    ASSERT_EQ(desc->getInvVarianceDesc()->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getBiasDesc()->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_BN_INF_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("BatchnormInferenceOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_BN_INF_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("mean_uid=" + std::to_string(K_BN_INF_TENSOR_MEAN_UID)), std::string::npos);
    ASSERT_NE(str.find("inv_variance_uid=" + std::to_string(K_BN_INF_TENSOR_INV_VARIANCE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_BN_INF_TENSOR_SCALE_UID)),
              std::string::npos);
    ASSERT_NE(str.find("bias_uid=" + std::to_string(K_BN_INF_TENSOR_BIAS_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_BN_INF_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBatchnormInferenceOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_BN_INF_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_BN_INF_TENSOR_MEAN_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(tensors[3]->getData().uid, K_BN_INF_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[4]->getData().uid, K_BN_INF_TENSOR_BIAS_UID);
    ASSERT_EQ(tensors[5]->getData().uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BatchnormInferenceAttributes);

    auto* attrs = node->attributes.AsBatchnormInferenceAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_BN_INF_TENSOR_X_UID);
    ASSERT_EQ(attrs->mean_tensor_uid, K_BN_INF_TENSOR_MEAN_UID);
    ASSERT_EQ(attrs->inv_variance_tensor_uid, K_BN_INF_TENSOR_INV_VARIANCE_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_BN_INF_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->bias_tensor_uid, K_BN_INF_TENSOR_BIAS_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_BN_INF_TENSOR_Y_UID);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BATCHNORM_INF_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestBatchnormInferenceOperationDescriptor,
       GetTensorDescriptorsOrderIsXMeanInvVarianceScaleBiasY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    // Verify ordering: [X, MEAN, INV_VARIANCE, SCALE, BIAS, Y]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getMeanDesc());
    EXPECT_EQ(tensors[2], desc->getInvVarianceDesc());
    EXPECT_EQ(tensors[3], desc->getScaleDesc());
    EXPECT_EQ(tensors[4], desc->getBiasDesc());
    EXPECT_EQ(tensors[5], desc->getYDesc());
}

TEST_F(TestBatchnormInferenceOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 6);
    ASSERT_EQ(tensors[0]->getData().uid, K_BN_INF_TENSOR_X_UID);
}

TEST_F(TestBatchnormInferenceOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
