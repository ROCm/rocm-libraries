// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleDequantizeOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestBlockScaleDequantizeOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BlockScaleDequantizeOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BlockScaleDequantizeOperationDescriptor>();
    }

    void setTensors() const
    {
        auto desc = getDescriptor();
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_scaleDesc);
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_yDesc);
    }

    void setBlockScaleDequantizeParams() const
    {
        auto desc = getDescriptor();
        std::vector<int64_t> blockSize = {K_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE};

        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                           HIPDNN_TYPE_INT64,
                           1,
                           blockSize.data());
    }

    void setRequiredAttributes() const
    {
        setTensors();
        setBlockScaleDequantizeParams();
        auto computeType = HIPDNN_DATA_FLOAT;
        getDescriptor()->setAttribute(HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT,
                                      HIPDNN_TYPE_DATA_TYPE,
                                      1,
                                      &computeType);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _xDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BlockScaleDequantizeOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID,
                                       toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_DIMS),
                                       toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_UID,
                                           toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_DIMS),
                                           toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_STRIDES));
        _yDesc = createFinalizedTensor(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_UID,
                                       toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_DIMS),
                                       toVec(K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _scaleDesc.reset();
        _yDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR_EXT);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeFailsWithoutXTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_yDesc);
    setBlockScaleDequantizeParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeFailsWithoutScaleTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_yDesc);
    setBlockScaleDequantizeParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeFailsWithoutYTensor)
{
    auto desc = getDescriptor();
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_xDesc);
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                       1,
                       &_scaleDesc);
    setBlockScaleDequantizeParams();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeFailsWithoutBlockSize)
{
    auto desc = getDescriptor();
    setTensors();

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeFailsWithoutComputeType)
{
    setTensors();
    setBlockScaleDequantizeParams();
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_unfinalizedTensor),
        HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           2,
                           &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetBlockSize)
{
    auto desc = getDescriptor();
    std::vector<int64_t> blockSize = {K_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE};

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       blockSize.data()));
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT,
                                                  HIPDNN_TYPE_DATA_TYPE,
                                                  2,
                                                  &computeType),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetBlockSizeParamsWrongType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> padding = {1, 1};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                           HIPDNN_TYPE_CHAR,
                           2,
                           padding.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           &_xDesc),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       &retrievedX));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeBlockScaleDequantizeParams)
{
    makeFinalized();
    auto desc = getDescriptor();

    // block_size
    std::vector<int64_t> blockSize(1);
    int64_t blockSizeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &blockSizeCount,
                                       blockSize.data()));

    ASSERT_EQ(blockSizeCount, 1);
    EXPECT_EQ(blockSize, (std::vector<int64_t>{K_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE}));
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT,
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

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           &dummy),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_Y_EXT,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeBlockSizeQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeBlockSizeQueryThenRetrieve)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Query: get the element count
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);

    // Retrieve: use the queried count to allocate and fetch
    std::vector<int64_t> blockSize(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE_EXT,
                                       HIPDNN_TYPE_INT64,
                                       elementCount,
                                       &retrievedCount,
                                       blockSize.data()));
    ASSERT_EQ(retrievedCount, 1);
    EXPECT_EQ(blockSize, (std::vector<int64_t>{K_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE}));
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_X_EXT,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           0,
                           nullptr,
                           nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, ToStringContainsExpectedInfo)
{
    setRequiredAttributes();
    auto desc = getDescriptor();

    std::string str = desc->toString();
    ASSERT_NE(str.find("BlockScaleDequantizeOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=50"), std::string::npos);
    ASSERT_NE(str.find("scale_uid=51"), std::string::npos);
    ASSERT_NE(str.find("y_uid=52"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID);
    ASSERT_EQ(tensors[1]->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_UID);
    ASSERT_EQ(tensors[2]->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_UID);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(
        HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);

    auto* attrs = node->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_Y_UID);
    ASSERT_EQ(attrs->block_size.size(), 1);
    ASSERT_EQ(attrs->block_size[0], K_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setRequiredAttributes();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(
        HIPDNN_ATTR_BLOCK_SCALE_DEQUANTIZE_COMP_TYPE_EXT, HIPDNN_TYPE_DATA_TYPE, 1, &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetTensorDescriptorsOrderIsXScaleY)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    // Verify ordering: [X, SCALE, Y] matches UIDs [50, 51, 52]
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getScaleDesc());
    EXPECT_EQ(tensors[2], desc->getYDesc());
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();

    auto graphOp = _wrapper->tryAsInterface<IGraphOperation>();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface is the same underlying object
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0]->getData().uid, K_BLOCK_SCALE_DEQUANTIZE_TENSOR_X_UID);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsInterface<IGraphOperation>();
    EXPECT_EQ(graphOp, nullptr);
}
