// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/BlockScaleQuantizeOperationDescriptor.hpp"
#include "descriptors/IGraphOperation.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_quantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleQuantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_tests::constants::K_BSQ_BLOCK_SIZE;
using hipdnn_tests::constants::K_BSQ_TENSOR_SCALE_DIMS;
using hipdnn_tests::constants::K_BSQ_TENSOR_SCALE_STRIDES;
using hipdnn_tests::constants::K_BSQ_TENSOR_SCALE_UID;
using hipdnn_tests::constants::K_BSQ_TENSOR_X_DIMS;
using hipdnn_tests::constants::K_BSQ_TENSOR_X_STRIDES;
using hipdnn_tests::constants::K_BSQ_TENSOR_X_UID;
using hipdnn_tests::constants::K_BSQ_TENSOR_Y_DIMS;
using hipdnn_tests::constants::K_BSQ_TENSOR_Y_STRIDES;
using hipdnn_tests::constants::K_BSQ_TENSOR_Y_UID;

class TestBlockScaleQuantizeOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BlockScaleQuantizeOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BlockScaleQuantizeOperationDescriptor>();
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
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC, _yDesc);
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC, _scaleDesc);
        if(std::find(
               skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE)
           == skip.end())
        {
            int32_t blockSize = K_BSQ_BLOCK_SIZE;
            desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                               HIPDNN_TYPE_INT32,
                               1,
                               &blockSize);
        }
        if(std::find(skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC)
           == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                               HIPDNN_TYPE_DATA_TYPE,
                               1,
                               &computeType);
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
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BlockScaleQuantizeOperationDescriptor>();
        _xDesc = createFinalizedTensor(K_BSQ_TENSOR_X_UID,
                                       hipdnn_tests::toVec(K_BSQ_TENSOR_X_DIMS),
                                       hipdnn_tests::toVec(K_BSQ_TENSOR_X_STRIDES));
        _yDesc = createFinalizedTensor(K_BSQ_TENSOR_Y_UID,
                                       hipdnn_tests::toVec(K_BSQ_TENSOR_Y_DIMS),
                                       hipdnn_tests::toVec(K_BSQ_TENSOR_Y_STRIDES));
        _scaleDesc = createFinalizedTensor(K_BSQ_TENSOR_SCALE_UID,
                                           hipdnn_tests::toVec(K_BSQ_TENSOR_SCALE_DIMS),
                                           hipdnn_tests::toVec(K_BSQ_TENSOR_SCALE_STRIDES));
        _unfinalizedTensor = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
        _xDesc.reset();
        _yDesc.reset();
        _scaleDesc.reset();
        _unfinalizedTensor.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_QUANTIZE_DESCRIPTOR);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

class TestBlockScaleQuantizeOperationDescriptorFinalizeFailsWithout
    : public TestBlockScaleQuantizeOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestBlockScaleQuantizeOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestBlockScaleQuantizeOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    // Verify UID extracted via getData()
    ASSERT_EQ(desc->getData().x_tensor_uid, K_BSQ_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_unfinalizedTensor),
                               HIPDNN_STATUS_BAD_PARAM_NOT_FINALIZED);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorFailsWrongType)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(
            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  2,
                                                  &_xDesc),
                               HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// SetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                           HIPDNN_TYPE_DATA_TYPE,
                           2,
                           &computeType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - Optional Fields
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetAndGetBlockSize)
{
    auto desc = getDescriptor();
    int32_t blockSize = K_BSQ_BLOCK_SIZE;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE, HIPDNN_TYPE_INT32, 1, &blockSize));

    ASSERT_EQ(desc->getData().block_size, K_BSQ_BLOCK_SIZE);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetAndGetAxis)
{
    auto desc = getDescriptor();
    int64_t axis = 2;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT, HIPDNN_TYPE_INT64, 1, &axis));

    ASSERT_TRUE(desc->getData().axis.has_value());
    ASSERT_EQ(desc->getData().axis.value(), 2);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetAndGetTranspose)
{
    auto desc = getDescriptor();
    bool transpose = true;
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &transpose));

    ASSERT_TRUE(desc->getData().transpose);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  &_xDesc),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, SetAttributeUnsupported)
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

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTensorDescriptor)
{
    makeFinalized();
    auto desc = getDescriptor();

    HipdnnBackendDescriptor* retrievedX = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&retrievedX)));
    const std::unique_ptr<HipdnnBackendDescriptor> ownedX(retrievedX);

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(retrievedX, nullptr);
}

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeBlockSizeAfterFinalize)
{
    setAllAttributesExcept();
    getDescriptor()->finalize();
    auto desc = getDescriptor();

    int32_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       1,
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved, K_BSQ_BLOCK_SIZE);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeAxisAfterFinalize)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    int64_t axis = 3;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT, HIPDNN_TYPE_INT64, 1, &axis);
    desc->finalize();

    int64_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       1,
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved, 3);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeAxisQueryReturnsZeroWhenAbsent)
{
    setAllAttributesExcept();
    getDescriptor()->finalize();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 0);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeAxisQueryReturnsOneWhenPresent)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    int64_t axis = 1;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT, HIPDNN_TYPE_INT64, 1, &axis);
    desc->finalize();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT,
                                       HIPDNN_TYPE_INT64,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTransposeAfterFinalize)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    bool transpose = true;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &transpose);
    desc->finalize();

    bool retrieved = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &retrieved));
    ASSERT_TRUE(retrieved);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  &dummy),
                               HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  1,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeUnsupported)
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

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTensorXQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTensorYQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTensorScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_SCALE_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeComputeTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeBlockSizeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_XDESC,
                                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                                  0,
                                                  nullptr,
                                                  nullptr),
                               HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// Accessor Tests
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, FinalizePreservesTensorReferences)
{
    makeFinalized();
    auto desc = getDescriptor();

    // Verify the tensor descriptors are preserved
    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);

    // Verify UIDs match
    ASSERT_EQ(desc->getXDesc()->getData().uid, K_BSQ_TENSOR_X_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_BSQ_TENSOR_Y_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_BSQ_TENSOR_SCALE_UID);
}

// =============================================================================
// Finalize Succeeds Without Optional Fields
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, FinalizeSucceedsWithoutOptionalAxis)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT});
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, FinalizeSucceedsWithoutOptionalTranspose)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_TRANSPOSE_EXT});
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("BlockScaleQuantizeOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_BSQ_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_BSQ_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_BSQ_TENSOR_SCALE_UID)), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBlockScaleQuantizeOperationDescriptor, GetTensorDescriptorsReturnsAllTensors)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto tensors = desc->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getYDesc());
    EXPECT_EQ(tensors[2], desc->getScaleDesc());
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BlockScaleQuantizeAttributes);

    auto* attrs = node->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_BSQ_TENSOR_X_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_BSQ_TENSOR_Y_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_BSQ_TENSOR_SCALE_UID);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept();

    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::HALF);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, BuildNodeWithBlockSizeAndAxis)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    int64_t axis = 1;
    desc->setAttribute(
        HIPDNN_ATTR_OPERATION_BLOCK_SCALE_QUANTIZE_AXIS_EXT, HIPDNN_TYPE_INT64, 1, &axis);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);

    auto* attrs = node->attributes.AsBlockScaleQuantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_EQ(attrs->block_size, K_BSQ_BLOCK_SIZE);
    ASSERT_TRUE(attrs->axis.has_value());
    EXPECT_EQ(attrs->axis.value(), 1);
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, TryAsInterfaceReturnsValidGraphOp)
{
    makeFinalized();
    auto desc = getDescriptor();

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    // Verify the returned interface exposes the same tensor shared_ptrs
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    ASSERT_EQ(tensors[0].get(), desc->getXDesc().get());
    ASSERT_EQ(tensors[1].get(), desc->getYDesc().get());
    ASSERT_EQ(tensors[2].get(), desc->getScaleDesc().get());
}

TEST_F(TestBlockScaleQuantizeOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
