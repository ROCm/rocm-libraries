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
#include <hipdnn_flatbuffers_sdk/data_objects/block_scale_dequantize_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/BlockScaleDequantizeConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

#include <algorithm>
#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

class TestBlockScaleDequantizeOperationDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<BlockScaleDequantizeOperationDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<BlockScaleDequantizeOperationDescriptor>();
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
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC, _xDesc);
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC, _scaleDesc);
        setIf(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC, _yDesc);

        if(std::find(
               skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE)
           == skip.end())
        {
            std::vector<int32_t> blockSize = {K_BSD_BLOCK_SIZE};
            desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                               HIPDNN_TYPE_INT32,
                               1,
                               blockSize.data());
        }
        if(std::find(
               skip.begin(), skip.end(), HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC)
           == skip.end())
        {
            auto computeType = HIPDNN_DATA_FLOAT;
            desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
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
    std::unique_ptr<HipdnnBackendDescriptor> _scaleDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _yDesc = nullptr;
    std::unique_ptr<HipdnnBackendDescriptor> _unfinalizedTensor = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<BlockScaleDequantizeOperationDescriptor>();
        _xDesc = createFinalizedTensor(
            K_BSD_TENSOR_X_UID, toVec(K_BSD_TENSOR_X_DIMS), toVec(K_BSD_TENSOR_X_STRIDES));
        _scaleDesc = createFinalizedTensor(K_BSD_TENSOR_SCALE_UID,
                                           toVec(K_BSD_TENSOR_SCALE_DIMS),
                                           toVec(K_BSD_TENSOR_SCALE_STRIDES));
        _yDesc = createFinalizedTensor(
            K_BSD_TENSOR_Y_UID, toVec(K_BSD_TENSOR_Y_DIMS), toVec(K_BSD_TENSOR_Y_STRIDES));
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
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_OPERATION_BLOCK_SCALE_DEQUANTIZE_DESCRIPTOR);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, FinalizeWithRequiredAttributes)
{
    setAllAttributesExcept();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

// =============================================================================
// Parameterized Finalize-Fails-Without Tests
// =============================================================================

class TestBlockScaleDequantizeOperationDescriptorFinalizeFailsWithout
    : public TestBlockScaleDequantizeOperationDescriptor,
      public ::testing::WithParamInterface<hipdnnBackendAttributeName_t>
{
};

TEST_P(TestBlockScaleDequantizeOperationDescriptorFinalizeFailsWithout, FinalizeFailsWithout)
{
    setAllAttributesExcept({GetParam()});
    ASSERT_THROW_HIPDNN_STATUS(getDescriptor()->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

INSTANTIATE_TEST_SUITE_P(RequiredAttributes,
                         TestBlockScaleDequantizeOperationDescriptorFinalizeFailsWithout,
                         ::testing::Values(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                           HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC));

// =============================================================================
// SetAttribute Tests - Tensor Descriptors
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorX)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_xDesc));

    ASSERT_EQ(desc->getData().x_tensor_uid, K_BSD_TENSOR_X_UID);
    ASSERT_NE(desc->getXDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorScale)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_scaleDesc));

    ASSERT_EQ(desc->getData().scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorDescriptorY)
{
    auto desc = getDescriptor();
    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &_yDesc));

    ASSERT_EQ(desc->getData().y_tensor_uid, K_BSD_TENSOR_Y_UID);
    ASSERT_NE(desc->getYDesc(), nullptr);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsNotFinalized)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
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
            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC, HIPDNN_TYPE_INT64, 1, &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsWrongElementCount)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           2,
                           &_xDesc),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetTensorFailsNullPointer)
{
    auto desc = getDescriptor();
    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
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
    std::vector<int32_t> blockSize = {K_BSD_BLOCK_SIZE};

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       1,
                                       blockSize.data()));

    // Verify round-trip via getAttribute after finalize
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE});
    desc->finalize();

    std::vector<int32_t> retrieved(1);
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       1,
                                       &count,
                                       retrieved.data()));
    ASSERT_EQ(count, 1);
    EXPECT_EQ(retrieved, blockSize);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetIsNegativeScale)
{
    auto desc = getDescriptor();
    bool isNegativeScale = true;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &isNegativeScale));

    ASSERT_EQ(desc->getData().is_negative_scale, true);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetComputeDataType)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &computeType));

    ASSERT_EQ(desc->getComputeDataType(), DataType::FLOAT);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, SetComputeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
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
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
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
        desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
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
// GetAttribute Tests - Tensor Descriptors (parameterized)
// =============================================================================

struct TensorAttrCase
{
    hipdnnBackendAttributeName_t attr;
    const char* name;
    int64_t expectedUid;
};

class TestBlockScaleDequantizeOperationDescriptorGetTensor
    : public TestBlockScaleDequantizeOperationDescriptor,
      public ::testing::WithParamInterface<TensorAttrCase>
{
};

TEST_P(TestBlockScaleDequantizeOperationDescriptorGetTensor,
       GetAttributeTensorDescriptorReturnsCorrectTensor)
{
    makeFinalized();
    auto desc = getDescriptor();
    const auto& tc = GetParam();

    HipdnnBackendDescriptor* rawRetrieved = nullptr;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(tc.attr,
                                       HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                       1,
                                       &elementCount,
                                       static_cast<void*>(&rawRetrieved)));

    ASSERT_EQ(elementCount, 1);
    ASSERT_NE(rawRetrieved, nullptr);
    const std::unique_ptr<HipdnnBackendDescriptor> retrieved(rawRetrieved);

    auto tensorImpl = HipdnnBackendDescriptor::unpackDescriptor<TensorDescriptor>(
        retrieved.get(),
        HIPDNN_STATUS_INTERNAL_ERROR,
        "Failed to unpack retrieved tensor descriptor");

    ASSERT_NE(tensorImpl, nullptr);
    EXPECT_EQ(tensorImpl->getData().uid, tc.expectedUid);
}

TEST_P(TestBlockScaleDequantizeOperationDescriptorGetTensor, QueryModeReturnsOne)
{
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
    TestBlockScaleDequantizeOperationDescriptorGetTensor,
    ::testing::Values(
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC, "X", K_BSD_TENSOR_X_UID},
        TensorAttrCase{HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_SCALE_DESC,
                       "Scale",
                       K_BSD_TENSOR_SCALE_UID},
        TensorAttrCase{
            HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_YDESC, "Y", K_BSD_TENSOR_Y_UID}),
    [](const ::testing::TestParamInfo<TensorAttrCase>& info) { return info.param.name; });

// =============================================================================
// GetAttribute Tests - Data Fields
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeBlockSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int32_t> blockSize(1);
    int64_t blockSizeCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       1,
                                       &blockSizeCount,
                                       blockSize.data()));

    ASSERT_EQ(blockSizeCount, 1);
    EXPECT_EQ(blockSize, (std::vector<int32_t>{K_BSD_BLOCK_SIZE}));
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeIsNegativeScaleDefaultFalse)
{
    // is_negative_scale is optional; when not set its default value (false) should round-trip
    makeFinalized();
    auto desc = getDescriptor();

    bool isNegativeScale = true; // initialise to wrong value to confirm it is overwritten
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &isNegativeScale));

    ASSERT_EQ(elementCount, 1);
    EXPECT_FALSE(isNegativeScale);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeIsNegativeScaleWhenTrue)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    bool isNegativeScale = true;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &isNegativeScale);
    desc->finalize();

    bool retrieved = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                                       HIPDNN_TYPE_BOOLEAN,
                                       1,
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(elementCount, 1);
    EXPECT_TRUE(retrieved);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeComputeType)
{
    auto desc = getDescriptor();
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC});
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
    desc->finalize();

    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                                       HIPDNN_TYPE_DATA_TYPE,
                                       1,
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, HIPDNN_DATA_HALF);
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeBlockSizeQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeIsNegativeScaleQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                                       HIPDNN_TYPE_BOOLEAN,
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
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
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
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       0,
                                       &elementCount,
                                       nullptr));
    ASSERT_EQ(elementCount, 1);

    // Retrieve: use the queried count to allocate and fetch
    std::vector<int32_t> blockSize(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_BLOCK_SIZE,
                                       HIPDNN_TYPE_INT32,
                                       elementCount,
                                       &retrievedCount,
                                       blockSize.data()));
    ASSERT_EQ(retrievedCount, 1);
    EXPECT_EQ(blockSize, (std::vector<int32_t>{K_BSD_BLOCK_SIZE}));
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setAllAttributesExcept();

    HipdnnBackendDescriptor* dummy = nullptr;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
                           HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                           1,
                           nullptr,
                           static_cast<void*>(&dummy)),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
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

TEST_F(TestBlockScaleDequantizeOperationDescriptor, GetAttributeTensorQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_XDESC,
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

    ASSERT_NE(desc->getXDesc(), nullptr);
    ASSERT_NE(desc->getScaleDesc(), nullptr);
    ASSERT_NE(desc->getYDesc(), nullptr);

    ASSERT_EQ(desc->getXDesc()->getData().uid, K_BSD_TENSOR_X_UID);
    ASSERT_EQ(desc->getScaleDesc()->getData().uid, K_BSD_TENSOR_SCALE_UID);
    ASSERT_EQ(desc->getYDesc()->getData().uid, K_BSD_TENSOR_Y_UID);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, ToStringContainsExpectedInfo)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("BlockScaleDequantizeOperationDescriptor"), std::string::npos);
    ASSERT_NE(str.find("x_uid=" + std::to_string(K_BSD_TENSOR_X_UID)), std::string::npos);
    ASSERT_NE(str.find("scale_uid=" + std::to_string(K_BSD_TENSOR_SCALE_UID)), std::string::npos);
    ASSERT_NE(str.find("y_uid=" + std::to_string(K_BSD_TENSOR_Y_UID)), std::string::npos);
    ASSERT_NE(str.find("is_negative_scale=0"), std::string::npos);
    ASSERT_NE(str.find("compute_data_type="), std::string::npos);
}

// =============================================================================
// IGraphOperation Interface Tests
// =============================================================================

TEST_F(TestBlockScaleDequantizeOperationDescriptor, BuildNodeProducesCorrectNodeT)
{
    makeFinalized();

    auto desc = getDescriptor();
    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);
    ASSERT_EQ(node->compute_data_type, DataType::FLOAT);
    ASSERT_EQ(node->attributes.type, NodeAttributes::BlockScaleDequantizeAttributes);

    auto* attrs = node->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    ASSERT_EQ(attrs->x_tensor_uid, K_BSD_TENSOR_X_UID);
    ASSERT_EQ(attrs->scale_tensor_uid, K_BSD_TENSOR_SCALE_UID);
    ASSERT_EQ(attrs->y_tensor_uid, K_BSD_TENSOR_Y_UID);
    ASSERT_EQ(attrs->block_size.size(), 1);
    ASSERT_EQ(attrs->block_size[0], K_BSD_BLOCK_SIZE);
    EXPECT_FALSE(attrs->is_negative_scale);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, BuildNodeWithIsNegativeScaleTrue)
{
    setAllAttributesExcept();
    auto desc = getDescriptor();
    bool isNegativeScale = true;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_NEG_SCALE,
                       HIPDNN_TYPE_BOOLEAN,
                       1,
                       &isNegativeScale);
    desc->finalize();

    auto node = desc->buildNode();
    ASSERT_NE(node, nullptr);

    auto* attrs = node->attributes.AsBlockScaleDequantizeAttributes();
    ASSERT_NE(attrs, nullptr);
    EXPECT_TRUE(attrs->is_negative_scale);
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, BuildNodeWithHalfComputeType)
{
    setAllAttributesExcept({HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC});
    auto desc = getDescriptor();
    auto computeType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_OPERATION_BLOCK_SCALE_DEQUANTIZE_MATH_PREC,
                       HIPDNN_TYPE_DATA_TYPE,
                       1,
                       &computeType);
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

    auto graphOp = _wrapper->tryAsGraphOperation();
    ASSERT_NE(graphOp, nullptr);

    auto desc = getDescriptor();
    auto tensors = graphOp->getTensorDescriptors();
    ASSERT_EQ(tensors.size(), 3);
    EXPECT_EQ(tensors[0], desc->getXDesc());
    EXPECT_EQ(tensors[1], desc->getScaleDesc());
    EXPECT_EQ(tensors[2], desc->getYDesc());
}

TEST_F(TestBlockScaleDequantizeOperationDescriptor, TryAsInterfaceReturnsNullForWrongType)
{
    // TensorDescriptor does not implement IGraphOperation
    auto graphOp = _xDesc->tryAsGraphOperation();
    EXPECT_EQ(graphOp, nullptr);
}
