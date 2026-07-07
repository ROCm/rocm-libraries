// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DescriptorTestUtils.hpp"
#include "HipdnnException.hpp"
#include "TensorDescriptorTestUtils.hpp"
#include "TestMacros.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

#include <array>
#include <cstring>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_backend::test_utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;

class TestTensorDescriptor : public ::testing::Test
{
public:
    std::shared_ptr<TensorDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<TensorDescriptor>();
    }

    void setRequiredAttributes() const
    {
        auto desc = getDescriptor();
        std::vector<int64_t> dims = {1, 3, 32, 32};
        std::vector<int64_t> strides = {3072, 1024, 32, 1};
        auto dataType = HIPDNN_DATA_FLOAT;

        desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
        desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    }

    void makeFinalized() const
    {
        setRequiredAttributes();
        getDescriptor()->finalize();
    }

protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<TensorDescriptor>();
    }

    void TearDown() override
    {
        _wrapper.reset();
    }
};

// =============================================================================
// Lifecycle Tests
// =============================================================================

TEST_F(TestTensorDescriptor, CreateDescriptor)
{
    auto desc = getDescriptor();
    ASSERT_NE(desc, nullptr);
    ASSERT_FALSE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
}

TEST_F(TestTensorDescriptor, FinalizeWithRequiredAttributes)
{
    setRequiredAttributes();
    ASSERT_NO_THROW(getDescriptor()->finalize());
    ASSERT_TRUE(getDescriptor()->isFinalized());
}

TEST_F(TestTensorDescriptor, FinalizeFailsWithoutDimensions)
{
    auto desc = getDescriptor();
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    auto dataType = HIPDNN_DATA_FLOAT;

    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, FinalizeFailsWithoutStrides)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dims = {1, 3, 32, 32};
    auto dataType = HIPDNN_DATA_FLOAT;

    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, FinalizeFailsWithoutDataType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};

    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, FinalizeFailsDimsStridesMismatch)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32}; // Only 3 elements
    auto dataType = HIPDNN_DATA_FLOAT;

    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 3, strides.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    ASSERT_THROW_HIPDNN_STATUS(desc->finalize(), HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - UNIQUE_ID
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeUniqueId)
{
    auto desc = getDescriptor();
    int64_t uid = 42;

    ASSERT_NO_THROW(desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid));

    // Verify via getData()
    ASSERT_EQ(desc->getData().uid, 42);
}

TEST_F(TestTensorDescriptor, SetAttributeUniqueIdWrongElementCount)
{
    auto desc = getDescriptor();
    int64_t uid = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 2, &uid),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeUniqueIdWrongType)
{
    auto desc = getDescriptor();
    int64_t uid = 42;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_CHAR, 1, &uid),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - NAME
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeName)
{
    auto desc = getDescriptor();
    const char* name = "test_tensor";

    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, static_cast<int64_t>(strlen(name)), name));

    ASSERT_EQ(desc->getData().name, "test_tensor");
}

TEST_F(TestTensorDescriptor, SetAttributeNameWrongType)
{
    auto desc = getDescriptor();
    const char* name = "test";

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_INT64, 4, name),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - DATA_TYPE
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeDataType)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_HALF;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType));

    ASSERT_EQ(desc->getData().data_type, hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
}

TEST_F(TestTensorDescriptor, SetAttributeDataTypeWrongElementCount)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 2, &dataType),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeDataTypeWrongType)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_INT64, 1, &dataType),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - DIMENSIONS
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeDimensions)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dims = {2, 64, 112, 112};

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.dims.size(), 4);
    ASSERT_EQ(data.dims[0], 2);
    ASSERT_EQ(data.dims[1], 64);
    ASSERT_EQ(data.dims[2], 112);
    ASSERT_EQ(data.dims[3], 112);
}

TEST_F(TestTensorDescriptor, SetAttributeDimensionsWrongType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> dims = {1, 2, 3, 4};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_CHAR, 4, dims.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - STRIDES
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeStrides)
{
    auto desc = getDescriptor();
    std::vector<int64_t> strides = {802816, 12544, 112, 1};

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data()));

    auto& data = desc->getData();
    ASSERT_EQ(data.strides.size(), 4);
    ASSERT_EQ(data.strides[0], 802816);
    ASSERT_EQ(data.strides[1], 12544);
    ASSERT_EQ(data.strides[2], 112);
    ASSERT_EQ(data.strides[3], 1);
}

TEST_F(TestTensorDescriptor, SetAttributeStridesWrongType)
{
    auto desc = getDescriptor();
    std::vector<int64_t> strides = {1, 2, 3, 4};

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_BOOLEAN, 4, strides.data()),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Tests - IS_VIRTUAL
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeIsVirtual)
{
    auto desc = getDescriptor();
    bool isVirtual = true;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual));

    ASSERT_EQ(desc->getData().virtual_, true);
}

TEST_F(TestTensorDescriptor, SetAttributeIsVirtualWrongElementCount)
{
    auto desc = getDescriptor();
    bool isVirtual = true;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 2, &isVirtual),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeIsVirtualWrongType)
{
    auto desc = getDescriptor();
    bool isVirtual = true;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_INT64, 1, &isVirtual),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// SetAttribute Error Cases
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeFailsNullPointer)
{
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestTensorDescriptor, SetAttributeFailsAfterFinalize)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t uid = 100;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestTensorDescriptor, SetAttributeUnsupported)
{
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

// =============================================================================
// GetAttribute Tests - UNIQUE_ID
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeUniqueId)
{
    auto desc = getDescriptor();
    int64_t uid = 123;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    setRequiredAttributes();
    desc->finalize();

    int64_t retrievedUid = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &elementCount, &retrievedUid));

    ASSERT_EQ(retrievedUid, 123);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Tests - NAME
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeName)
{
    auto desc = getDescriptor();
    const char* name = "my_tensor";
    desc->setAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, static_cast<int64_t>(strlen(name)), name);
    setRequiredAttributes();
    desc->finalize();

    std::array<char, 64> buffer = {0};
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, 64, &elementCount, buffer.data()));

    ASSERT_STREQ(buffer.data(), "my_tensor");
}

TEST_F(TestTensorDescriptor, GetAttributeNamePartialCopy)
{
    auto desc = getDescriptor();
    const char* name = "this_is_a_long_tensor_name";
    desc->setAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, static_cast<int64_t>(strlen(name)), name);
    setRequiredAttributes();
    desc->finalize();

    std::array<char, 10> buffer = {0};
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, 10, &elementCount, buffer.data()));

    // Should copy 9 characters + null terminator within the 10-byte buffer
    ASSERT_EQ(elementCount, 10);
    ASSERT_STREQ(buffer.data(), "this_is_a");
}

// =============================================================================
// GetAttribute Tests - DATA_TYPE
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeDataType)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    hipdnnDataType_t retrievedType = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrievedType));

    ASSERT_EQ(retrievedType, HIPDNN_DATA_BFLOAT16);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Tests - DIMENSIONS
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeDimensions)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int64_t> dims(4);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, &elementCount, dims.data()));

    ASSERT_EQ(elementCount, 4);
    ASSERT_EQ(dims[0], 1);
    ASSERT_EQ(dims[1], 3);
    ASSERT_EQ(dims[2], 32);
    ASSERT_EQ(dims[3], 32);
}

TEST_F(TestTensorDescriptor, GetAttributeDimensionsPartialCopy)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int64_t> dims(2);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 2, &elementCount, dims.data()));

    // Should only copy 2 elements
    ASSERT_EQ(elementCount, 2);
    ASSERT_EQ(dims[0], 1);
    ASSERT_EQ(dims[1], 3);
}

// =============================================================================
// GetAttribute Tests - STRIDES
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeStrides)
{
    makeFinalized();
    auto desc = getDescriptor();

    std::vector<int64_t> strides(4);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, &elementCount, strides.data()));

    ASSERT_EQ(elementCount, 4);
    ASSERT_EQ(strides[0], 3072);
    ASSERT_EQ(strides[1], 1024);
    ASSERT_EQ(strides[2], 32);
    ASSERT_EQ(strides[3], 1);
}

// =============================================================================
// GetAttribute Tests - IS_VIRTUAL
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeIsVirtual)
{
    auto desc = getDescriptor();
    bool isVirtual = true;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &isVirtual);
    setRequiredAttributes();
    desc->finalize();

    bool retrievedVirtual = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 1, &elementCount, &retrievedVirtual));

    ASSERT_EQ(retrievedVirtual, true);
    ASSERT_EQ(elementCount, 1);
}

// =============================================================================
// GetAttribute Error Cases
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeFailsBeforeFinalize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();

    int64_t uid = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, nullptr, &uid),
        HIPDNN_STATUS_NOT_INITIALIZED);
}

TEST_F(TestTensorDescriptor, GetAttributeFailsNullPointer)
{
    makeFinalized();
    auto desc = getDescriptor();

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, nullptr, nullptr),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

TEST_F(TestTensorDescriptor, GetAttributeUnsupported)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t dummy = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_ENGINEHEUR_MODE, HIPDNN_TYPE_INT64, 1, nullptr, &dummy),
        HIPDNN_STATUS_NOT_SUPPORTED);
}

TEST_F(TestTensorDescriptor, GetAttributeWrongType)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t uid = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_CHAR, 1, nullptr, &uid),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, GetAttributeQueryFailsNullElementCount)
{
    makeFinalized();
    auto desc = getDescriptor();
    int64_t uid = 0;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 0, nullptr, &uid),
        HIPDNN_STATUS_BAD_PARAM_NULL_POINTER);
}

// =============================================================================
// GetAttribute Tests - elementCount output with nullptr
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeElementCountNullable)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t uid = 0;
    // elementCount is nullptr - should still work
    ASSERT_NO_THROW(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, nullptr, &uid));
}

// =============================================================================
// GetAttribute Tests - Two-Call Query Pattern
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeDimensionsQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, 4);
}

TEST_F(TestTensorDescriptor, GetAttributeStridesQueryReturnsSize)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, 4);
}

TEST_F(TestTensorDescriptor, GetAttributeUniqueIdQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTensorDescriptor, GetAttributeDataTypeQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTensorDescriptor, GetAttributeIsVirtualQueryReturnsOne)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_IS_VIRTUAL, HIPDNN_TYPE_BOOLEAN, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, 1);
}

TEST_F(TestTensorDescriptor, GetAttributeDimensionsQueryThenRetrieve)
{
    makeFinalized();
    auto desc = getDescriptor();

    // First call: query the size
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, 4);

    // Second call: retrieve the data
    std::vector<int64_t> dims(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS,
                                       HIPDNN_TYPE_INT64,
                                       elementCount,
                                       &retrievedCount,
                                       dims.data()));

    ASSERT_EQ(retrievedCount, 4);
    ASSERT_EQ(dims[0], 1);
    ASSERT_EQ(dims[1], 3);
    ASSERT_EQ(dims[2], 32);
    ASSERT_EQ(dims[3], 32);
}

TEST_F(TestTensorDescriptor, GetAttributeNameQueryReturnsSize)
{
    auto desc = getDescriptor();
    const char* name = "test_tensor";
    desc->setAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, static_cast<int64_t>(strlen(name)), name);
    setRequiredAttributes();
    desc->finalize();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, static_cast<int64_t>(strlen("test_tensor") + 1));
}

TEST_F(TestTensorDescriptor, GetAttributeNameQueryThenRetrieve)
{
    auto desc = getDescriptor();
    const char* name = "test_tensor";
    desc->setAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, static_cast<int64_t>(strlen(name)), name);
    setRequiredAttributes();
    desc->finalize();

    // First call: query the size
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_NAME_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, static_cast<int64_t>(strlen("test_tensor") + 1));

    // Second call: retrieve the data
    std::vector<char> buffer(static_cast<size_t>(elementCount));
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_NAME_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       elementCount,
                                       &retrievedCount,
                                       buffer.data()));

    ASSERT_STREQ(buffer.data(), "test_tensor");
    ASSERT_EQ(retrievedCount, elementCount);
}

TEST_F(TestTensorDescriptor, GetAttributeValueFloat32QueryReturnsSize)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    float setVal = 1.5f;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &setVal);
    desc->finalize();

    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));

    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(float)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueFloat32QueryThenRetrieve)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    float setVal = 1.5f;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &setVal);
    desc->finalize();

    // First call: query the size
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr));
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(float)));

    // Second call: retrieve the data
    float retrieved = 0.0f;
    int64_t retrievedCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, elementCount, &retrievedCount, &retrieved));

    ASSERT_FLOAT_EQ(retrieved, 1.5f);
    ASSERT_EQ(retrievedCount, static_cast<int64_t>(sizeof(float)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueQueryFailsWhenNotSet)
{
    makeFinalized();
    auto desc = getDescriptor();

    int64_t elementCount = 0;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, 0, &elementCount, nullptr),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, IsByValueReturnsFalseWhenNoValueSet)
{
    makeFinalized();
    auto desc = getDescriptor();

    bool isByValue = true;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_IS_BY_VALUE, HIPDNN_TYPE_BOOLEAN, 1, &count, &isByValue));
    EXPECT_FALSE(isByValue);
}

TEST_F(TestTensorDescriptor, IsByValueReturnsTrueWhenValueSet)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    float val = 1.5f;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val);
    desc->finalize();

    bool isByValue = false;
    int64_t count = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_IS_BY_VALUE, HIPDNN_TYPE_BOOLEAN, 1, &count, &isByValue));
    EXPECT_TRUE(isByValue);
}

// =============================================================================
// SetAttribute Tests - VALUE (pass-by-value)
// =============================================================================

TEST_F(TestTensorDescriptor, SetAttributeValueFloat32)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    float val = 1.5f;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val));

    auto* stored = desc->getData().value.AsFloat32Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_FLOAT_EQ(stored->value(), 1.5f);
}

TEST_F(TestTensorDescriptor, SetAttributeValueDouble)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_DOUBLE;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    double val = 2.718281828;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(double), &val));

    auto* stored = desc->getData().value.AsFloat64Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_DOUBLE_EQ(stored->value(), 2.718281828);
}

TEST_F(TestTensorDescriptor, SetAttributeValueInt32)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT32;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    int32_t val = 42;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int32_t), &val));

    auto* stored = desc->getData().value.AsInt32Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), 42);
}

TEST_F(TestTensorDescriptor, SetAttributeValueInt64)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT64;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    int64_t val = 123456789012345LL;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int64_t), &val));

    auto* stored = desc->getData().value.AsInt64Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), 123456789012345LL);
}

TEST_F(TestTensorDescriptor, SetAttributeValueBoolean)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BOOLEAN;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    bool val = true;

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(bool), &val));

    auto* stored = desc->getData().value.AsBoolValue();
    ASSERT_NE(stored, nullptr);
    ASSERT_TRUE(stored->value());
}

TEST_F(TestTensorDescriptor, SetGetValueBoolean)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BOOLEAN;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    bool setVal = true;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(bool), &setVal));

    auto* stored = desc->getData().value.AsBoolValue();
    ASSERT_NE(stored, nullptr);
    ASSERT_TRUE(stored->value());

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    bool retrieved = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(bool), &elementCount, &retrieved));
    ASSERT_TRUE(retrieved);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(bool)));
}

TEST_F(TestTensorDescriptor, SetAttributeValueFloat16)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    hipdnn_data_sdk::types::half val(1.5f);

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(val), &val));

    auto* stored = desc->getData().value.AsFloat16Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_FLOAT_EQ(stored->value(), 1.5f);
}

TEST_F(TestTensorDescriptor, SetAttributeValueBFloat16)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    hipdnn_data_sdk::types::bfloat16 val(1.5f);

    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(val), &val));

    auto* stored = desc->getData().value.AsBFloat16Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_FLOAT_EQ(stored->value(), 1.5f);
}

TEST_F(TestTensorDescriptor, SetAttributeValueWrongElementCount)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    float val = 1.0f;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, 2, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeValueUnsupportedType)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    bool val = true;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_BOOLEAN, 1, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeValueFailsWithoutDataType)
{
    auto desc = getDescriptor();
    float val = 1.0f;

    ASSERT_THROW_HIPDNN_STATUS(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, SetAttributeValueOverwritesPrevious)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    float val1 = 1.0f;
    float val2 = 2.0f;

    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val1);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val2);

    auto* stored = desc->getData().value.AsFloat32Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_FLOAT_EQ(stored->value(), 2.0f);
}

TEST_F(TestTensorDescriptor, SetAttributeValueCopiesData)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FLOAT;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    {
        float val = 3.14f;
        desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &val);
    }
    // val is out of scope — descriptor must have its own copy
    auto* stored = desc->getData().value.AsFloat32Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_FLOAT_EQ(stored->value(), 3.14f);
}

// =============================================================================
// GetAttribute Tests - VALUE
// =============================================================================

TEST_F(TestTensorDescriptor, GetAttributeValueFloat32)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    float setVal = 1.5f;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &setVal);
    desc->finalize();

    float retrieved = 0.0f;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &elementCount, &retrieved));

    ASSERT_FLOAT_EQ(retrieved, 1.5f);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(float)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueDouble)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_DOUBLE;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    double setVal = 2.718281828;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(double), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    double retrieved = 0.0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(double), &elementCount, &retrieved));

    ASSERT_DOUBLE_EQ(retrieved, 2.718281828);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(double)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueInt32)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT32;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    int32_t setVal = 42;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int32_t), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    int32_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(int32_t),
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, 42);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(int32_t)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueInt64)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT64;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    int64_t setVal = 123456789012345LL;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int64_t), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    int64_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(int64_t),
                                       &elementCount,
                                       &retrieved));

    ASSERT_EQ(retrieved, 123456789012345LL);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(int64_t)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueBoolean)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BOOLEAN;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    bool setVal = true;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(bool), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    bool retrieved = false;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(bool), &elementCount, &retrieved));

    ASSERT_TRUE(retrieved);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(bool)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueFloat16)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_HALF;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    hipdnn_data_sdk::types::half setVal(1.5f);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(setVal), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    hipdnn_data_sdk::types::half retrieved(0.0f);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(retrieved),
                                       &elementCount,
                                       &retrieved));

    ASSERT_FLOAT_EQ(static_cast<float>(retrieved), 1.5f);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(retrieved)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueBFloat16)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_BFLOAT16;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);
    hipdnn_data_sdk::types::bfloat16 setVal(1.5f);
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(setVal), &setVal);

    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    hipdnn_data_sdk::types::bfloat16 retrieved(0.0f);
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(retrieved),
                                       &elementCount,
                                       &retrieved));

    ASSERT_FLOAT_EQ(static_cast<float>(retrieved), 1.5f);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(retrieved)));
}

TEST_F(TestTensorDescriptor, GetAttributeValueNotSet)
{
    makeFinalized();
    auto desc = getDescriptor();

    float val = 0.0f;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(
            HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), nullptr, &val),
        HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptor, GetAttributeValueWrongAttributeType)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    float setVal = 1.0f;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(float), &setVal);
    desc->finalize();

    float retrieved = 0.0f;
    ASSERT_THROW_HIPDNN_STATUS(
        desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_FLOAT, 1, nullptr, &retrieved),
        HIPDNN_STATUS_BAD_PARAM);
}

// =============================================================================
// ToString Test
// =============================================================================

TEST_F(TestTensorDescriptor, ToStringContainsExpectedInfo)
{
    auto desc = getDescriptor();
    int64_t uid = 999;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &uid);
    setRequiredAttributes();

    const std::string str = desc->toString();
    ASSERT_NE(str.find("TensorDescriptor"), std::string::npos);
    ASSERT_NE(str.find("999"), std::string::npos);
}

TEST_F(TestTensorDescriptor, ToStringValueBool)
{
    auto desc = getDescriptor();
    setRequiredAttributes();
    auto dataType = HIPDNN_DATA_BOOLEAN;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    bool valTrue = true;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(valTrue), &valTrue);
    ASSERT_NE(desc->toString().find("value=true"), std::string::npos);

    bool valFalse = false;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(valFalse), &valFalse);
    ASSERT_NE(desc->toString().find("value=false"), std::string::npos);
}

TEST_F(TestTensorDescriptor, ToStringHandlesUnsetDataType)
{
    auto desc = getDescriptor();
    // Do not set data type — it remains UNSET

    std::string str;
    ASSERT_NO_THROW(str = desc->toString());
    ASSERT_NE(str.find("UNSET"), std::string::npos);
}

// =============================================================================
// Data Type Round-Trip Tests
// =============================================================================

struct DataTypeRoundTripParam
{
    hipdnnDataType_t apiType;
    DataType sdkType;
    const char* name;
};

class TestTensorDescriptorDataTypeRoundTrip
    : public ::testing::TestWithParam<DataTypeRoundTripParam>
{
protected:
    std::unique_ptr<HipdnnBackendDescriptor> _wrapper = nullptr;

    void SetUp() override
    {
        _wrapper = createDescriptor<TensorDescriptor>();
    }

    std::shared_ptr<TensorDescriptor> getDescriptor() const
    {
        return _wrapper->asDescriptor<TensorDescriptor>();
    }
};

TEST_P(TestTensorDescriptorDataTypeRoundTrip, SetAndGetDataType)
{
    auto param = GetParam();
    auto desc = getDescriptor();

    // Set the data type
    auto dataType = param.apiType;
    ASSERT_NO_THROW(
        desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType));

    // Verify internal SDK type
    ASSERT_EQ(desc->getData().data_type, param.sdkType);

    // Finalize so we can call getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    // Get the data type back
    hipdnnDataType_t retrieved = HIPDNN_DATA_FLOAT;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &elementCount, &retrieved));

    ASSERT_EQ(retrieved, param.apiType);
    ASSERT_EQ(elementCount, 1);
}

INSTANTIATE_TEST_SUITE_P(
    AllDataTypes,
    TestTensorDescriptorDataTypeRoundTrip,
    ::testing::Values(DataTypeRoundTripParam{HIPDNN_DATA_FLOAT, DataType::FLOAT, "Float"},
                      DataTypeRoundTripParam{HIPDNN_DATA_DOUBLE, DataType::DOUBLE, "Double"},
                      DataTypeRoundTripParam{HIPDNN_DATA_HALF, DataType::HALF, "Half"},
                      DataTypeRoundTripParam{HIPDNN_DATA_INT8, DataType::INT8, "Int8"},
                      DataTypeRoundTripParam{HIPDNN_DATA_INT32, DataType::INT32, "Int32"},
                      DataTypeRoundTripParam{HIPDNN_DATA_UINT8, DataType::UINT8, "Uint8"},
                      DataTypeRoundTripParam{HIPDNN_DATA_BFLOAT16, DataType::BFLOAT16, "BFloat16"},
                      DataTypeRoundTripParam{HIPDNN_DATA_FP8_E4M3, DataType::FP8_E4M3, "Fp8E4M3"},
                      DataTypeRoundTripParam{HIPDNN_DATA_FP8_E5M2, DataType::FP8_E5M2, "Fp8E5M2"},
                      DataTypeRoundTripParam{HIPDNN_DATA_INT64, DataType::INT64, "Int64"},
                      DataTypeRoundTripParam{HIPDNN_DATA_BOOLEAN, DataType::BOOLEAN, "Boolean"}),
    [](const ::testing::TestParamInfo<DataTypeRoundTripParam>& info) { return info.param.name; });

// =============================================================================
// Pass-By-Value Native Type Tests
// =============================================================================

TEST_F(TestTensorDescriptor, SetGetValueUint8)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_UINT8;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    uint8_t setVal = 200;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(uint8_t), &setVal));

    auto* stored = desc->getData().value.AsFloat8Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), 200);

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    uint8_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(uint8_t),
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved, 200);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(uint8_t)));
}

TEST_F(TestTensorDescriptor, SetGetValueInt8)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT8;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    int8_t setVal = -42;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int8_t), &setVal));

    auto* stored = desc->getData().value.AsFloat8Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), static_cast<uint8_t>(static_cast<int8_t>(-42)));

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    int8_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int8_t), &elementCount, &retrieved));
    ASSERT_EQ(retrieved, -42);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(int8_t)));
}

TEST_F(TestTensorDescriptor, SetGetValueFp8E4M3)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FP8_E4M3;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    auto setVal = hipdnn_data_sdk::types::fp8_e4m3(1.5f);
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(setVal), &setVal));

    auto* stored = desc->getData().value.AsFloat8Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), setVal.data);

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    hipdnn_data_sdk::types::fp8_e4m3 retrieved;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(retrieved),
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved.data, setVal.data);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(retrieved)));
}

TEST_F(TestTensorDescriptor, SetGetValueFp8E5M2)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_FP8_E5M2;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    auto setVal = hipdnn_data_sdk::types::fp8_e5m2(1.5f);
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(setVal), &setVal));

    auto* stored = desc->getData().value.AsFloat8Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), setVal.data);

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    hipdnn_data_sdk::types::fp8_e5m2 retrieved;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(retrieved),
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved.data, setVal.data);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(retrieved)));
}

TEST_F(TestTensorDescriptor, SetGetValueInt64)
{
    auto desc = getDescriptor();
    auto dataType = HIPDNN_DATA_INT64;
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dataType);

    int64_t setVal = 9876543210LL;
    ASSERT_NO_THROW(desc->setAttribute(
        HIPDNN_ATTR_TENSOR_VALUE_EXT, HIPDNN_TYPE_CHAR, sizeof(int64_t), &setVal));

    auto* stored = desc->getData().value.AsInt64Value();
    ASSERT_NE(stored, nullptr);
    ASSERT_EQ(stored->value(), 9876543210LL);

    // Round-trip through getAttribute
    std::vector<int64_t> dims = {1, 3, 32, 32};
    std::vector<int64_t> strides = {3072, 1024, 32, 1};
    desc->setAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, dims.data());
    desc->setAttribute(HIPDNN_ATTR_TENSOR_STRIDES, HIPDNN_TYPE_INT64, 4, strides.data());
    desc->finalize();

    int64_t retrieved = 0;
    int64_t elementCount = 0;
    ASSERT_NO_THROW(desc->getAttribute(HIPDNN_ATTR_TENSOR_VALUE_EXT,
                                       HIPDNN_TYPE_CHAR,
                                       sizeof(int64_t),
                                       &elementCount,
                                       &retrieved));
    ASSERT_EQ(retrieved, 9876543210LL);
    ASSERT_EQ(elementCount, static_cast<int64_t>(sizeof(int64_t)));
}
