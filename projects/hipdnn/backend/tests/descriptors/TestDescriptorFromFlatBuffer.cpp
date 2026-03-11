// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "TestMacros.hpp"
#include "descriptors/TensorDescriptor.hpp"
#include "hipdnn_backend.h"

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/constants/ConvFpropConstants.hpp>
#include <hipdnn_test_sdk/utilities/ToVec.hpp>

#include <memory>
#include <vector>

using namespace hipdnn_backend;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_tests::constants;
using hipdnn_tests::toVec;

// =============================================================================
// TensorDescriptor::fromFlatBuffer() Tests
// =============================================================================

class TestTensorDescriptorFromFlatBuffer : public ::testing::Test
{
protected:
    // Creates a TensorAttributesT with standard test values
    static TensorAttributesT createStandardTensorAttrs(int64_t uid,
                                                       DataType dataType = DataType::FLOAT)
    {
        TensorAttributesT attrs;
        attrs.uid = uid;
        attrs.data_type = dataType;
        attrs.dims = toVec(K_TENSOR_X_DIMS);
        attrs.strides = toVec(K_TENSOR_X_STRIDES);
        attrs.virtual_ = false;
        return attrs;
    }
};

TEST_F(TestTensorDescriptorFromFlatBuffer, CreatesValidFinalizedDescriptor)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    ASSERT_EQ(desc->getType(), HIPDNN_BACKEND_TENSOR_DESCRIPTOR);
    EXPECT_EQ(desc->getData().uid, K_TENSOR_X_UID);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PopulatesAllFieldsCorrectly)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().uid, K_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getData().dims, toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getData().strides, toVec(K_TENSOR_X_STRIDES));
    EXPECT_FALSE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, MoveOverloadPopulatesAllFields)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(std::move(attrs));

    ASSERT_NE(desc, nullptr);
    ASSERT_TRUE(desc->isFinalized());
    EXPECT_EQ(desc->getData().uid, K_TENSOR_X_UID);
    EXPECT_EQ(desc->getData().data_type, DataType::FLOAT);
    EXPECT_EQ(desc->getData().dims, toVec(K_TENSOR_X_DIMS));
    EXPECT_EQ(desc->getData().strides, toVec(K_TENSOR_X_STRIDES));
    EXPECT_FALSE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PreservesVirtualFlag)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    attrs.virtual_ = true;
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_TRUE(desc->getData().virtual_);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, GetAttributeWorksAfterFromFlatBuffer)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    // Verify getAttribute returns the correct UID
    int64_t uid = 0;
    int64_t elementCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_UNIQUE_ID, HIPDNN_TYPE_INT64, 1, &elementCount, &uid);
    ASSERT_EQ(uid, K_TENSOR_X_UID);
    ASSERT_EQ(elementCount, 1);

    // Verify getAttribute returns the correct data type
    hipdnnDataType_t dataType = {};
    int64_t dtCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_DATA_TYPE, HIPDNN_TYPE_DATA_TYPE, 1, &dtCount, &dataType);
    ASSERT_EQ(dataType, HIPDNN_DATA_FLOAT);
    ASSERT_EQ(dtCount, 1);

    // Verify getAttribute returns the correct dims
    std::vector<int64_t> dims(4);
    int64_t dimCount = 0;
    desc->getAttribute(HIPDNN_ATTR_TENSOR_DIMENSIONS, HIPDNN_TYPE_INT64, 4, &dimCount, dims.data());
    ASSERT_EQ(dimCount, 4);
    ASSERT_EQ(dims, toVec(K_TENSOR_X_DIMS));
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithMissingDims)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    attrs.data_type = DataType::FLOAT;
    // dims left empty
    attrs.strides = {1};

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithMissingStrides)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    attrs.data_type = DataType::FLOAT;
    attrs.dims = {1};
    // strides left empty

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, FailsWithUnsetDataType)
{
    TensorAttributesT attrs;
    attrs.uid = 1;
    // data_type defaults to UNSET
    attrs.dims = {1};
    attrs.strides = {1};

    ASSERT_THROW_HIPDNN_STATUS(TensorDescriptor::fromFlatBuffer(attrs), HIPDNN_STATUS_BAD_PARAM);
}

TEST_F(TestTensorDescriptorFromFlatBuffer, PreservesName)
{
    auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID);
    attrs.name = "test_tensor";
    auto desc = TensorDescriptor::fromFlatBuffer(attrs);

    ASSERT_EQ(desc->getData().name, "test_tensor");
}

TEST_F(TestTensorDescriptorFromFlatBuffer, RoundTripWithMultipleDataTypes)
{
    for(auto dt : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE})
    {
        auto attrs = createStandardTensorAttrs(K_TENSOR_X_UID, dt);
        auto desc = TensorDescriptor::fromFlatBuffer(attrs);

        ASSERT_EQ(desc->getData().data_type, dt) << "Failed for DataType: " << EnumNameDataType(dt);
        ASSERT_TRUE(desc->isFinalized()) << "Not finalized for DataType: " << EnumNameDataType(dt);
    }
}
