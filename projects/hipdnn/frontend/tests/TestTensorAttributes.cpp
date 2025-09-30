// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestTensorAttributes, DefaultConstructor)
{
    TensorAttributes tensor;
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_EQ(tensor.get_name(), "");
    EXPECT_EQ(tensor.get_data_type(), DataType::NOT_SET);
    EXPECT_TRUE(tensor.get_stride().empty());
    EXPECT_TRUE(tensor.get_dim().empty());
    EXPECT_EQ(tensor.get_volume(), 1);
    EXPECT_FALSE(tensor.get_is_virtual());
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TestTensorAttributes, SetAndGetUid)
{
    TensorAttributes tensor;
    tensor.set_uid(42);
    EXPECT_EQ(tensor.get_uid(), 42);
    EXPECT_TRUE(tensor.has_uid());

    tensor.clear_uid();
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TestTensorAttributes, SetAndGetName)
{
    TensorAttributes tensor;
    tensor.set_name("TestTensor");
    EXPECT_EQ(tensor.get_name(), "TestTensor");
}

TEST(TestTensorAttributes, SetAndGetDataType)
{
    TensorAttributes tensor;
    tensor.set_data_type(DataType::FLOAT);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);
}

TEST(TestTensorAttributes, SetAndGetStride)
{
    TensorAttributes tensor;
    tensor.set_stride({1, 2, 3});
    EXPECT_EQ(tensor.get_stride(), std::vector<int64_t>({1, 2, 3}));
}

TEST(TestTensorAttributes, SetAndGetDim)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_EQ(tensor.get_dim(), std::vector<int64_t>({4, 5, 6}));
    EXPECT_EQ(tensor.get_volume(), 4 * 5 * 6);
}

TEST(TestTensorAttributes, SetAndGetIsVirtual)
{
    TensorAttributes tensor;
    tensor.set_is_virtual(true);
    EXPECT_TRUE(tensor.get_is_virtual());

    tensor.set_is_virtual(false);
    EXPECT_FALSE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetOutput)
{
    TensorAttributes tensor;
    tensor.set_output(true);
    EXPECT_FALSE(tensor.get_is_virtual());

    tensor.set_output(false);
    EXPECT_TRUE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetFromGraphAttributes)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::FLOAT);
    graphAttributes.set_intermediate_data_type(DataType::HALF);

    TensorAttributes tensor;
    tensor.set_is_virtual(false).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);

    tensor.set_data_type(DataType::NOT_SET);
    tensor.set_is_virtual(true).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::HALF);
}

TEST(TestTensorAttributes, PackAttributes)
{
    TensorAttributes tensor;
    tensor.set_uid(1)
        .set_name("PackedTensor")
        .set_data_type(DataType::FLOAT)
        .set_stride({1, 2, 3})
        .set_dim({4, 5, 6})
        .set_is_virtual(true);

    flatbuffers::FlatBufferBuilder builder;
    auto packed = tensor.pack_attributes(builder);
    builder.Finish(packed);

    auto bufferPointer = builder.GetBufferPointer();
    auto tensorAttributesFlatbuffer
        = flatbuffers::GetRoot<hipdnn_sdk::data_objects::TensorAttributes>(bufferPointer);
    auto unpacked = std::unique_ptr<hipdnn_sdk::data_objects::TensorAttributesT>(
        tensorAttributesFlatbuffer->UnPack());

    EXPECT_EQ(unpacked->uid, 1);
    EXPECT_EQ(unpacked->name, "PackedTensor");
    EXPECT_EQ(unpacked->data_type, hipdnn_sdk::data_objects::DataType::FLOAT);
    EXPECT_EQ(unpacked->strides, std::vector<int64_t>({1, 2, 3}));
    EXPECT_EQ(unpacked->dims, std::vector<int64_t>({4, 5, 6}));
    EXPECT_TRUE(unpacked->virtual_);
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveValidCase)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 6, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveEmptyDims)
{
    TensorAttributes tensor;
    // No dimensions set
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveZeroDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 0, 6}); // Zero in middle dimension
    tensor.set_stride({0, 6, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveNegativeDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({4, -5, 6}); // Negative dimension
    tensor.set_stride({30, 6, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveZeroStride)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 0, 1}); // Zero stride
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveNegativeStride)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, -6, 1}); // Negative stride
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveStrideSizeMismatch)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({30, 6}); // Only 2 strides for 3 dimensions
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveMoreStridesThanDims)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5});
    tensor.set_stride({20, 5, 1}); // 3 strides for 2 dimensions
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveAllZeroDims)
{
    TensorAttributes tensor;
    tensor.set_dim({0, 0, 0});
    tensor.set_stride({0, 0, 0});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveAllNegativeDims)
{
    TensorAttributes tensor;
    tensor.set_dim({-1, -2, -3});
    tensor.set_stride({6, 3, 1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveMixedInvalidValues)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 0, -6}); // Mix of valid, zero, and negative
    tensor.set_stride({0, -6, 1}); // Mix of zero, negative, and valid
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveSingleDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({10});
    tensor.set_stride({1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveSingleZeroDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({0});
    tensor.set_stride({1});
    EXPECT_FALSE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositiveLargeDimensions)
{
    TensorAttributes tensor;
    tensor.set_dim({1024, 2048, 4096});
    tensor.set_stride({8388608, 4096, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsAndStridesSetAndPositive5DValidCase)
{
    TensorAttributes tensor;
    tensor.set_dim({2, 3, 4, 5, 6});
    tensor.set_stride({360, 120, 30, 6, 1});
    EXPECT_TRUE(tensor.validate_dims_and_strides_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveValidCase)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveEmptyDims)
{
    TensorAttributes tensor;
    // No dimensions set
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveZeroDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 0, 6}); // Zero in middle dimension
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveNegativeDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({4, -5, 6}); // Negative dimension
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveAllZeroDims)
{
    TensorAttributes tensor;
    tensor.set_dim({0, 0, 0});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveAllNegativeDims)
{
    TensorAttributes tensor;
    tensor.set_dim({-1, -2, -3});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveMixedInvalidValues)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 0, -6}); // Mix of valid, zero, and negative
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveSingleDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({10});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveSingleZeroDimension)
{
    TensorAttributes tensor;
    tensor.set_dim({0});
    EXPECT_FALSE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveLargeDimensions)
{
    TensorAttributes tensor;
    tensor.set_dim({1024, 2048, 4096});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositive5DValidCase)
{
    TensorAttributes tensor;
    tensor.set_dim({2, 3, 4, 5, 6});
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}

TEST(TestTensorAttributes, ValidateDimsSetAndPositiveWithStridesIgnored)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({0, -1, 2}); // Invalid strides should be ignored
    EXPECT_TRUE(tensor.validate_dims_set_and_positive());
}
