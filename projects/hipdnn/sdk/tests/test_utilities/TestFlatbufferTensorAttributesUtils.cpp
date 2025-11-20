// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/test_utilities/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestFlatbufferTensorAttributesUtils, UnpackTensorAttributes)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims = {1, 3, 224, 224};
    std::vector<int64_t> strides = {150528, 50176, 224, 1};
    auto attributeOffset
        = CreateTensorAttributesDirect(builder, 1, "x", DataType::FLOAT, &strides, &dims);
    builder.Finish(attributeOffset);

    auto tensorAttr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());
    auto unpacked = unpackTensorAttributes(*tensorAttr);

    EXPECT_EQ(unpacked.uid, 1);
    EXPECT_EQ(unpacked.name, "x");
    EXPECT_EQ(unpacked.data_type, DataType::FLOAT);
    EXPECT_EQ(unpacked.dims, dims);
    EXPECT_EQ(unpacked.strides, strides);
}

TEST(TestFlatbufferTensorAttributesUtils, CreateShallowTensor)
{
    TensorAttributesT attr;
    attr.uid = 2;
    attr.name = "y";
    attr.data_type = DataType::FLOAT;
    attr.dims = {2, 2};
    attr.strides = {2, 1};

    std::array<float, 4> data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto tensor = createShallowTensor<float>(attr, data.data());

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), attr.dims);
    EXPECT_EQ(tensor->strides(), attr.strides);
    EXPECT_EQ(tensor->memory().hostData(), data.data());
}
