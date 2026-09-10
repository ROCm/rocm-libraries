// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;

TEST(TestFlatbufferTensorAttributesUtils, UnpackTensorAttributes)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {1, 3, 224, 224};
    const std::vector<int64_t> strides = {150528, 50176, 224, 1};
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

TEST(TestFlatbufferTensorAttributesUtils, BindShallowTensorReturnsMatchingTensor)
{
    TensorAttributesT attr;
    attr.uid = 7;
    attr.name = "z";
    attr.data_type = DataType::FLOAT;
    attr.dims = {2, 2};
    attr.strides = {2, 1};

    std::array<float, 4> data = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::unordered_map<int64_t, void*> variantPack{{7, data.data()}};

    auto tensor = bindShallowTensor<float>(attr, variantPack);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), attr.dims);
    EXPECT_EQ(tensor->strides(), attr.strides);
    EXPECT_EQ(tensor->memory().hostData(), data.data());
}

TEST(TestFlatbufferTensorAttributesUtils, BindShallowTensorThrowsOnMissingUid)
{
    TensorAttributesT attr;
    attr.uid = 8;
    attr.data_type = DataType::FLOAT;
    attr.dims = {2, 2};
    attr.strides = {2, 1};

    const std::unordered_map<int64_t, void*> variantPack; // uid 8 absent

    EXPECT_THROW(bindShallowTensor<float>(attr, variantPack), std::out_of_range);
}

TEST(TestFlatbufferTensorAttributesUtils, BindOptionalShallowTensorNulloptYieldsNullptr)
{
    const std::optional<TensorAttributesT> attr = std::nullopt;
    const std::unordered_map<int64_t, void*> variantPack;

    auto tensor = bindOptionalShallowTensor<float>(attr, variantPack);

    EXPECT_EQ(tensor, nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, BindOptionalShallowTensorPresentBinds)
{
    TensorAttributesT attr;
    attr.uid = 9;
    attr.data_type = DataType::FLOAT;
    attr.dims = {3};
    attr.strides = {1};

    std::array<float, 3> data = {1.0f, 2.0f, 3.0f};
    const std::unordered_map<int64_t, void*> variantPack{{9, data.data()}};

    auto tensor
        = bindOptionalShallowTensor<float>(std::optional<TensorAttributesT>(attr), variantPack);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), attr.dims);
    EXPECT_EQ(tensor->memory().hostData(), data.data());
}

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorBoolean)
{
    const std::vector<int64_t> dims = {2, 2};
    const std::vector<int64_t> strides = {2, 1};

    auto tensor = createTensor(DataType::BOOLEAN, dims, strides);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), dims);
    EXPECT_EQ(tensor->strides(), strides);
    EXPECT_EQ(tensor->elementSize(), sizeof(bool));
}
