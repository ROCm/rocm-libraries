// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp4Tensor.hpp>
#include <hipdnn_data_sdk/utilities/PackedFp6Tensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;

using hipdnn_data_sdk::utilities::PackedFp4Tensor;
using hipdnn_data_sdk::utilities::PackedFp6Tensor;
using hipdnn_data_sdk::utilities::Tensor;

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

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorFp4UnpackedByDefault)
{
    const std::vector<int64_t> dims = {2, 4};
    const std::vector<int64_t> strides = {4, 1};

    auto tensor = createTensor(DataType::FP4_E2M1, dims, strides);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), dims);
    EXPECT_EQ(tensor->strides(), strides);
    EXPECT_NE(dynamic_cast<Tensor<hipdnn_data_sdk::types::fp4_e2m1>*>(tensor.get()), nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorFp4PackedWhenRequested)
{
    const std::vector<int64_t> dims = {2, 4};
    const std::vector<int64_t> strides = {4, 1};

    auto tensor = createTensor(DataType::FP4_E2M1, dims, strides, /*packSubByteElements=*/true);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->dims(), dims);
    EXPECT_EQ(tensor->strides(), strides);
    EXPECT_NE(dynamic_cast<PackedFp4Tensor*>(tensor.get()), nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorFp6UnpackedByDefault)
{
    const std::vector<int64_t> dims = {2, 4};
    const std::vector<int64_t> strides = {4, 1};

    auto e2m3 = createTensor(DataType::FP6_E2M3, dims, strides);
    auto e3m2 = createTensor(DataType::FP6_E3M2, dims, strides);

    ASSERT_NE(e2m3, nullptr);
    ASSERT_NE(e3m2, nullptr);
    EXPECT_EQ(e2m3->dims(), dims);
    EXPECT_EQ(e3m2->dims(), dims);
    EXPECT_EQ(e2m3->strides(), strides);
    EXPECT_EQ(e3m2->strides(), strides);
    EXPECT_NE(dynamic_cast<Tensor<hipdnn_data_sdk::types::fp6_e2m3>*>(e2m3.get()), nullptr);
    EXPECT_NE(dynamic_cast<Tensor<hipdnn_data_sdk::types::fp6_e3m2>*>(e3m2.get()), nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorFp6PackedWhenRequested)
{
    const std::vector<int64_t> dims = {2, 4};
    const std::vector<int64_t> strides = {4, 1};

    auto e2m3 = createTensor(DataType::FP6_E2M3, dims, strides, /*packSubByteElements=*/true);
    auto e3m2 = createTensor(DataType::FP6_E3M2, dims, strides, /*packSubByteElements=*/true);

    ASSERT_NE(e2m3, nullptr);
    ASSERT_NE(e3m2, nullptr);
    EXPECT_EQ(e2m3->dims(), dims);
    EXPECT_EQ(e3m2->dims(), dims);
    EXPECT_NE(dynamic_cast<PackedFp6Tensor<hipdnn_data_sdk::types::fp6_e2m3>*>(e2m3.get()),
              nullptr);
    EXPECT_NE(dynamic_cast<PackedFp6Tensor<hipdnn_data_sdk::types::fp6_e3m2>*>(e3m2.get()),
              nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, PackSubByteElementsThrowsForUnimplementedSubByteTypes)
{
    const std::vector<int64_t> dims = {2, 2};
    const std::vector<int64_t> strides = {2, 1};

    EXPECT_THROW(createTensor(DataType::INT4, dims, strides, /*packSubByteElements=*/true),
                 std::runtime_error);

    EXPECT_NO_THROW(createTensor(DataType::INT4, dims, strides));
}

TEST(TestFlatbufferTensorAttributesUtils, CreateTensorFromAttributeForwardsPackFlag)
{
    flatbuffers::FlatBufferBuilder builder;
    const std::vector<int64_t> dims = {2, 4};
    const std::vector<int64_t> strides = {4, 1};
    auto attributeOffset
        = CreateTensorAttributesDirect(builder, 1, "x", DataType::FP4_E2M1, &strides, &dims);
    builder.Finish(attributeOffset);
    auto tensorAttr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    auto unpacked = createTensorFromAttribute(*tensorAttr);
    auto packed = createTensorFromAttribute(*tensorAttr, /*packSubByteElements=*/true);

    ASSERT_NE(unpacked, nullptr);
    ASSERT_NE(packed, nullptr);
    EXPECT_EQ(unpacked->dims(), dims);
    EXPECT_EQ(unpacked->strides(), strides);
    EXPECT_EQ(packed->dims(), dims);
    EXPECT_EQ(packed->strides(), strides);
    EXPECT_NE(dynamic_cast<Tensor<hipdnn_data_sdk::types::fp4_e2m1>*>(unpacked.get()), nullptr);
    EXPECT_NE(dynamic_cast<PackedFp4Tensor*>(packed.get()), nullptr);
}

TEST(TestFlatbufferTensorAttributesUtils, IsSubByteDataType)
{
    for(const auto dataType : EnumValuesDataType())
    {
        const bool expected = dataType == DataType::FP4_E2M1 || dataType == DataType::FP6_E2M3
                              || dataType == DataType::FP6_E3M2;
        EXPECT_EQ(isSubByteDataType(dataType), expected) << EnumNameDataType(dataType);
    }
}
