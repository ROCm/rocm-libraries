// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <memory>
#include <variant>
#include <vector>

#include <hipdnn_sdk/test_utilities/cpu_graph_executor/TensorVariant.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::data_objects;

TEST(TestTensorVariantUtils, CreateHostOnlyShallowTensorVariantFloat)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims{1, 3, 4, 4};
    std::vector<int64_t> strides{48, 16, 4, 1};
    auto attrOffset
        = CreateTensorAttributesDirect(builder, 7, "x", DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    std::array<float, 48> backing = {0};
    auto variant = TensorVariantUtils::createHostOnlyShallowTensorVariant(*attr, backing.data());
    ASSERT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<float>>>(variant));
    auto& t = *std::get<std::unique_ptr<TensorBase<float>>>(variant);
    EXPECT_EQ(t.dims(), dims);
    EXPECT_EQ(t.strides(), strides);
    EXPECT_EQ(t.memory().hostData(), backing.data());
}

TEST(TestTensorVariantUtils, CreateHostOnlyShallowTensorVariantHalf)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims{1, 1, 1, 2};
    std::vector<int64_t> strides{2, 2, 2, 1};
    auto attrOffset
        = CreateTensorAttributesDirect(builder, 7, "x", DataType::HALF, &strides, &dims);
    builder.Finish(attrOffset);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    std::array<half, 2> backing = {};
    auto variant = TensorVariantUtils::createHostOnlyShallowTensorVariant(*attr, backing.data());
    EXPECT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<half>>>(variant));
    auto& ptr = *std::get<std::unique_ptr<TensorBase<half>>>(variant);
    EXPECT_EQ(ptr.dims(), dims);
    EXPECT_EQ(ptr.strides(), strides);
    EXPECT_EQ(ptr.memory().hostData(), backing.data());
}

TEST(TestTensorVariantUtils, UnsupportedDataTypeThrows)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims{1};
    std::vector<int64_t> strides{1};
    auto attrOffset
        = CreateTensorAttributesDirect(builder, 7, "x", DataType::INT32, &strides, &dims);
    builder.Finish(attrOffset);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    std::array<int, 1> dummy = {0};
    EXPECT_THROW(TensorVariantUtils::createHostOnlyShallowTensorVariant(*attr, dummy.data()),
                 std::runtime_error);
}

TEST(TestTensorVariantUtils, CreateWithEmptyDimsAndStrides)
{
    flatbuffers::FlatBufferBuilder builder;
    std::vector<int64_t> dims{};
    std::vector<int64_t> strides{};
    auto attrOffset
        = CreateTensorAttributesDirect(builder, 7, "x", DataType::FLOAT, &strides, &dims);
    builder.Finish(attrOffset);
    auto* attr = flatbuffers::GetRoot<TensorAttributes>(builder.GetBufferPointer());

    float backing = 0;
    auto variant = TensorVariantUtils::createHostOnlyShallowTensorVariant(*attr, &backing);
    ASSERT_TRUE(std::holds_alternative<std::unique_ptr<TensorBase<float>>>(variant));
    auto& t = *std::get<std::unique_ptr<TensorBase<float>>>(variant);
    EXPECT_TRUE(t.dims().empty());
    EXPECT_TRUE(t.strides().empty());
    EXPECT_EQ(t.memory().hostData(), &backing);
}
