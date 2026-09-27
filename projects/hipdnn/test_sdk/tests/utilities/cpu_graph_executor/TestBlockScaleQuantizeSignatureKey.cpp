// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleQuantizeSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;

TEST(TestBlockScaleQuantizeSignatureKey, EqualityOperator)
{
    const BlockScaleQuantizeSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const BlockScaleQuantizeSignatureKey key3{
        DataType::HALF, DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key4{
        DataType::HALF, DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    // Different input type
    const BlockScaleQuantizeSignatureKey key5{
        DataType::FLOAT, DataType::FP8_E5M2, DataType::FLOAT, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key6{
        DataType::HALF, DataType::FP8_E5M2, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    // Different output type
    const BlockScaleQuantizeSignatureKey key7{
        DataType::BFLOAT16, DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key8{
        DataType::BFLOAT16, DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    // Different scale type
    const BlockScaleQuantizeSignatureKey key9{
        DataType::FLOAT, DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key10{
        DataType::FLOAT, DataType::FP8_E5M2, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    // Different compute type
    const BlockScaleQuantizeSignatureKey key11{
        DataType::FLOAT, DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key12{
        DataType::FLOAT, DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::DOUBLE};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestBlockScaleQuantizeSignatureKey, HashFunction)
{
    const BlockScaleQuantizeSignatureKey key1{
        DataType::FLOAT, DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key2{
        DataType::FLOAT, DataType::FP4_E2M1, DataType::FP8_E8M0, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const BlockScaleQuantizeSignatureKey key3{
        DataType::HALF, DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key4{
        DataType::BFLOAT16, DataType::FP4_E2M1, DataType::FLOAT, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key5{
        DataType::FLOAT, DataType::FP6_E2M3, DataType::FLOAT, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey key6{
        DataType::HALF, DataType::FP8_E5M2, DataType::FP8_E8M0, DataType::DOUBLE};

    const auto hash3 = key3.hashSelf();
    const auto hash4 = key4.hashSelf();
    const auto hash5 = key5.hashSelf();
    const auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestBlockScaleQuantizeSignatureKey, Copy)
{
    const BlockScaleQuantizeSignatureKey original{
        DataType::FLOAT, DataType::FP8_E4M3, DataType::FP8_E8M0, DataType::FLOAT};
    const BlockScaleQuantizeSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::FP8_E4M3);
    EXPECT_EQ(copied.scaleDataType, DataType::FP8_E8M0);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestBlockScaleQuantizeSignatureKey, CreateFromNodeAndTensorMap)
{
    const BlockScaleQuantizeSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    auto builder = createValidBlockScaleQuantizeGraph();
    const auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const BlockScaleQuantizeSignatureKey keyFromNode(graphWrap.getNode(0),
                                                     graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
