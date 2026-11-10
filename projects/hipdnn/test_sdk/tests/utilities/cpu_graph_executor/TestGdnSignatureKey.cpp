// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GdnSignatureKey.hpp>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestGdnSignatureKey, EqualityOperator)
{
    const GdnSignatureKey key1{DataType::FLOAT, DataType::INT32, DataType::FLOAT};
    const GdnSignatureKey key2{DataType::FLOAT, DataType::INT32, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GdnSignatureKey ioDiffers{DataType::BFLOAT16, DataType::INT32, DataType::FLOAT};
    EXPECT_FALSE(key1 == ioDiffers);

    const GdnSignatureKey cuSeqlensDiffers{DataType::FLOAT, DataType::INT64, DataType::FLOAT};
    EXPECT_FALSE(key1 == cuSeqlensDiffers);

    const GdnSignatureKey computeDiffers{DataType::FLOAT, DataType::INT32, DataType::HALF};
    EXPECT_FALSE(key1 == computeDiffers);
}

TEST(TestGdnSignatureKey, HashFunction)
{
    const GdnSignatureKey key1{DataType::HALF, DataType::INT64, DataType::FLOAT};
    const GdnSignatureKey key2{DataType::HALF, DataType::INT64, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GdnSignatureKey key3{DataType::HALF, DataType::INT32, DataType::FLOAT};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestGdnSignatureKey, ConstructFromNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidGdnGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const GdnSignatureKey key(graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_EQ(key.ioDataType, DataType::FLOAT);
    EXPECT_EQ(key.cuSeqlensDataType, DataType::INT32);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
}

TEST(TestGdnSignatureKey, ConstructFromWrongNodeThrows)
{
    auto builder = hipdnn_test_sdk::utilities::createValidKdaGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(GdnSignatureKey(graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT),
                 std::runtime_error);
}

TEST(TestGdnSignatureKey, GetPlanBuildersCoversSupportedTypes)
{
    const auto builders = GdnSignatureKey::getPlanBuilders();

    // 3 io types * 2 cu_seqlens types, FLOAT compute.
    EXPECT_EQ(builders.size(), 6u);
    for(const auto ioType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto cuSeqlensType : {DataType::INT32, DataType::INT64})
        {
            EXPECT_EQ(builders.count(GdnSignatureKey{ioType, cuSeqlensType, DataType::FLOAT}), 1u);
        }
    }
}
