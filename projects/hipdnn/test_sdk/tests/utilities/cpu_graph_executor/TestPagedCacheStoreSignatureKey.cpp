// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PagedCacheStoreSignatureKey.hpp>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestPagedCacheStoreSignatureKey, EqualityOperator)
{
    const PagedCacheStoreSignatureKey key1{
        DataType::FLOAT, DataType::INT32, DataType::FLOAT, DataType::FLOAT};
    const PagedCacheStoreSignatureKey key2{
        DataType::FLOAT, DataType::INT32, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const PagedCacheStoreSignatureKey slotDiffers{
        DataType::FLOAT, DataType::INT64, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == slotDiffers);

    const PagedCacheStoreSignatureKey containerDiffers{
        DataType::FLOAT, DataType::INT32, DataType::FP8_E4M3, DataType::FLOAT};
    EXPECT_FALSE(key1 == containerDiffers);

    const PagedCacheStoreSignatureKey xDiffers{
        DataType::HALF, DataType::INT32, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == xDiffers);
}

TEST(TestPagedCacheStoreSignatureKey, HashFunction)
{
    const PagedCacheStoreSignatureKey key1{
        DataType::FLOAT, DataType::INT32, DataType::FLOAT, DataType::FLOAT};
    const PagedCacheStoreSignatureKey key2{
        DataType::FLOAT, DataType::INT32, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const PagedCacheStoreSignatureKey key3{
        DataType::HALF, DataType::INT32, DataType::HALF, DataType::FLOAT};
    const PagedCacheStoreSignatureKey key4{
        DataType::HALF, DataType::INT64, DataType::HALF, DataType::FLOAT};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestPagedCacheStoreSignatureKey, ConstructFromNode)
{
    auto builder = hipdnn_test_sdk::utilities::createValidPagedCacheStoreGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    // Lowered nodes carry no compute type: an UNSET one still keys on FLOAT.
    const PagedCacheStoreSignatureKey key(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::UNSET);

    EXPECT_EQ(key.xDataType, DataType::FLOAT);
    EXPECT_EQ(key.slotMappingDataType, DataType::INT32);
    EXPECT_EQ(key.containerDataType, DataType::FLOAT);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
}

TEST(TestPagedCacheStoreSignatureKey, GetPlanBuildersCoversSupportedTypes)
{
    const auto builders = PagedCacheStoreSignatureKey::getPlanBuilders();

    // 3 x types * 2 slot types * (same-type container + FP8_E4M3 + FP8_E5M2).
    EXPECT_EQ(builders.size(), 18u);
    for(const auto xType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto slotType : {DataType::INT32, DataType::INT64})
        {
            for(const auto containerType : {xType, DataType::FP8_E4M3, DataType::FP8_E5M2})
            {
                EXPECT_EQ(builders.count(PagedCacheStoreSignatureKey{
                              xType, slotType, containerType, DataType::FLOAT}),
                          1u);
            }
        }
    }
}
