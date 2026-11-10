// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/CausalConv1dSignatureKey.hpp>

using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestCausalConv1dSignatureKey, EqualityOperator)
{
    const CausalConv1dSignatureKey key1{
        DataType::FLOAT, DataType::INT32, DataType::UNSET, DataType::FLOAT};
    const CausalConv1dSignatureKey key2{
        DataType::FLOAT, DataType::INT32, DataType::UNSET, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const CausalConv1dSignatureKey dataDiffers{
        DataType::HALF, DataType::INT32, DataType::UNSET, DataType::FLOAT};
    EXPECT_FALSE(key1 == dataDiffers);

    const CausalConv1dSignatureKey cuSeqlensDiffers{
        DataType::FLOAT, DataType::INT64, DataType::UNSET, DataType::FLOAT};
    EXPECT_FALSE(key1 == cuSeqlensDiffers);

    const CausalConv1dSignatureKey stateIndicesDiffers{
        DataType::FLOAT, DataType::INT32, DataType::INT32, DataType::FLOAT};
    EXPECT_FALSE(key1 == stateIndicesDiffers);

    const CausalConv1dSignatureKey computeDiffers{
        DataType::FLOAT, DataType::INT32, DataType::UNSET, DataType::HALF};
    EXPECT_FALSE(key1 == computeDiffers);
}

TEST(TestCausalConv1dSignatureKey, HashFunction)
{
    const CausalConv1dSignatureKey key1{
        DataType::FLOAT, DataType::INT32, DataType::INT64, DataType::FLOAT};
    const CausalConv1dSignatureKey key2{
        DataType::FLOAT, DataType::INT32, DataType::INT64, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const CausalConv1dSignatureKey key3{
        DataType::FLOAT, DataType::INT32, DataType::INT32, DataType::FLOAT};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestCausalConv1dSignatureKey, ConstructFromNodeWithStateIndices)
{
    auto builder = hipdnn_test_sdk::utilities::createValidCausalConv1dGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const CausalConv1dSignatureKey key(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_EQ(key.dataType, DataType::FLOAT);
    EXPECT_EQ(key.cuSeqlensDataType, DataType::INT32);
    EXPECT_EQ(key.stateIndicesDataType, DataType::INT32);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
}

TEST(TestCausalConv1dSignatureKey, ConstructFromNodeWithoutStateIndices)
{
    auto builder = hipdnn_test_sdk::utilities::createValidCausalConv1dGraph(false);
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const CausalConv1dSignatureKey key(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_EQ(key.stateIndicesDataType, DataType::UNSET);
}

TEST(TestCausalConv1dSignatureKey, GetPlanBuildersCoversSupportedTypes)
{
    const auto builders = CausalConv1dSignatureKey::getPlanBuilders();

    // 3 data types * 2 cu_seqlens types * (no state_indices, INT32, INT64).
    EXPECT_EQ(builders.size(), 18u);
    for(const auto dataType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto cuSeqlensType : {DataType::INT32, DataType::INT64})
        {
            for(const auto stateIndicesType : {DataType::UNSET, DataType::INT32, DataType::INT64})
            {
                EXPECT_EQ(builders.count(CausalConv1dSignatureKey{
                              dataType, cuSeqlensType, stateIndicesType, DataType::FLOAT}),
                          1u);
            }
        }
    }
}
