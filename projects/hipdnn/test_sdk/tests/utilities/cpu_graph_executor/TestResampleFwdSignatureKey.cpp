// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleFwdSignatureKey.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestResampleFwdSignatureKey, EqualityAndHash)
{
    const ResampleFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const ResampleFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const ResampleFwdSignatureKey key3{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};

    EXPECT_TRUE(key1 == key2);
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());
    EXPECT_FALSE(key1 == key3);
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestResampleFwdSignatureKey, CreateFromNodeWithoutIndex)
{
    auto builder = createValidResampleFwdGraph();
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleFwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    EXPECT_EQ(key.xDataType, DataType::FLOAT);
    EXPECT_EQ(key.yDataType, DataType::FLOAT);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
    EXPECT_EQ(key.indexDataType, DataType::UNSET);
}

TEST(TestResampleFwdSignatureKey, CreateFromNodeWithIndex)
{
    auto builder = createValidResampleFwdGraph(true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleFwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    EXPECT_EQ(key.indexDataType, DataType::INT32);
}

TEST(TestResampleFwdSignatureKey, PlanBuildersContainIndexAndNoIndexVariants)
{
    auto builders = ResampleFwdSignatureKey::getPlanBuilders();

    EXPECT_NE(builders.find(ResampleFwdSignatureKey(
                  DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET)),
              builders.end());
    EXPECT_NE(builders.find(ResampleFwdSignatureKey(
                  DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32)),
              builders.end());
}
