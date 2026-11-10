// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/GatherSignatureKey.hpp>

#include <stdexcept>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestGatherSignatureKey, EqualityOperator)
{
    const GatherSignatureKey key1{DataType::FLOAT, DataType::INT32};
    const GatherSignatureKey key2{DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key1 == key2);

    const GatherSignatureKey key3{DataType::FLOAT, DataType::INT64};
    EXPECT_FALSE(key1 == key3);

    const GatherSignatureKey key4{DataType::HALF, DataType::INT32};
    EXPECT_FALSE(key1 == key4);
}

TEST(TestGatherSignatureKey, HashFunction)
{
    const GatherSignatureKey key1{DataType::FLOAT, DataType::INT32};
    const GatherSignatureKey key2{DataType::FLOAT, DataType::INT32};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GatherSignatureKey key3{DataType::FLOAT, DataType::INT64};
    const GatherSignatureKey key4{DataType::BFLOAT16, DataType::INT32};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
    EXPECT_NE(key1.hashSelf(), key4.hashSelf());
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestGatherSignatureKey, Copy)
{
    const GatherSignatureKey original{DataType::BFLOAT16, DataType::INT64};
    const GatherSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.indexDataType, DataType::INT64);
}

TEST(TestGatherSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = createValidGatherGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const GatherSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == (GatherSignatureKey{DataType::FLOAT, DataType::INT64}));
}

TEST(TestGatherSignatureKey, CreateFromWrongNodeTypeThrows)
{
    auto builder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(GatherSignatureKey(graphWrap.getNode(0), graphWrap.getTensorMap()),
                 std::runtime_error);
}

TEST(TestGatherSignatureKey, PlanBuildersCoverFloatingTypesAndBothIndexTypes)
{
    const auto builders = GatherSignatureKey::getPlanBuilders();
    EXPECT_EQ(builders.size(), 6U);
    for(const auto data : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto index : {DataType::INT32, DataType::INT64})
        {
            EXPECT_EQ(builders.count(GatherSignatureKey{data, index}), 1U);
        }
    }
}
