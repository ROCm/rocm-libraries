// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/TopkSignatureKey.hpp>

#include <stdexcept>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestTopkSignatureKey, EqualityOperator)
{
    const TopkSignatureKey key1{DataType::FLOAT, DataType::INT32};
    const TopkSignatureKey key2{DataType::FLOAT, DataType::INT32};
    EXPECT_TRUE(key1 == key2);

    const TopkSignatureKey key3{DataType::FLOAT, DataType::INT64};
    EXPECT_FALSE(key1 == key3);

    const TopkSignatureKey key4{DataType::HALF, DataType::INT32};
    EXPECT_FALSE(key1 == key4);
}

TEST(TestTopkSignatureKey, HashFunction)
{
    const TopkSignatureKey key1{DataType::FLOAT, DataType::INT32};
    const TopkSignatureKey key2{DataType::FLOAT, DataType::INT32};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const TopkSignatureKey key3{DataType::FLOAT, DataType::INT64};
    const TopkSignatureKey key4{DataType::BFLOAT16, DataType::INT32};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
    EXPECT_NE(key1.hashSelf(), key4.hashSelf());
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestTopkSignatureKey, Copy)
{
    const TopkSignatureKey original{DataType::BFLOAT16, DataType::INT64};
    const TopkSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.indexDataType, DataType::INT64);
}

TEST(TestTopkSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = createValidTopkGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const TopkSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == (TopkSignatureKey{DataType::FLOAT, DataType::INT32}));
}

TEST(TestTopkSignatureKey, PlanBuildersCoverFloatingTypesAndBothIndexTypes)
{
    const auto builders = TopkSignatureKey::getPlanBuilders();
    EXPECT_EQ(builders.size(), 6U);
    for(const auto data : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        for(const auto index : {DataType::INT32, DataType::INT64})
        {
            EXPECT_EQ(builders.count(TopkSignatureKey{data, index}), 1U);
        }
    }
}
