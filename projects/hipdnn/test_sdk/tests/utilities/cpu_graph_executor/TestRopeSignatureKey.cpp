// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RopeSignatureKey.hpp>

#include <stdexcept>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestRopeSignatureKey, EqualityOperator)
{
    const RopeSignatureKey key1{DataType::FLOAT};
    const RopeSignatureKey key2{DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const RopeSignatureKey key3{DataType::HALF};
    EXPECT_FALSE(key1 == key3);
}

TEST(TestRopeSignatureKey, HashFunction)
{
    const RopeSignatureKey key1{DataType::FLOAT};
    const RopeSignatureKey key2{DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const RopeSignatureKey key3{DataType::HALF};
    const RopeSignatureKey key4{DataType::BFLOAT16};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
    EXPECT_NE(key1.hashSelf(), key4.hashSelf());
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestRopeSignatureKey, Copy)
{
    const RopeSignatureKey original{DataType::BFLOAT16};
    const RopeSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dataType, DataType::BFLOAT16);
}

TEST(TestRopeSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = createValidRopeGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const RopeSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == RopeSignatureKey{DataType::FLOAT});
}

TEST(TestRopeSignatureKey, PlanBuildersCoverFloatingTypes)
{
    const auto builders = RopeSignatureKey::getPlanBuilders();
    EXPECT_EQ(builders.size(), 3U);
    for(const auto data : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        EXPECT_EQ(builders.count(RopeSignatureKey{data}), 1U);
    }
}
