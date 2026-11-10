// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeRoutingSignatureKey.hpp>

#include <stdexcept>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestMoeRoutingSignatureKey, EqualityOperator)
{
    const MoeRoutingSignatureKey key1{DataType::INT32};
    const MoeRoutingSignatureKey key2{DataType::INT32};
    EXPECT_TRUE(key1 == key2);

    const MoeRoutingSignatureKey key3{DataType::INT64};
    EXPECT_FALSE(key1 == key3);
}

TEST(TestMoeRoutingSignatureKey, HashFunction)
{
    const MoeRoutingSignatureKey key1{DataType::INT32};
    const MoeRoutingSignatureKey key2{DataType::INT32};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const MoeRoutingSignatureKey key3{DataType::INT64};
    EXPECT_NE(key1.hashSelf(), key3.hashSelf());
}

TEST(TestMoeRoutingSignatureKey, Copy)
{
    const MoeRoutingSignatureKey original{DataType::INT64};
    const MoeRoutingSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.offsetDataType, DataType::INT64);
}

TEST(TestMoeRoutingSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = createValidMoeRoutingGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const MoeRoutingSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == MoeRoutingSignatureKey{DataType::INT32});
}

TEST(TestMoeRoutingSignatureKey, CreateFromWrongNodeTypeThrows)
{
    auto builder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(MoeRoutingSignatureKey(graphWrap.getNode(0), graphWrap.getTensorMap()),
                 std::runtime_error);
}

TEST(TestMoeRoutingSignatureKey, PlanBuildersCoverBothOffsetTypes)
{
    const auto builders = MoeRoutingSignatureKey::getPlanBuilders();
    EXPECT_EQ(builders.size(), 2U);
    EXPECT_EQ(builders.count(MoeRoutingSignatureKey{DataType::INT32}), 1U);
    EXPECT_EQ(builders.count(MoeRoutingSignatureKey{DataType::INT64}), 1U);
}
