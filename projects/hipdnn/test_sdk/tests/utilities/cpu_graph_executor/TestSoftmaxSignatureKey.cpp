// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SoftmaxSignatureKey.hpp>

#include <stdexcept>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;

TEST(TestSoftmaxSignatureKey, EqualityOperator)
{
    const SoftmaxSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SoftmaxSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    // Different pDataType, sDataType, computeDataType.
    EXPECT_FALSE(key1 == (SoftmaxSignatureKey{DataType::HALF, DataType::FLOAT, DataType::FLOAT}));
    EXPECT_FALSE(key1 == (SoftmaxSignatureKey{DataType::FLOAT, DataType::HALF, DataType::FLOAT}));
    EXPECT_FALSE(key1 == (SoftmaxSignatureKey{DataType::FLOAT, DataType::FLOAT, DataType::HALF}));
}

TEST(TestSoftmaxSignatureKey, HashFunction)
{
    const SoftmaxSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SoftmaxSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const SoftmaxSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const SoftmaxSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const SoftmaxSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
    EXPECT_NE(key3.hashSelf(), key5.hashSelf());
    EXPECT_NE(key4.hashSelf(), key5.hashSelf());
}

TEST(TestSoftmaxSignatureKey, Copy)
{
    const SoftmaxSignatureKey original{DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT};
    const SoftmaxSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.pDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.sDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestSoftmaxSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = createValidSoftmaxGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    const SoftmaxSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode
                == (SoftmaxSignatureKey{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT}));
}

TEST(TestSoftmaxSignatureKey, CreateFromWrongNodeTypeThrows)
{
    auto builder = createValidMoeGroupedMatmulBwdGraph();
    const GraphWrapper graphWrap(builder.GetBufferPointer(), builder.GetSize());

    EXPECT_THROW(
        SoftmaxSignatureKey(graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT),
        std::runtime_error);
}

TEST(TestSoftmaxSignatureKey, PlanBuildersCoverFloatingTypes)
{
    const auto builders = SoftmaxSignatureKey::getPlanBuilders();
    EXPECT_EQ(builders.size(), 5U);
    for(const auto data : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        EXPECT_EQ(builders.count(SoftmaxSignatureKey{data, data, DataType::FLOAT}), 1U);
    }
    EXPECT_EQ(builders.count(SoftmaxSignatureKey{DataType::HALF, DataType::FLOAT, DataType::FLOAT}),
              1U);
    EXPECT_EQ(
        builders.count(SoftmaxSignatureKey{DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT}),
        1U);
}
