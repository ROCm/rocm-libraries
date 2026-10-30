// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleBwdSignatureKey.hpp>

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_flatbuffers_sdk::flatbuffer_utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_test_sdk::utilities;

TEST(TestResampleBwdSignatureKey, EqualityOperator)
{
    const ResampleBwdSignatureKey baseKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const ResampleBwdSignatureKey matchingKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_TRUE(baseKey == matchingKey);

    // Mismatch in dyDataType
    const ResampleBwdSignatureKey diffDyKey{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(baseKey == diffDyKey);

    // Mismatch in dxDataType
    const ResampleBwdSignatureKey diffDxKey{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::UNSET};
    EXPECT_FALSE(baseKey == diffDxKey);

    // Mismatch in computeDataType
    const ResampleBwdSignatureKey diffComputeKey{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::UNSET};
    EXPECT_FALSE(baseKey == diffComputeKey);

    // Mismatch in indexDataType
    const ResampleBwdSignatureKey diffIndexKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32};
    EXPECT_FALSE(baseKey == diffIndexKey);
}

TEST(TestResampleBwdSignatureKey, HashFunctionUniqueness)
{
    const ResampleBwdSignatureKey baseKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};
    const ResampleBwdSignatureKey duplicateKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET};

    EXPECT_EQ(baseKey.hashSelf(), duplicateKey.hashSelf());

    std::vector<std::size_t> hashes;
    hashes.push_back(
        ResampleBwdSignatureKey(DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::UNSET)
            .hashSelf());
    hashes.push_back(
        ResampleBwdSignatureKey(DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::UNSET)
            .hashSelf());
    hashes.push_back(
        ResampleBwdSignatureKey(DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::UNSET)
            .hashSelf());
    hashes.push_back(
        ResampleBwdSignatureKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32)
            .hashSelf());

    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]);
        }
    }
}

TEST(TestResampleBwdSignatureKey, CopyConstructor)
{
    const ResampleBwdSignatureKey original{
        DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::INT32};
    const ResampleBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::HALF);
    EXPECT_EQ(copied.dxDataType, DataType::HALF);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
    EXPECT_EQ(copied.indexDataType, DataType::INT32);
}

TEST(TestResampleBwdSignatureKey, CreateFromNodeWithoutIndex)
{
    auto builder = createValidResampleBwdGraph(false);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    EXPECT_EQ(key.dyDataType, DataType::FLOAT);
    EXPECT_EQ(key.dxDataType, DataType::FLOAT);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
    EXPECT_EQ(key.indexDataType, DataType::UNSET);
}

TEST(TestResampleBwdSignatureKey, CreateFromNodeWithIndex)
{
    auto builder = createValidResampleBwdGraph(true);
    const GraphWrapper graph(builder.GetBufferPointer(), builder.GetSize());

    const ResampleBwdSignatureKey key(graph.getNode(0), graph.getTensorMap());

    EXPECT_EQ(key.dyDataType, DataType::FLOAT);
    EXPECT_EQ(key.dxDataType, DataType::FLOAT);
    EXPECT_EQ(key.computeDataType, DataType::FLOAT);
    EXPECT_EQ(key.indexDataType, DataType::INT32);
}

TEST(TestResampleBwdSignatureKey, PlanBuildersContainIndexAndNoIndexVariants)
{
    auto builders = ResampleBwdSignatureKey::getPlanBuilders();

    EXPECT_NE(builders.find(ResampleBwdSignatureKey(
                  DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::UNSET)),
              builders.end());
    EXPECT_NE(builders.find(ResampleBwdSignatureKey(
                  DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::INT32)),
              builders.end());
}
