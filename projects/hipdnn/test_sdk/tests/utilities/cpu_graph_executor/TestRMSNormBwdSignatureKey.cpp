// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "RMSNormGraphUtils.hpp"
#include "RMSNormTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormBwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestRMSNormBwdSignatureKey, EqualityOperator)
{
    const RMSNormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const RMSNormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const RMSNormBwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const RMSNormBwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    const RMSNormBwdSignatureKey key5{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key5);

    const RMSNormBwdSignatureKey key6{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key6);

    const RMSNormBwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key7);

    // Mismatch in dx
    const RMSNormBwdSignatureKey key8{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key1 == key8);

    const RMSNormBwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key1 == key9);
}

TEST(TestRMSNormBwdSignatureKey, HashFunction)
{
    const RMSNormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const RMSNormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    std::vector<std::size_t> hashes;
    hashes.push_back(
        RMSNormBwdSignatureKey(
            DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        RMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        RMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        RMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        RMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF)
            .hashSelf());

    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]);
        }
    }
}

TEST(TestRMSNormBwdSignatureKey, Copy)
{
    const RMSNormBwdSignatureKey original{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const RMSNormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::HALF);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleDataType, DataType::HALF);
    EXPECT_EQ(copied.dxDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestRMSNormBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const RMSNormBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};

    const std::vector<int64_t> dims = {1, 2, 3, 4};
    auto graph = buildRMSNormBwdGraph(
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, dims, TensorLayout::NHWC);

    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const RMSNormBwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
