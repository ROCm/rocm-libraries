// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "MoeGroupedMatmulBwdGraphUtils.hpp"
#include "MoeGroupedMatmulBwdTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulBwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestMoeGroupedMatmulBwdSignatureKey, EqualityOperator)
{
    const MoeGroupedMatmulBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const MoeGroupedMatmulBwdSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key4{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const MoeGroupedMatmulBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key6{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const MoeGroupedMatmulBwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestMoeGroupedMatmulBwdSignatureKey, HashFunction)
{
    const MoeGroupedMatmulBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const MoeGroupedMatmulBwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const MoeGroupedMatmulBwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestMoeGroupedMatmulBwdSignatureKey, Copy)
{
    const MoeGroupedMatmulBwdSignatureKey original{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.doutputDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.tokenDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.dweightDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestMoeGroupedMatmulBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const MoeGroupedMatmulBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    MoeGroupedMatmulBwdTensorBundle<float> tensorBundle(2, 3, 4, 8);

    auto graphTuple = buildMoeGroupedMatmulBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const MoeGroupedMatmulBwdSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
