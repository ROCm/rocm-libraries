// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "SdpaGraphUtils.hpp"
#include "SdpaTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestSdpaFwdSignatureKey, EqualityOperator)
{
    const SdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const SdpaFwdSignatureKey key3{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    const SdpaFwdSignatureKey key4{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    const SdpaFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key6{DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    const SdpaFwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key8{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const SdpaFwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestSdpaFwdSignatureKey, HashFunction)
{
    const SdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const SdpaFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const SdpaFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const SdpaFwdSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestSdpaFwdSignatureKey, Copy)
{
    const SdpaFwdSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::FLOAT};
    const SdpaFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.qDataType, DataType::FLOAT);
    EXPECT_EQ(copied.kDataType, DataType::HALF);
    EXPECT_EQ(copied.vDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.oDataType, DataType::FLOAT);
}

TEST(TestSdpaFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const SdpaFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    const std::vector<int64_t> qDims = {1, 1, 1, 1};
    const std::vector<int64_t> kDims = {1, 1, 1, 1};
    const std::vector<int64_t> vDims = {1, 1, 1, 1};

    SdpaFwdTensorBundle<float> tensorBundle(qDims, kDims, vDims);
    auto graphTuple = buildSdpaFwdGraph(tensorBundle, DataType::FLOAT);
    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const SdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
