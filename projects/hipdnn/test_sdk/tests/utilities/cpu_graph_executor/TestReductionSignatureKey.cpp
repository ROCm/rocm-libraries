// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>

#include "ReductionGraphUtils.hpp"
#include "ReductionTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestReductionSignatureKey, EqualityOperator)
{
    const ReductionSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const ReductionSignatureKey key3{DataType::HALF, DataType::HALF, DataType::FLOAT};
    const ReductionSignatureKey key4{DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    // Different xDataType
    const ReductionSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    // Different yDataType
    const ReductionSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    // Different computeDataType
    const ReductionSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestReductionSignatureKey, HashFunction)
{
    const ReductionSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // Keys differing in each field should produce different hashes
    const ReductionSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const ReductionSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestReductionSignatureKey, Copy)
{
    const ReductionSignatureKey original{DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT};
    const ReductionSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.yDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestReductionSignatureKey, CreateFromNodeAndTensorMap)
{
    const ReductionSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const std::vector<int64_t> xDims = {2, 3, 4, 4};
    const std::vector<int64_t> yDims = {2, 3, 1, 1};

    ReductionTensorBundle<float> tensorBundle(xDims, yDims, getGlobalTestSeed());

    auto graphTuple = buildReductionGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const ReductionSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
