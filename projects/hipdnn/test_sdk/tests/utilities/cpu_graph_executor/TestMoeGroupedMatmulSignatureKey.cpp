// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <sstream>
#include <unordered_map>

#include "MoeGroupedMatmulGraphUtils.hpp"
#include "MoeGroupedMatmulTensorBundles.hpp"
#include "PointwiseGraphUtils.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestMoeGroupedMatmulSignatureKey, EqualityOperator)
{
    const MoeGroupedMatmulSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const MoeGroupedMatmulSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key4{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const MoeGroupedMatmulSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key6{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const MoeGroupedMatmulSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const MoeGroupedMatmulSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    const MoeGroupedMatmulSignatureKey key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestMoeGroupedMatmulSignatureKey, HashFunctionSupportsUnorderedLookup)
{
    const MoeGroupedMatmulSignatureKey key{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey equalKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const MoeGroupedMatmulSignatureKey differentKey{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    EXPECT_EQ(key.hashSelf(), equalKey.hashSelf());

    std::unordered_map<MoeGroupedMatmulSignatureKey, int, MoeGroupedMatmulSignatureKey> builders;
    builders.emplace(key, 42);
    EXPECT_EQ(builders.at(equalKey), 42);
    EXPECT_EQ(builders.find(differentKey), builders.end());
}

TEST(TestMoeGroupedMatmulSignatureKey, Copy)
{
    const MoeGroupedMatmulSignatureKey original{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const MoeGroupedMatmulSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.tokenDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.weightDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestMoeGroupedMatmulSignatureKey, CreateFromNodeAndTensorMap)
{
    const MoeGroupedMatmulSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    MoeGroupedMatmulTensorBundle<float> tensorBundle(
        2, 3, 4, 6, 6, MoeGroupedMatmulMode::NONE, 0, 1);
    auto graphTuple = buildMoeGroupedMatmulGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    const auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const MoeGroupedMatmulSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestMoeGroupedMatmulSignatureKey, CastFailureThrows)
{
    const std::vector<int64_t> dims = {1, 3, 4, 4};
    auto graphTuple = buildPointwiseUnaryGraph(dims,
                                               dims,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               hipdnn_frontend::PointwiseMode::RELU_FWD,
                                               1,
                                               TensorLayout::NCHW);
    auto [serializedGraph, serErr] = std::get<0>(graphTuple)->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();
    const auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    EXPECT_THROW(MoeGroupedMatmulSignatureKey(
                     graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT),
                 std::runtime_error);
}

TEST(TestMoeGroupedMatmulSignatureKey, GetPlanBuildersReturnsExactlySevenEntries)
{
    EXPECT_EQ(MoeGroupedMatmulSignatureKey::getPlanBuilders().size(), 7U);
}

TEST(TestMoeGroupedMatmulSignatureKey, StreamOutputContainsOpName)
{
    const MoeGroupedMatmulSignatureKey key{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::ostringstream oss;
    oss << key;
    EXPECT_NE(oss.str().find("MoeGroupedMatmul("), std::string::npos) << oss.str();
}
