// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestConvolutionWrwSignatureKey, EqualityOperator)
{
    const ConvolutionWrwSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const ConvolutionWrwSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const ConvolutionWrwSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const ConvolutionWrwSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestConvolutionWrwSignatureKey, HashFunction)
{
    const ConvolutionWrwSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const ConvolutionWrwSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const ConvolutionWrwSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestConvolutionWrwSignatureKey, Copy)
{
    const ConvolutionWrwSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionWrwSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.dyDataType, DataType::HALF);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestConvolutionWrwSignatureKey, CreateFromNodeAndTensorMap)
{
    const ConvolutionWrwSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> dwDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionWrwTensorBundle<float> tensorBundle(xDims, dwDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionWrwGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const ConvolutionWrwSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
