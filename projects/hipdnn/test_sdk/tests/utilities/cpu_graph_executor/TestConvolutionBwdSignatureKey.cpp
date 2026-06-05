// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestConvolutionBwdSignatureKey, EqualityOperator)
{
    const ConvolutionBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const ConvolutionBwdSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const ConvolutionBwdSignatureKey key4{
        DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    const ConvolutionBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key6{
        DataType::HALF, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    const ConvolutionBwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestConvolutionBwdSignatureKey, HashFunction)
{
    const ConvolutionBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const ConvolutionBwdSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const ConvolutionBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestConvolutionBwdSignatureKey, Copy)
{
    const ConvolutionBwdSignatureKey original{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const ConvolutionBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.wDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestConvolutionBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const ConvolutionBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const std::vector<int64_t> dxDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const ConvolutionBwdSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
