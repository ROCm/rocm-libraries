// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/ConvolutionFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestConvolutionFwdSignatureKey, EqualityOperator)
{
    ConvolutionFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    ConvolutionFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    ConvolutionFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    ConvolutionFwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    ConvolutionFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    ConvolutionFwdSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    ConvolutionFwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    ConvolutionFwdSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestConvolutionFwdSignatureKey, HashFunction)
{
    ConvolutionFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    ConvolutionFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    ConvolutionFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    ConvolutionFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    ConvolutionFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestConvolutionFwdSignatureKey, Copy)
{
    ConvolutionFwdSignatureKey original{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    ConvolutionFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.wDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestConvolutionFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    ConvolutionFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> yDims = {1, 1, 2, 2};

    ConvolutionFwdTensorBundle<float> tensorBundle(xDims, wDims, yDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionFwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionFwdSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
