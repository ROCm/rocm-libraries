// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionWrwSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk_test_utils;

TEST(TestConvolutionWrwSignatureKey, EqualityOperator)
{
    ConvolutionWrwSignatureKey key1{DataType::FLOAT, DataType::FLOAT};
    ConvolutionWrwSignatureKey key2{DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    ConvolutionWrwSignatureKey key3{DataType::HALF, DataType::FLOAT};
    ConvolutionWrwSignatureKey key4{DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    ConvolutionWrwSignatureKey key5{DataType::FLOAT, DataType::FLOAT};
    ConvolutionWrwSignatureKey key6{DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    ConvolutionWrwSignatureKey key7{DataType::FLOAT, DataType::FLOAT};
    ConvolutionWrwSignatureKey key8{DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestConvolutionWrwSignatureKey, HashFunction)
{
    ConvolutionWrwSignatureKey key1{DataType::FLOAT, DataType::FLOAT};
    ConvolutionWrwSignatureKey key2{DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    ConvolutionWrwSignatureKey key3{DataType::HALF, DataType::FLOAT};
    ConvolutionWrwSignatureKey key4{DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();

    EXPECT_TRUE(hash3 != hash4);
}

TEST(TestConvolutionWrwSignatureKey, Copy)
{
    ConvolutionWrwSignatureKey original{DataType::FLOAT, DataType::HALF};
    ConvolutionWrwSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.accumulatorDataType, DataType::HALF);
}

TEST(TestConvolutionWrwSignatureKey, CreateFromNodeAndTensorMap)
{
    ConvolutionWrwSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> xDims = {1, 1, 2, 2};
    std::vector<int64_t> dwDims = {1, 1, 1, 1};
    std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionWrwTensorBundle<float> tensorBundle(xDims, dwDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionWrwGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionWrwSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
