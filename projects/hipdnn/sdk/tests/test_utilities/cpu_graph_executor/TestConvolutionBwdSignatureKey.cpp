// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "ConvolutionGraphUtils.hpp"
#include "ConvolutionTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/ConvolutionBwdSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk_test_utils;

TEST(TestConvolutionBwdInferenceSignatureKey, EqualityOperator)
{
    ConvolutionBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT};
    ConvolutionBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    ConvolutionBwdSignatureKey key3{DataType::HALF, DataType::FLOAT};
    ConvolutionBwdSignatureKey key4{DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    ConvolutionBwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT};
    ConvolutionBwdSignatureKey key6{DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    ConvolutionBwdSignatureKey key7{DataType::FLOAT, DataType::FLOAT};
    ConvolutionBwdSignatureKey key8{DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestConvolutionBwdInferenceSignatureKey, HashFunction)
{
    ConvolutionBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT};
    ConvolutionBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    ConvolutionBwdSignatureKey key3{DataType::HALF, DataType::FLOAT};
    ConvolutionBwdSignatureKey key4{DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();

    EXPECT_TRUE(hash3 != hash4);
}

TEST(TestConvolutionBwdInferenceSignatureKey, Copy)
{
    ConvolutionBwdSignatureKey original{DataType::FLOAT, DataType::HALF};
    ConvolutionBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.accumulatorDataType, DataType::HALF);
}

TEST(TestConvolutionBwdInferenceSignatureKey, CreateFromNodeAndTensorMap)
{
    ConvolutionBwdSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dxDims = {1, 1, 2, 2};
    std::vector<int64_t> wDims = {1, 1, 1, 1};
    std::vector<int64_t> dyDims = {1, 1, 2, 2};

    ConvolutionBwdTensorBundle<float> tensorBundle(dxDims, wDims, dyDims, 1, TensorLayout::NCHW);

    auto graphTuple = buildConvolutionBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    ConvolutionBwdSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
