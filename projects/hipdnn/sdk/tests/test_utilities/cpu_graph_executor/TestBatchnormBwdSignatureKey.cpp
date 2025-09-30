// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormBwdSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk_test_utils;

TEST(TestBatchnormBwdSignatureKey, EqualityOperator)
{
    BatchnormBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BatchnormBwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    BatchnormBwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    BatchnormBwdSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BatchnormBwdSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestBatchnormBwdSignatureKey, HashFunction)
{
    BatchnormBwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormBwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormBwdSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormBwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormBwdSignatureKey, Copy)
{
    BatchnormBwdSignatureKey original{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
    BatchnormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
}

TEST(TestBatchnormBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    BatchnormBwdSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple
        = buildBatchnormBwdGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormBwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
