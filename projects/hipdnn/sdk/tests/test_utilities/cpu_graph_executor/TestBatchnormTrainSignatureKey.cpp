// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/test_utilities/cpu_graph_executor/BatchnormTrainSignatureKey.hpp>

using namespace hipdnn_sdk::test_utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk_test_utils;

TEST(TestBatchnormTrainSignatureKey, EqualityOperator)
{
    BatchnormTrainSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BatchnormTrainSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    BatchnormTrainSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    BatchnormTrainSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BatchnormTrainSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key10{DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestBatchnormTrainSignatureKey, HashFunction)
{
    BatchnormTrainSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormTrainSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormTrainSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormTrainSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestBatchnormTrainSignatureKey, Copy)
{
    BatchnormTrainSignatureKey original{DataType::FLOAT, DataType::HALF, DataType::DOUBLE};
    BatchnormTrainSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
}

TEST(TestBatchnormTrainSignatureKey, CreateFromNodeAndTensorMap)
{
    BatchnormTrainSignatureKey expectedKey{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    BatchnormTrainTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormTrainGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, false);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormTrainSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
