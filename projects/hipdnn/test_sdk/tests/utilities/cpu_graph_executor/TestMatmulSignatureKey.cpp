// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "MatmulGraphUtils.hpp"
#include "MatmulTensorBundles.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/MatmulSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestMatmulSignatureKey, EqualityOperator)
{
    MatmulSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    MatmulSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    MatmulSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    MatmulSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    MatmulSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    MatmulSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    MatmulSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    MatmulSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestMatmulSignatureKey, HashFunction)
{
    MatmulSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    MatmulSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    MatmulSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    MatmulSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    MatmulSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestMatmulSignatureKey, Copy)
{
    MatmulSignatureKey original{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    MatmulSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.aDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.bDataType, DataType::FLOAT);
    EXPECT_EQ(copied.cDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestMatmulSignatureKey, CreateFromNodeAndTensorMap)
{
    MatmulSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> aDims = {1, 1, 4, 2};
    std::vector<int64_t> bDims = {1, 1, 2, 3};
    std::vector<int64_t> cDims = {1, 1, 4, 3};

    MatmulTensorBundle<float> tensorBundle(aDims, bDims, cDims, false, false, 1);

    auto graphTuple = buildMatmulGraph(tensorBundle, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();

    auto graphWrap
        = hipdnn_plugin_sdk::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    MatmulSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    EXPECT_TRUE(keyFromNode == expectedKey);
}
