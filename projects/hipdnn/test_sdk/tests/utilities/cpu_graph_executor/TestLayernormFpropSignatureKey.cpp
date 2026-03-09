// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "LayernormGraphUtils.hpp"
#include "LayernormTensorBundles.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestLayernormFpropSignatureKey, EqualityOperator)
{
    LayernormFpropSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    LayernormFpropSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    LayernormFpropSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    LayernormFpropSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    LayernormFpropSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    LayernormFpropSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    LayernormFpropSignatureKey key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestLayernormFpropSignatureKey, HashFunction)
{
    LayernormFpropSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    LayernormFpropSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    LayernormFpropSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    LayernormFpropSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    LayernormFpropSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();
    auto hash7 = key7.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6 && hash3 != hash7 && hash4 != hash7
                && hash5 != hash7 && hash6 != hash7);
}

TEST(TestLayernormFpropSignatureKey, Copy)
{
    LayernormFpropSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::BFLOAT16};
    LayernormFpropSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanInvVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestLayernormFpropSignatureKey, CreateFromNodeAndTensorMap)
{
    LayernormFpropSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildLayernormFpropGraph(DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          DataType::FLOAT,
                                          dims,
                                          TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(flatbufferGraph.data(),
                                                                         flatbufferGraph.size());

    LayernormFpropSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
