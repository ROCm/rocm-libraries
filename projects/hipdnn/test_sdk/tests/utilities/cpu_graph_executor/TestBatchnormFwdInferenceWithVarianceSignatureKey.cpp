// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_sdk::data_objects;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestBatchnormFwdInferenceWithVarianceSignatureKey, EqualityOperator)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BatchnormFwdInferenceWithVarianceSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    BatchnormFwdInferenceWithVarianceSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    BatchnormFwdInferenceWithVarianceSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    BatchnormFwdInferenceWithVarianceSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BatchnormFwdInferenceWithVarianceSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    BatchnormFwdInferenceWithVarianceSignatureKey key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestBatchnormFwdInferenceWithVarianceSignatureKey, HashFunction)
{
    BatchnormFwdInferenceWithVarianceSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormFwdInferenceWithVarianceSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    BatchnormFwdInferenceWithVarianceSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormFwdInferenceWithVarianceSignatureKey key7{
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

TEST(TestBatchnormFwdInferenceWithVarianceSignatureKey, Copy)
{
    BatchnormFwdInferenceWithVarianceSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::BFLOAT16};
    BatchnormFwdInferenceWithVarianceSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestBatchnormFwdInferenceWithVarianceSignatureKey, CreateFromNodeAndTensorMap)
{
    BatchnormFwdInferenceWithVarianceSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    double epsilon = 1e-5;

    auto graph = buildBatchnormFwdInferenceWithVarianceGraph(DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             DataType::FLOAT,
                                                             dims,
                                                             TensorLayout::NHWC,
                                                             epsilon);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_plugin::GraphWrapper(flatbufferGraph.data(), flatbufferGraph.size());

    BatchnormFwdInferenceWithVarianceSignatureKey keyFromNode(graphWrap.getNode(0),
                                                              graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
