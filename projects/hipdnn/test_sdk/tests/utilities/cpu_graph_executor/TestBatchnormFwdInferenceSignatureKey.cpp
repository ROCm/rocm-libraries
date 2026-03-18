// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestBatchnormFwdInferenceSignatureKey, EqualityOperator)
{
    BatchnormFwdInferenceSignatureKey const key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    BatchnormFwdInferenceSignatureKey const key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    BatchnormFwdInferenceSignatureKey const key4{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    BatchnormFwdInferenceSignatureKey const key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key5 == key6);

    BatchnormFwdInferenceSignatureKey const key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    BatchnormFwdInferenceSignatureKey const key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    BatchnormFwdInferenceSignatureKey const key11{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key12{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestBatchnormFwdInferenceSignatureKey, HashFunction)
{
    BatchnormFwdInferenceSignatureKey const key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    BatchnormFwdInferenceSignatureKey const key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    BatchnormFwdInferenceSignatureKey const key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    BatchnormFwdInferenceSignatureKey const key7{
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

TEST(TestBatchnormFwdInferenceSignatureKey, Copy)
{
    BatchnormFwdInferenceSignatureKey const original{
        DataType::FLOAT, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::BFLOAT16};
    BatchnormFwdInferenceSignatureKey const copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestBatchnormFwdInferenceSignatureKey, CreateFromNodeAndTensorMap)
{
    BatchnormFwdInferenceSignatureKey const expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> const dims = {1, 1, 1, 1};
    auto graph = buildBatchnormFwdInferenceGraph(DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 DataType::FLOAT,
                                                 dims,
                                                 TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(flatbufferGraph.data(),
                                                                         flatbufferGraph.size());

    BatchnormFwdInferenceSignatureKey const keyFromNode(graphWrap.getNode(0),
                                                        graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
