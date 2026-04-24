// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "BatchnormGraphUtils.hpp"
#include "BatchnormTensorBundles.hpp"
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormBwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestBatchnormBwdSignatureKey, EqualityOperator)
{
    const BatchnormBwdSignatureKey key1{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key2{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const BatchnormBwdSignatureKey key3{DataType::HALF,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key4{DataType::HALF,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const BatchnormBwdSignatureKey key5{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key6{DataType::HALF,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const BatchnormBwdSignatureKey key7{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key8{DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const BatchnormBwdSignatureKey key9{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key10{DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::DOUBLE,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    const BatchnormBwdSignatureKey key11{DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT};
    const BatchnormBwdSignatureKey key12{DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::FLOAT,
                                         DataType::DOUBLE};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestBatchnormBwdSignatureKey, HashFunction)
{
    const BatchnormBwdSignatureKey key1{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key2{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const BatchnormBwdSignatureKey key3{DataType::HALF,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key4{DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key5{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key6{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key7{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF,
                                        DataType::FLOAT};
    const BatchnormBwdSignatureKey key8{DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::FLOAT,
                                        DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();
    auto hash7 = key7.hashSelf();
    auto hash8 = key8.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6 && hash3 != hash7 && hash4 != hash7
                && hash5 != hash7 && hash6 != hash7 && hash3 != hash8 && hash4 != hash8
                && hash5 != hash8 && hash6 != hash8 && hash7 != hash8);
}

TEST(TestBatchnormBwdSignatureKey, Copy)
{
    const BatchnormBwdSignatureKey original{DataType::BFLOAT16,
                                            DataType::FLOAT,
                                            DataType::HALF,
                                            DataType::DOUBLE,
                                            DataType::FLOAT,
                                            DataType::BFLOAT16};
    const BatchnormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestBatchnormBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    const BatchnormBwdSignatureKey expectedKey{DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT,
                                               DataType::FLOAT};
    const std::vector<int64_t> dims = {2, 1, 1, 1};
    BatchnormBwdTensorBundle<float, float, float> tensorBundle(dims, 1, TensorLayout::NCHW);

    auto graphTuple = buildBatchnormBwdGraph(
        tensorBundle, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);

    auto& graph = std::get<0>(graphTuple);
    auto [serializedGraph, serErr] = graph->to_binary();
    ASSERT_TRUE(serErr.is_good()) << serErr.get_message();

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        serializedGraph.data(), serializedGraph.size());

    const BatchnormBwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
