// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <unordered_map>
#include <unordered_set>

#include "RMSNormGraphUtils.hpp"
#include "RMSNormTensorBundles.hpp"
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdSignatureKey.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_test_sdk::detail;
using namespace hipdnn_data_sdk::data_objects;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_sdk_test_utils;

TEST(TestRMSNormFwdSignatureKey, EqualityOperator)
{
    RMSNormFwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    RMSNormFwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    RMSNormFwdSignatureKey key4{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    RMSNormFwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key6{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    RMSNormFwdSignatureKey key7{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key8{DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    RMSNormFwdSignatureKey key9{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestRMSNormFwdSignatureKey, HashFunction)
{
    RMSNormFwdSignatureKey key1{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key2{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    RMSNormFwdSignatureKey key3{DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    RMSNormFwdSignatureKey key4{DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    RMSNormFwdSignatureKey key5{DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    RMSNormFwdSignatureKey key6{DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestRMSNormFwdSignatureKey, Copy)
{
    RMSNormFwdSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::BFLOAT16};
    RMSNormFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleDataType, DataType::HALF);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestRMSNormFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    RMSNormFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    std::vector<int64_t> dims = {1, 1, 1, 1};
    auto graph = buildRMSNormFwdGraph(
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, dims, TensorLayout::NHWC);
    auto flatbufferGraph = graph->buildFlatbufferOperationGraph();
    auto graphWrap = hipdnn_data_sdk::flatbuffer_utilities::GraphWrapper(flatbufferGraph.data(),
                                                                         flatbufferGraph.size());

    RMSNormFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    EXPECT_TRUE(keyFromNode == expectedKey);
}
