// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuRMSNormSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

// =============================================================
// Test GpuRMSNormFwdSignatureKey
// =============================================================

TEST(TestGpuRMSNormFwdSignatureKey, EqualityOperator)
{
    const GpuRMSNormFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuRMSNormFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const GpuRMSNormFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const GpuRMSNormFwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);

    const GpuRMSNormFwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key10{
        DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);
}

TEST(TestGpuRMSNormFwdSignatureKey, HashFunction)
{
    const GpuRMSNormFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuRMSNormFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuRMSNormFwdSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5 && hash3 != hash6
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestGpuRMSNormFwdSignatureKey, Copy)
{
    const GpuRMSNormFwdSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::BFLOAT16};
    const GpuRMSNormFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleDataType, DataType::HALF);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestGpuRMSNormFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormFwdSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuRMSNormFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}

// =============================================================
// Test GpuRMSNormBwdSignatureKey
// =============================================================

TEST(TestGpuRMSNormBwdSignatureKey, EqualityOperator)
{
    const GpuRMSNormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuRMSNormBwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const GpuRMSNormBwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    const GpuRMSNormBwdSignatureKey key5{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key5);

    const GpuRMSNormBwdSignatureKey key6{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key6);

    const GpuRMSNormBwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key7);

    const GpuRMSNormBwdSignatureKey key8{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key1 == key8);

    const GpuRMSNormBwdSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key1 == key9);
}

TEST(TestGpuRMSNormBwdSignatureKey, HashFunction)
{
    const GpuRMSNormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuRMSNormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    std::vector<std::size_t> hashes;
    hashes.push_back(
        GpuRMSNormBwdSignatureKey(
            DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuRMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuRMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuRMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuRMSNormBwdSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF)
            .hashSelf());

    for(size_t i = 0; i < hashes.size(); ++i)
    {
        for(size_t j = i + 1; j < hashes.size(); ++j)
        {
            EXPECT_NE(hashes[i], hashes[j]);
        }
    }
}

TEST(TestGpuRMSNormBwdSignatureKey, Copy)
{
    const GpuRMSNormBwdSignatureKey original{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const GpuRMSNormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.gradOutputDataType, DataType::HALF);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleDataType, DataType::HALF);
    EXPECT_EQ(copied.gradInputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestGpuRMSNormBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidRMSNormBwdGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuRMSNormBwdSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuRMSNormBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
