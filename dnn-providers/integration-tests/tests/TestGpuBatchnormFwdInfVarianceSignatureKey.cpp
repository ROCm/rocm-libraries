// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuBatchnormFwdInfVarianceSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuBatchnormFwdInfVarianceSignatureKey, EqualityOperator)
{
    const GpuBatchnormFwdInfVarianceSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdInfVarianceSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuBatchnormFwdInfVarianceSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuBatchnormFwdInfVarianceSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const GpuBatchnormFwdInfVarianceSignatureKey key5{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key5);

    const GpuBatchnormFwdInfVarianceSignatureKey key6{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key6);

    const GpuBatchnormFwdInfVarianceSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key7);

    const GpuBatchnormFwdInfVarianceSignatureKey key8{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key1 == key8);

    const GpuBatchnormFwdInfVarianceSignatureKey key9{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key1 == key9);
}

TEST(TestGpuBatchnormFwdInfVarianceSignatureKey, HashFunction)
{
    const GpuBatchnormFwdInfVarianceSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdInfVarianceSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    std::vector<std::size_t> hashes;
    hashes.push_back(
        GpuBatchnormFwdInfVarianceSignatureKey(
            DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuBatchnormFwdInfVarianceSignatureKey(
            DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuBatchnormFwdInfVarianceSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuBatchnormFwdInfVarianceSignatureKey(
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT)
            .hashSelf());
    hashes.push_back(
        GpuBatchnormFwdInfVarianceSignatureKey(
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

TEST(TestGpuBatchnormFwdInfVarianceSignatureKey, Copy)
{
    const GpuBatchnormFwdInfVarianceSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::FLOAT, DataType::BFLOAT16};
    const GpuBatchnormFwdInfVarianceSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::BFLOAT16);
}

TEST(TestGpuBatchnormFwdInfVarianceSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormWithVarianceInferenceGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdInfVarianceSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuBatchnormFwdInfVarianceSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
