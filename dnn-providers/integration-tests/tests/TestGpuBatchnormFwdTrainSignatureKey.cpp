// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuBatchnormFwdTrainSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuBatchnormFwdTrainSignatureKey, EqualityOperator)
{
    const GpuBatchnormFwdTrainSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuBatchnormFwdTrainSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE};
    const GpuBatchnormFwdTrainSignatureKey key4{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE};
    EXPECT_TRUE(key3 == key4);

    const GpuBatchnormFwdTrainSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::BFLOAT16, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const GpuBatchnormFwdTrainSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    const GpuBatchnormFwdTrainSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::DOUBLE};
    EXPECT_FALSE(key7 == key8);

    const GpuBatchnormFwdTrainSignatureKey key9{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key10{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key9 == key10);

    const GpuBatchnormFwdTrainSignatureKey key11{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key12{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, DataType::DOUBLE};
    EXPECT_FALSE(key11 == key12);
}

TEST(TestGpuBatchnormFwdTrainSignatureKey, HashFunction)
{
    const GpuBatchnormFwdTrainSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuBatchnormFwdTrainSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE};
    const GpuBatchnormFwdTrainSignatureKey key4{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuBatchnormFwdTrainSignatureKey key5{
        DataType::FLOAT, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT, DataType::DOUBLE};
    const GpuBatchnormFwdTrainSignatureKey key6{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::FLOAT, DataType::DOUBLE};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash3 != hash6 && hash4 != hash5
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestGpuBatchnormFwdTrainSignatureKey, Copy)
{
    const GpuBatchnormFwdTrainSignatureKey original{
        DataType::BFLOAT16, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::DOUBLE};
    const GpuBatchnormFwdTrainSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.inputDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::DOUBLE);
}

TEST(TestGpuBatchnormFwdTrainSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormFwdTrainingGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuBatchnormFwdTrainSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuBatchnormFwdTrainSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
