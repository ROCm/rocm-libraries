// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "harness/gpu-graph-executor/detail/GpuConvolutionWrwSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuConvolutionWrwSignatureKey, EqualityOperator)
{
    const GpuConvolutionWrwSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuConvolutionWrwSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const GpuConvolutionWrwSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const GpuConvolutionWrwSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestGpuConvolutionWrwSignatureKey, HashFunction)
{
    const GpuConvolutionWrwSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuConvolutionWrwSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const GpuConvolutionWrwSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestGpuConvolutionWrwSignatureKey, Copy)
{
    const GpuConvolutionWrwSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionWrwSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::FLOAT);
    EXPECT_EQ(copied.dyDataType, DataType::HALF);
    EXPECT_EQ(copied.dwDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::FLOAT);
}

TEST(TestGpuConvolutionWrwSignatureKey, CreateFromNodeAndTensorMap)
{
    auto builder = hipdnn_test_sdk::utilities::createValidConvWrwGraph();
    auto graph = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    const GpuConvolutionWrwSignatureKey keyFromNode(
        graph.getNode(0), graph.getTensorMap(), graph.getNode(0).compute_data_type());
    const GpuConvolutionWrwSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
