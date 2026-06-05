// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "ConvolutionFwdGraphTestUtils.hpp"
#include "harness/gpu_graph_executor/detail/GpuConvolutionFwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuConvolutionFwdSignatureKey, EqualityOperator)
{
    const GpuConvolutionFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuConvolutionFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    const GpuConvolutionFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key6{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key5 == key6);

    const GpuConvolutionFwdSignatureKey key7{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key8{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key7 == key8);
}

TEST(TestGpuConvolutionFwdSignatureKey, HashFunction)
{
    const GpuConvolutionFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuConvolutionFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const GpuConvolutionFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuConvolutionFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};

    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestGpuConvolutionFwdSignatureKey, Copy)
{
    const GpuConvolutionFwdSignatureKey original{
        DataType::BFLOAT16, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const GpuConvolutionFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.wDataType, DataType::FLOAT);
    EXPECT_EQ(copied.outputDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::HALF);
}

TEST(TestGpuConvolutionFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t W_UID = 11;
    constexpr int64_t Y_UID = 12;

    const std::vector<int64_t> xDims = {1, 1, 2, 2};
    const std::vector<int64_t> wDims = {1, 1, 1, 1};
    const std::vector<int64_t> yDims = {1, 1, 2, 2};

    auto graphBuilder = createConvFwdGraph(X_UID,
                                           W_UID,
                                           Y_UID,
                                           xDims,
                                           wDims,
                                           yDims,
                                           generateStrides(xDims),
                                           generateStrides(wDims),
                                           generateStrides(yDims),
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           DataType::FLOAT);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuConvolutionFwdSignatureKey keyFromNode(
        graphWrap.getNode(0), graphWrap.getTensorMap(), DataType::FLOAT);

    const GpuConvolutionFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
