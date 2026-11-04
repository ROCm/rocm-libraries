// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "MatmulGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuMatmulSignatureKey.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuMatmulSignatureKey, EqualityOperator)
{
    const GpuMatmulSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuMatmulSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuMatmulSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuMatmulSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);
    EXPECT_FALSE(key1 == key3);

    const GpuMatmulSignatureKey key5{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_FALSE(key1 == key5);
}

TEST(TestGpuMatmulSignatureKey, HashFunction)
{
    const GpuMatmulSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuMatmulSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuMatmulSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuMatmulSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestGpuMatmulSignatureKey, Copy)
{
    const GpuMatmulSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE};
    const GpuMatmulSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(original.aDataType, copied.aDataType);
    EXPECT_EQ(original.bDataType, copied.bDataType);
    EXPECT_EQ(original.cDataType, copied.cDataType);
    EXPECT_EQ(original.computeDataType, copied.computeDataType);
}

TEST(TestGpuMatmulSignatureKey, CreateFromNodeAndTensorMap)
{
    constexpr int64_t A_UID = 10;
    constexpr int64_t B_UID = 11;
    constexpr int64_t C_UID = 12;
    const std::vector<int64_t> dims = {4, 8, 2, 2};
    const std::vector<int64_t> strides = generateStrides(dims);

    auto graphBuilder = createMatmulGraph(A_UID,
                                          B_UID,
                                          C_UID,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          dims,
                                          strides,
                                          DataType::FLOAT,
                                          DataType::HALF,
                                          DataType::BFLOAT16,
                                          DataType::DOUBLE);
    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuMatmulSignatureKey keyFromNode(
        graphWrapper.getNode(0), graphWrapper.getTensorMap(), DataType::DOUBLE);
    const GpuMatmulSignatureKey expectedKey{
        DataType::FLOAT, DataType::HALF, DataType::BFLOAT16, DataType::DOUBLE};
    EXPECT_TRUE(keyFromNode == expectedKey);
}
