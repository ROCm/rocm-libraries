// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "LayernormFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormFwdSignatureKey.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuLayernormFwdSignatureKey, EqualityOperator)
{
    const GpuLayernormFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuLayernormFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuLayernormFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const GpuLayernormFwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    EXPECT_FALSE(key1 == key3);

    const GpuLayernormFwdSignatureKey key5{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key1 == key5);
}

TEST(TestGpuLayernormFwdSignatureKey, HashFunction)
{
    const GpuLayernormFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuLayernormFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuLayernormFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuLayernormFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const GpuLayernormFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestGpuLayernormFwdSignatureKey, Copy)
{
    const GpuLayernormFwdSignatureKey original{
        DataType::BFLOAT16, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::DOUBLE};
    const GpuLayernormFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.xDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanInvVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.yDataType, DataType::FLOAT);
    EXPECT_EQ(copied.computeDataType, DataType::DOUBLE);
}

TEST(TestGpuLayernormFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    constexpr int64_t X_UID = 10;
    constexpr int64_t Y_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t BIAS_UID = 13;
    constexpr int64_t EPSILON_UID = 14;
    constexpr int64_t MEAN_UID = 15;
    constexpr int64_t INV_VARIANCE_UID = 16;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormFwdGraph(X_UID,
                                                Y_UID,
                                                SCALE_UID,
                                                BIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                DataType::BFLOAT16,
                                                DataType::FLOAT,
                                                DataType::HALF,
                                                DataType::DOUBLE,
                                                DataType::DOUBLE);

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuLayernormFwdSignatureKey keyFromNode(
        graphWrapper.getNode(0), graphWrapper.getTensorMap(), DataType::DOUBLE);

    const GpuLayernormFwdSignatureKey expectedKey{
        DataType::BFLOAT16, DataType::HALF, DataType::DOUBLE, DataType::FLOAT, DataType::DOUBLE};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
