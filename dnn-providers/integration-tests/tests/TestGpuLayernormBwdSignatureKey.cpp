// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <hipdnn_data_sdk/utilities/Constants.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "LayernormBwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuLayernormBwdSignatureKey.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuLayernormBwdSignatureKey, EqualityOperator)
{
    const GpuLayernormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuLayernormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuLayernormBwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    const GpuLayernormBwdSignatureKey key4{
        DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF};
    EXPECT_TRUE(key3 == key4);

    EXPECT_FALSE(key1 == key3);

    const GpuLayernormBwdSignatureKey key5{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    EXPECT_FALSE(key1 == key5);
}

TEST(TestGpuLayernormBwdSignatureKey, HashFunction)
{
    const GpuLayernormBwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuLayernormBwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuLayernormBwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuLayernormBwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::HALF};
    const GpuLayernormBwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    auto hash3 = key3.hashSelf();
    auto hash4 = key4.hashSelf();
    auto hash5 = key5.hashSelf();
    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash4 != hash5);
}

TEST(TestGpuLayernormBwdSignatureKey, Copy)
{
    const GpuLayernormBwdSignatureKey original{
        DataType::FLOAT, DataType::HALF, DataType::DOUBLE, DataType::BFLOAT16, DataType::DOUBLE};
    const GpuLayernormBwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.dyDataType, DataType::FLOAT);
    EXPECT_EQ(copied.scaleBiasDataType, DataType::HALF);
    EXPECT_EQ(copied.meanInvVarianceDataType, DataType::DOUBLE);
    EXPECT_EQ(copied.dxDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.computeDataType, DataType::DOUBLE);
}

TEST(TestGpuLayernormBwdSignatureKey, CreateFromNodeAndTensorMap)
{
    constexpr int64_t DY_UID = 10;
    constexpr int64_t X_UID = 11;
    constexpr int64_t SCALE_UID = 12;
    constexpr int64_t DX_UID = 13;
    constexpr int64_t DSCALE_UID = 14;
    constexpr int64_t DBIAS_UID = 15;
    constexpr int64_t EPSILON_UID = 16;
    constexpr int64_t MEAN_UID = 17;
    constexpr int64_t INV_VARIANCE_UID = 18;

    const std::vector<int64_t> ioDims = {2, 3, 4, 5};
    const TensorLayout layout = TensorLayout::NCHW;
    const double epsilon = LAYERNORM_DEFAULT_EPSILON;
    const int64_t normalizedDimCount = 2;

    auto graphBuilder = createLayernormBwdGraph(DY_UID,
                                                X_UID,
                                                SCALE_UID,
                                                DX_UID,
                                                DSCALE_UID,
                                                DBIAS_UID,
                                                EPSILON_UID,
                                                MEAN_UID,
                                                INV_VARIANCE_UID,
                                                ioDims,
                                                layout,
                                                epsilon,
                                                normalizedDimCount,
                                                DataType::FLOAT,
                                                DataType::BFLOAT16,
                                                DataType::HALF,
                                                DataType::DOUBLE,
                                                DataType::DOUBLE);

    auto graphWrapper = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuLayernormBwdSignatureKey keyFromNode(
        graphWrapper.getNode(0), graphWrapper.getTensorMap(), DataType::DOUBLE);

    const GpuLayernormBwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::HALF, DataType::DOUBLE, DataType::BFLOAT16, DataType::DOUBLE};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
