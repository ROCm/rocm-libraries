// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

TEST(TestGpuSdpaFwdSignatureKey, EqualityOperator)
{
    const GpuSdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    EXPECT_TRUE(key1 == key2);

    const GpuSdpaFwdSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key4{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT};
    EXPECT_TRUE(key3 == key4);

    // Differing output type makes the keys unequal.
    const GpuSdpaFwdSignatureKey key5{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_FALSE(key3 == key5);

    // Differing key (K) type makes the keys unequal.
    const GpuSdpaFwdSignatureKey key6{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    EXPECT_FALSE(key1 == key6);
}

TEST(TestGpuSdpaFwdSignatureKey, HashFunction)
{
    const GpuSdpaFwdSignatureKey key1{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key2{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // The same dtype placed in different fields must hash differently.
    const GpuSdpaFwdSignatureKey key3{
        DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key4{
        DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key5{
        DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey key6{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF};

    const auto hash3 = key3.hashSelf();
    const auto hash4 = key4.hashSelf();
    const auto hash5 = key5.hashSelf();
    const auto hash6 = key6.hashSelf();

    EXPECT_TRUE(hash3 != hash4 && hash3 != hash5 && hash3 != hash6 && hash4 != hash5
                && hash4 != hash6 && hash5 != hash6);
}

TEST(TestGpuSdpaFwdSignatureKey, Copy)
{
    const GpuSdpaFwdSignatureKey original{
        DataType::BFLOAT16, DataType::HALF, DataType::FLOAT, DataType::FLOAT};
    const GpuSdpaFwdSignatureKey copied{original};

    EXPECT_TRUE(original == copied);
    EXPECT_EQ(copied.qDataType, DataType::BFLOAT16);
    EXPECT_EQ(copied.kDataType, DataType::HALF);
    EXPECT_EQ(copied.vDataType, DataType::FLOAT);
    EXPECT_EQ(copied.oDataType, DataType::FLOAT);
}

TEST(TestGpuSdpaFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    constexpr int64_t Q_UID = 10;
    constexpr int64_t K_UID = 11;
    constexpr int64_t V_UID = 12;
    constexpr int64_t O_UID = 13;

    // [B=1, H=2, Sq=8, D=16], V/O head_dim_v = 16.
    const std::vector<int64_t> dims = {1, 2, 8, 16};

    auto graphBuilder
        = createSdpaFwdGraph(Q_UID, K_UID, V_UID, O_UID, dims, dims, dims, dims, DataType::FLOAT);

    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    const GpuSdpaFwdSignatureKey expectedKey{
        DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT};

    EXPECT_TRUE(keyFromNode == expectedKey);
}
