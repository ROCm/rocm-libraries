// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
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

// The mask dtype is part of the plan identity: keys that differ only in the mask dtype must
// compare unequal and hash apart, or a half-mask graph would resolve to the float-mask plan.
TEST(TestGpuSdpaFwdSignatureKey, MaskDataTypeDistinguishesKeys)
{
    const std::vector<DataType> maskTypes
        = {DataType::UNSET, DataType::FLOAT, DataType::HALF, DataType::BFLOAT16};

    for(size_t i = 0; i < maskTypes.size(); ++i)
    {
        const GpuSdpaFwdSignatureKey key{
            DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, maskTypes[i]};
        const GpuSdpaFwdSignatureKey sameKey{
            DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, maskTypes[i]};
        EXPECT_TRUE(key == sameKey);
        EXPECT_EQ(key.hashSelf(), sameKey.hashSelf());

        for(size_t j = i + 1; j < maskTypes.size(); ++j)
        {
            const GpuSdpaFwdSignatureKey otherKey{
                DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF, maskTypes[j]};
            EXPECT_FALSE(key == otherKey);
            EXPECT_NE(key.hashSelf(), otherKey.hashSelf());
        }
    }
}

TEST(TestGpuSdpaFwdSignatureKey, CreateFromNodeAndTensorMapWithMask)
{
    constexpr int64_t Q_UID = 10;
    constexpr int64_t K_UID = 11;
    constexpr int64_t V_UID = 12;
    constexpr int64_t O_UID = 13;
    constexpr int64_t MASK_UID = 14;

    const std::vector<int64_t> dims = {1, 2, 8, 16};
    const std::vector<int64_t> maskDims = {1, 1, 8, 8};

    for(const auto maskType : {DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
    {
        auto graphBuilder = createSdpaFwdGraph(Q_UID,
                                               K_UID,
                                               V_UID,
                                               O_UID,
                                               dims,
                                               dims,
                                               dims,
                                               dims,
                                               DataType::FLOAT,
                                               {},
                                               std::nullopt,
                                               /*scale=*/std::nullopt,
                                               MASK_UID,
                                               maskDims,
                                               maskType);
        auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
            graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

        const GpuSdpaFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());
        const GpuSdpaFwdSignatureKey expectedKey{
            DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, maskType};

        EXPECT_TRUE(keyFromNode == expectedKey);
    }
}

TEST(TestGpuSdpaFwdSignatureKey, CreateFromNodeWithDanglingMaskUidThrows)
{
    constexpr int64_t Q_UID = 10;
    constexpr int64_t K_UID = 11;
    constexpr int64_t V_UID = 12;
    constexpr int64_t O_UID = 13;
    constexpr int64_t MASK_UID = 14;

    const std::vector<int64_t> dims = {1, 2, 8, 16};

    // A mask uid with no tensor behind it.
    auto graphBuilder = createSdpaFwdGraph(Q_UID,
                                           K_UID,
                                           V_UID,
                                           O_UID,
                                           dims,
                                           dims,
                                           dims,
                                           dims,
                                           DataType::FLOAT,
                                           {},
                                           std::nullopt,
                                           /*scale=*/std::nullopt,
                                           MASK_UID,
                                           {1, 1, 8, 8},
                                           std::nullopt);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    EXPECT_THROW(GpuSdpaFwdSignatureKey(graphWrap.getNode(0), graphWrap.getTensorMap()),
                 std::out_of_range);
}

// Every registered Q/K/V/O combination has a plan builder for no mask and for each supported
// mask dtype, so any dtype mix works with any supported mask dtype.
TEST(TestGpuSdpaFwdSignatureKey, EveryQkvoCombinationRegistersEveryMaskDataType)
{
    const auto builders = GpuSdpaFwdSignatureKey::getPlanBuilders();

    const auto hasAllMasks = [&](DataType q, DataType k, DataType v, DataType o) {
        for(const auto maskType :
            {DataType::UNSET, DataType::FLOAT, DataType::HALF, DataType::BFLOAT16})
        {
            if(builders.find(GpuSdpaFwdSignatureKey{q, k, v, o, maskType}) == builders.end())
            {
                return false;
            }
        }
        return true;
    };

    EXPECT_TRUE(hasAllMasks(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT));
    EXPECT_TRUE(hasAllMasks(DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF));
    EXPECT_TRUE(hasAllMasks(
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16));
    EXPECT_TRUE(hasAllMasks(DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT));
    EXPECT_TRUE(
        hasAllMasks(DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT));
}
