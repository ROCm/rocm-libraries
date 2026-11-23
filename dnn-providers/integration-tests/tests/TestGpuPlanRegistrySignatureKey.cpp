// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp>

#include "MatmulGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/detail/GpuPlanRegistrySignatureKey.hpp"

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

namespace
{

// Builds a matmul key from a real flatbuffer node and tensor map.
GpuPlanRegistrySignatureKey
    makeMatmulKey(DataType aType, DataType bType, DataType cType, DataType computeType)
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
                                          aType,
                                          bType,
                                          cType,
                                          computeType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    return GpuMatmulSignatureKey(graphWrapper.getNode(0), graphWrapper.getTensorMap(), computeType);
}

// Builds a batchnorm inference key from a real flatbuffer node and tensor map.
GpuPlanRegistrySignatureKey makeBatchnormInferenceKey(DataType inputType)
{
    auto builder = hipdnn_test_sdk::utilities::createValidBatchnormInferenceGraph(
        {48, 16, 4, 1}, {2, 3, 4, 4}, inputType);
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper(
        builder.GetBufferPointer(), builder.GetSize());

    return GpuBatchnormFwdInfSignatureKey(graphWrapper.getNode(0),
                                          graphWrapper.getTensorMap(),
                                          graphWrapper.getNode(0).compute_data_type());
}

} // namespace

TEST(TestGpuPlanRegistrySignatureKey, EqualKeysOfSameTypeAreEqualAndHashEqual)
{
    const auto key1
        = makeMatmulKey(DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT);
    const auto key2
        = makeMatmulKey(DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT);

    const GpuPlanRegistrySignatureKeyHash hasher;
    const GpuPlanRegistrySignatureKeyEqual eq;

    EXPECT_TRUE(eq(key1, key2));
    EXPECT_TRUE(eq(key2, key1));
    EXPECT_TRUE(eq(key1, key1));
    EXPECT_EQ(hasher(key1), hasher(key2));

    const auto bnKey1 = makeBatchnormInferenceKey(DataType::FLOAT);
    const auto bnKey2 = makeBatchnormInferenceKey(DataType::FLOAT);

    EXPECT_TRUE(eq(bnKey1, bnKey2));
    EXPECT_EQ(hasher(bnKey1), hasher(bnKey2));
}

TEST(TestGpuPlanRegistrySignatureKey, KeysOfSameTypeWithDifferentDataAreNotEqual)
{
    const auto base
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    const auto diffA
        = makeMatmulKey(DataType::HALF, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    const auto diffB
        = makeMatmulKey(DataType::FLOAT, DataType::HALF, DataType::FLOAT, DataType::FLOAT);
    const auto diffC
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::HALF, DataType::FLOAT);
    const auto diffCompute
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::HALF);

    const GpuPlanRegistrySignatureKeyEqual eq;

    EXPECT_FALSE(eq(base, diffA));
    EXPECT_FALSE(eq(base, diffB));
    EXPECT_FALSE(eq(base, diffC));
    EXPECT_FALSE(eq(base, diffCompute));
    EXPECT_FALSE(eq(diffA, base));

    const auto bnFloat = makeBatchnormInferenceKey(DataType::FLOAT);
    const auto bnHalf = makeBatchnormInferenceKey(DataType::HALF);

    EXPECT_FALSE(eq(bnFloat, bnHalf));
    EXPECT_FALSE(eq(bnHalf, bnFloat));
}

TEST(TestGpuPlanRegistrySignatureKey, KeysOfDifferentTypesAreNotEqual)
{
    const auto matmulKey
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    const auto batchnormKey = makeBatchnormInferenceKey(DataType::FLOAT);

    const GpuPlanRegistrySignatureKeyEqual eq;

    ASSERT_NE(matmulKey.index(), batchnormKey.index());
    EXPECT_FALSE(eq(matmulKey, batchnormKey));
    EXPECT_FALSE(eq(batchnormKey, matmulKey));
}

TEST(TestGpuPlanRegistrySignatureKey, UnorderedMapLookupFindsOnlyMatchingKey)
{
    const auto matmulFloat
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    const auto matmulHalf
        = makeMatmulKey(DataType::HALF, DataType::HALF, DataType::HALF, DataType::FLOAT);
    const auto bnFloat = makeBatchnormInferenceKey(DataType::FLOAT);
    const auto bnHalf = makeBatchnormInferenceKey(DataType::HALF);

    std::unordered_map<GpuPlanRegistrySignatureKey,
                       int,
                       GpuPlanRegistrySignatureKeyHash,
                       GpuPlanRegistrySignatureKeyEqual>
        registry;
    registry.emplace(matmulFloat, 1);
    registry.emplace(bnFloat, 2);

    // Re-built, equal keys find the inserted entries.
    const auto matmulFloatAgain
        = makeMatmulKey(DataType::FLOAT, DataType::FLOAT, DataType::FLOAT, DataType::FLOAT);
    const auto bnFloatAgain = makeBatchnormInferenceKey(DataType::FLOAT);

    const auto matmulIt = registry.find(matmulFloatAgain);
    ASSERT_NE(matmulIt, registry.end());
    EXPECT_EQ(matmulIt->second, 1);

    const auto bnIt = registry.find(bnFloatAgain);
    ASSERT_NE(bnIt, registry.end());
    EXPECT_EQ(bnIt->second, 2);

    // Different-valued keys of the same type are not found.
    EXPECT_EQ(registry.find(matmulHalf), registry.end());
    EXPECT_EQ(registry.find(bnHalf), registry.end());
    EXPECT_EQ(registry.size(), 2U);
}
