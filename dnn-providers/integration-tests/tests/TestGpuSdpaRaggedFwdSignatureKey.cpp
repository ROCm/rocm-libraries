// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaTestUtils.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaFwdPlan.hpp"
#include "harness/gpu-graph-executor/detail/GpuSdpaRaggedFwdSignatureKey.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using namespace hipdnn_integration_tests::test_utils;
using namespace hipdnn_integration_tests::gpu_graph_executor::detail;

namespace
{

constexpr int64_t Q_UID = 10;
constexpr int64_t K_UID = 11;
constexpr int64_t V_UID = 12;
constexpr int64_t O_UID = 13;
constexpr int64_t RAGGED_OFFSET_Q_UID = 20;
constexpr int64_t RAGGED_OFFSET_KV_UID = 21;

// One ragged batch. Only dtypes and ragged_offset matter here, not the shape.
const std::vector<int64_t> DIMS = hipdnn_test_sdk::utilities::raggedDims(1, 8, 2, 16);

flatbuffers::FlatBufferBuilder makeRaggedGraph(DataType dataType)
{
    return createRaggedSdpaFwdGraph(Q_UID,
                                    K_UID,
                                    V_UID,
                                    O_UID,
                                    RAGGED_OFFSET_Q_UID,
                                    RAGGED_OFFSET_KV_UID,
                                    /*batch=*/1,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    DIMS,
                                    dataType);
}

// The ragged bf16 graph with Q's ragged_offset removed: K, V and O stay ragged.
flatbuffers::FlatBufferBuilder makeGraphWithDenseQ()
{
    auto ragged = makeRaggedGraph(DataType::BFLOAT16);
    auto graph = UnPackGraph(ragged.GetBufferPointer());
    for(auto& tensor : graph->tensors)
    {
        if(tensor->uid == Q_UID)
        {
            tensor->ragged_offset_tensor_uid = flatbuffers::nullopt;
        }
    }
    flatbuffers::FlatBufferBuilder builder;
    builder.Finish(Graph::Pack(builder, graph.get()));
    return builder;
}

} // namespace

TEST(TestGpuSdpaRaggedFwdSignatureKey, EqualityOperator)
{
    const GpuSdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_TRUE(key1 == key2);

    // Differing output type makes the keys unequal.
    const GpuSdpaRaggedFwdSignatureKey key3{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::FLOAT};
    EXPECT_FALSE(key1 == key3);
}

TEST(TestGpuSdpaRaggedFwdSignatureKey, HashFunction)
{
    const GpuSdpaRaggedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    // The same dtype placed in different fields must hash differently.
    const GpuSdpaRaggedFwdSignatureKey key3{
        DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaRaggedFwdSignatureKey key4{
        DataType::BFLOAT16, DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestGpuSdpaRaggedFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto graphBuilder = makeRaggedGraph(DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaRaggedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());

    const GpuSdpaRaggedFwdSignatureKey expectedKey{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};

    EXPECT_TRUE(keyFromNode == expectedKey);
}

// Ragged and dense nodes are both applicable, each through its own plan (checked below).
TEST(TestGpuSdpaRaggedFwdSignatureKey, ExecutorRoutesRaggedBf16NodeToRaggedPlan)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    auto raggedBuilder = makeRaggedGraph(DataType::BFLOAT16);
    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(raggedBuilder.GetBufferPointer(), raggedBuilder.GetSize()));

    // Same shape/dtype but no ragged_offset: a dense node, handled by the dense plan.
    auto denseBuilder = createSdpaFwdGraph(
        Q_UID, K_UID, V_UID, O_UID, DIMS, DIMS, DIMS, DIMS, DataType::BFLOAT16);
    EXPECT_TRUE(executor.isApplicable(denseBuilder.GetBufferPointer(), denseBuilder.GetSize()));
}

// The dense plan must not take a ragged node, or it would read packed data as dense.
TEST(TestGpuSdpaRaggedFwdSignatureKey, DensePlanRejectsRaggedNode)
{
    auto graphBuilder = makeRaggedGraph(DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());
    const GpuSdpaFwdPlanBuilder<DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16,
                                DataType::BFLOAT16>
        denseBuilder;
    EXPECT_FALSE(denseBuilder.isApplicable(graphWrap.getNode(0), graphWrap.getTensorMap()));
}

// Dense Q with ragged K/V/O fits neither plan. It must not be run by the dense one.
TEST(TestGpuSdpaRaggedFwdSignatureKey, ExecutorRejectsPartlyRaggedNode)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    auto graphBuilder = makeGraphWithDenseQ();
    GpuReferenceGraphExecutor executor;
    EXPECT_FALSE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}

// fp8 Q/K/V with bf16 O and scalar descales keys and routes to the ragged plan too.
TEST(TestGpuSdpaRaggedFwdSignatureKey, Fp8NodeKeyAndRouting)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

    RaggedSdpaFwdGraphOptions options;
    options.descaleQ = FloatOperandSpec{DESCALE_Q_UID};
    options.descaleK = FloatOperandSpec{DESCALE_K_UID};
    options.descaleV = FloatOperandSpec{DESCALE_V_UID};
    options.oDataType = DataType::BFLOAT16;
    auto graphBuilder = createRaggedSdpaFwdGraph(Q_UID,
                                                 K_UID,
                                                 V_UID,
                                                 O_UID,
                                                 RAGGED_OFFSET_Q_UID,
                                                 RAGGED_OFFSET_KV_UID,
                                                 /*batch=*/1,
                                                 DIMS,
                                                 DIMS,
                                                 DIMS,
                                                 DIMS,
                                                 DataType::FP8_E4M3,
                                                 options);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaRaggedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());
    const GpuSdpaRaggedFwdSignatureKey expectedKey{
        DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16};
    EXPECT_TRUE(keyFromNode == expectedKey);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}
