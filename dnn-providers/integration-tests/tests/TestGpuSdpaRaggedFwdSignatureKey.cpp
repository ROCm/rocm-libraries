// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include "SdpaFwdGraphTestUtils.hpp"
#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"
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

// Packed rank-4 [B=1, H=2, S=8, D=16] for a single ragged batch. Shape is irrelevant to keying/
// dispatch; only dtypes and the presence of ragged_offset on the primaries matter here.
const std::vector<int64_t> DIMS = {1, 2, 8, 16};

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

// The core dispatch contract, observed through the public executor surface: a bf16 node whose Q
// primary carries a ragged_offset_tensor_uid is applicable, and it can ONLY be applicable via the
// ragged plan because the dense plan rejects any primary with ragged_offset set. So applicability
// here proves the node was routed to (and accepted by) the ragged reference rather than the dense
// one.
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

// The fp8 combo (FP8_E4M3 Q/K/V -> BFLOAT16 O, with Q/K/V descale) keys and routes to the ragged
// plan the same way. Descale tensors are scalar [1] FLOAT.
TEST(TestGpuSdpaRaggedFwdSignatureKey, Fp8NodeKeyAndRouting)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    constexpr int64_t DESCALE_Q_UID = 30;
    constexpr int64_t DESCALE_K_UID = 31;
    constexpr int64_t DESCALE_V_UID = 32;

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
                                                 /*attrs=*/{},
                                                 /*statsUid=*/std::nullopt,
                                                 DESCALE_Q_UID,
                                                 DESCALE_K_UID,
                                                 DESCALE_V_UID,
                                                 /*oDataType=*/DataType::BFLOAT16);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaRaggedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());
    const GpuSdpaRaggedFwdSignatureKey expectedKey{
        DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::FP8_E4M3, DataType::BFLOAT16};
    EXPECT_TRUE(keyFromNode == expectedKey);

    GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graphBuilder.GetBufferPointer(), graphBuilder.GetSize()));
}
