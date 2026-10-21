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
#include "harness/gpu-graph-executor/detail/GpuSdpaPagedFwdSignatureKey.hpp"

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
constexpr int64_t RAGGED_OFFSET_O_UID = 21;
constexpr int64_t PAGE_TABLE_K_UID = 30;
constexpr int64_t PAGE_TABLE_V_UID = 31;
constexpr int64_t SEQ_LEN_KV_UID = 32;
constexpr int64_t SEQ_LEN_Q_UID = 33;

const std::vector<int64_t> Q_DIMS = hipdnn_test_sdk::utilities::raggedDims(2, 4, 2, 16);
const std::vector<int64_t> KV_DIMS = {6, 2, 8, 16};

flatbuffers::FlatBufferBuilder makePagedGraph(DataType dataType, bool raggedQ)
{
    PagedSdpaFwdGraphOptions options;
    if(raggedQ)
    {
        options.seqLenQUid = SEQ_LEN_Q_UID;
        options.raggedOffsetQUid = RAGGED_OFFSET_Q_UID;
        options.raggedOffsetOUid = RAGGED_OFFSET_O_UID;
    }
    return createPagedSdpaFwdGraph(Q_UID,
                                   K_UID,
                                   V_UID,
                                   O_UID,
                                   PAGE_TABLE_K_UID,
                                   PAGE_TABLE_V_UID,
                                   SEQ_LEN_KV_UID,
                                   /*maxSeqLenKv=*/24,
                                   Q_DIMS,
                                   KV_DIMS,
                                   KV_DIMS,
                                   Q_DIMS,
                                   dataType,
                                   options);
}

} // namespace

TEST(TestGpuSdpaPagedFwdSignatureKey, EqualityOperator)
{
    const GpuSdpaPagedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaPagedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_TRUE(key1 == key2);

    const GpuSdpaPagedFwdSignatureKey key3{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_FALSE(key1 == key3);
}

TEST(TestGpuSdpaPagedFwdSignatureKey, HashFunction)
{
    const GpuSdpaPagedFwdSignatureKey key1{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaPagedFwdSignatureKey key2{
        DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_EQ(key1.hashSelf(), key2.hashSelf());

    const GpuSdpaPagedFwdSignatureKey key3{
        DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16, DataType::BFLOAT16};
    const GpuSdpaPagedFwdSignatureKey key4{
        DataType::BFLOAT16, DataType::HALF, DataType::BFLOAT16, DataType::BFLOAT16};
    EXPECT_NE(key3.hashSelf(), key4.hashSelf());
}

TEST(TestGpuSdpaPagedFwdSignatureKey, CreateFromNodeAndTensorMap)
{
    auto graphBuilder = makePagedGraph(DataType::HALF, /*raggedQ=*/false);
    auto graphWrap = hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(
        graphBuilder.GetBufferPointer(), graphBuilder.GetSize());

    const GpuSdpaPagedFwdSignatureKey keyFromNode(graphWrap.getNode(0), graphWrap.getTensorMap());
    const GpuSdpaPagedFwdSignatureKey expectedKey{
        DataType::HALF, DataType::HALF, DataType::HALF, DataType::HALF};
    EXPECT_TRUE(keyFromNode == expectedKey);
}

TEST(TestGpuSdpaPagedFwdSignatureKey, RegistersBf16AndFp16Builders)
{
    const auto builders = GpuSdpaPagedFwdSignatureKey::getPlanBuilders();
    for(const auto dataType : {DataType::BFLOAT16, DataType::HALF})
    {
        EXPECT_EQ(
            builders.count(GpuSdpaPagedFwdSignatureKey{dataType, dataType, dataType, dataType}),
            1U);
    }
}

// Paged nodes route to the paged plan before the ragged check, so ragged Q with paged K/V is
// applicable too.
TEST(TestGpuSdpaPagedFwdSignatureKey, ExecutorRoutesPagedNodesToPagedPlan)
{
    using hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor;

    GpuReferenceGraphExecutor executor;
    for(const bool raggedQ : {false, true})
    {
        auto graph = makePagedGraph(DataType::BFLOAT16, raggedQ);
        EXPECT_TRUE(executor.isApplicable(graph.GetBufferPointer(), graph.GetSize()))
            << "raggedQ=" << raggedQ;
    }
}
