// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A paged-KV SDPA forward graph through the GPU reference executor: the plan must wire the page
// tables, sequence lengths and max_seq_len_kv into the kernel and give the CPU executor's result.

#include <gtest/gtest.h>

#include <array>

#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using namespace hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

// Q/O [B=2, H=2, Sq=3, D=8]; K/V containers [5 blocks, H=2, block 2, D=8]; page tables INT32
// [B, 1, 3, 1] (max_seq_len_kv = 6); seq lens INT32 [B, 1, 1, 1]; bottom-right causal.
// UIDs: Q=1 K=2 V=3 O=4 SeqLenQ=6 SeqLenKv=7 PageTableK=8 PageTableV=9.
const std::vector<int64_t> Q_DIMS = {2, 2, 3, 8};
const std::vector<int64_t> CONTAINER_DIMS = {5, 2, 2, 8};
const std::vector<int64_t> TABLE_DIMS = {2, 1, 3, 1};
const std::vector<int64_t> LEN_DIMS = {2, 1, 1, 1};

// Each side's paging: a container with its table, or (unpaged) dense [B, H, 6, D] with no table.
struct PagedLayout
{
    bool pagedK = true;
    bool pagedV = true;
    std::vector<int64_t> kDims = CONTAINER_DIMS;
    std::vector<int64_t> vDims = CONTAINER_DIMS;
    std::vector<int64_t> kTableDims = TABLE_DIMS;
    std::vector<int64_t> vTableDims = TABLE_DIMS;
};

flatbuffers::FlatBufferBuilder makePagedGraph(const PagedLayout& layout = {})
{
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(Q_DIMS);
    const auto kStrides = hipdnn_data_sdk::utilities::generateStrides(layout.kDims);
    const auto vStrides = hipdnn_data_sdk::utilities::generateStrides(layout.vDims);
    const auto kTableStrides = hipdnn_data_sdk::utilities::generateStrides(layout.kTableDims);
    const auto vTableStrides = hipdnn_data_sdk::utilities::generateStrides(layout.vTableDims);
    const auto lenStrides = hipdnn_data_sdk::utilities::generateStrides(LEN_DIMS);

    SdpaAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.seq_len_q_tensor_uid = 6;
    attrs.seq_len_kv_tensor_uid = 7;
    if(layout.pagedK)
    {
        attrs.page_table_k_tensor_uid = 8;
    }
    if(layout.pagedV)
    {
        attrs.page_table_v_tensor_uid = 9;
    }
    attrs.padding_mask = true;
    attrs.max_seq_len_kv = 6;
    attrs.attn_scale_value = 0.35F;
    attrs.right_bound = 0;
    attrs.diagonal_alignment = DiagonalAlignment::BOTTOM_RIGHT;

    flatbuffers::FlatBufferBuilder builder;
    std::vector<flatbuffers::Offset<TensorAttributes>> tensors;
    const auto add = [&](int64_t uid,
                         const char* name,
                         DataType type,
                         const std::vector<int64_t>& strides,
                         const std::vector<int64_t>& dims) {
        tensors.push_back(CreateTensorAttributesDirect(builder, uid, name, type, &strides, &dims));
    };
    add(1, "Q", DataType::FLOAT, qStrides, Q_DIMS);
    add(2, "K", DataType::FLOAT, kStrides, layout.kDims);
    add(3, "V", DataType::FLOAT, vStrides, layout.vDims);
    add(4, "O", DataType::FLOAT, qStrides, Q_DIMS);
    add(6, "SeqLenQ", DataType::INT32, lenStrides, LEN_DIMS);
    add(7, "SeqLenKv", DataType::INT32, lenStrides, LEN_DIMS);
    if(layout.pagedK)
    {
        add(8, "PageTableK", DataType::INT32, kTableStrides, layout.kTableDims);
    }
    if(layout.pagedV)
    {
        add(9, "PageTableV", DataType::INT32, vTableStrides, layout.vTableDims);
    }

    auto sdpaAttrs = CreateSdpaAttributes(builder, &attrs);
    std::vector<flatbuffers::Offset<Node>> nodes;
    nodes.push_back(CreateNodeDirect(builder,
                                     "sdpa_fwd_node",
                                     DataType::FLOAT,
                                     NodeAttributes::SdpaAttributes,
                                     sdpaAttrs.Union()));
    auto graph = CreateGraphDirect(builder,
                                   "SdpaFwdPagedGraph",
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   DataType::FLOAT,
                                   &tensors,
                                   &nodes);
    builder.Finish(graph);
    return builder;
}

// Writes a [B, 1, T, 1] page table from its B * T block indices.
void fillPageTable(Tensor<int32_t>& table, const std::vector<int32_t>& blocks)
{
    for(size_t i = 0; i < blocks.size(); ++i)
    {
        table.memory().hostData()[i] = blocks[i];
    }
}

// Runs `layout` through the CPU and GPU executors on the same inputs (the sequence lengths of
// ExecutorMatchesCpuExecutor) and compares O. kBlocks / vBlocks fill a paged side's table.
void expectExecutorMatchesCpuExecutor(const PagedLayout& layout,
                                      const std::vector<int32_t>& kBlocks,
                                      const std::vector<int32_t>& vBlocks)
{
    auto graph = makePagedGraph(layout);

    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    ASSERT_TRUE(cpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(gpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));

    Tensor<float> q(Q_DIMS);
    Tensor<float> k(layout.kDims);
    Tensor<float> v(layout.vDims);
    q.fillWithRandomValues(-1.0F, 1.0F, 411);
    k.fillWithRandomValues(-1.0F, 1.0F, 412);
    v.fillWithRandomValues(-1.0F, 1.0F, 413);
    Tensor<int32_t> seqLenQ(LEN_DIMS);
    Tensor<int32_t> seqLenKv(LEN_DIMS);
    seqLenQ.setHostValue(3, 0, 0, 0, 0);
    seqLenQ.setHostValue(2, 1, 0, 0, 0);
    seqLenKv.setHostValue(6, 0, 0, 0, 0);
    seqLenKv.setHostValue(3, 1, 0, 0, 0);
    Tensor<int32_t> pageTableK(layout.kTableDims);
    Tensor<int32_t> pageTableV(layout.vTableDims);
    fillPageTable(pageTableK, kBlocks);
    fillPageTable(pageTableV, vBlocks);

    Tensor<float> oCpu(Q_DIMS);
    Tensor<float> oGpu(Q_DIMS);
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    const auto pack = [&](Tensor<float>& o, bool device) {
        const auto ptr = [device](auto& t) -> void* {
            return device ? t.memory().deviceData() : t.memory().hostData();
        };
        std::unordered_map<int64_t, void*> variantPack{{1, ptr(q)},
                                                       {2, ptr(k)},
                                                       {3, ptr(v)},
                                                       {4, ptr(o)},
                                                       {6, ptr(seqLenQ)},
                                                       {7, ptr(seqLenKv)}};
        if(layout.pagedK)
        {
            variantPack[8] = ptr(pageTableK);
        }
        if(layout.pagedV)
        {
            variantPack[9] = ptr(pageTableV);
        }
        return variantPack;
    };
    cpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(oCpu, false));
    gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(oGpu, true));
    oGpu.memory().markDeviceModified();

    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(1e-5F, 1e-5F);
    EXPECT_TRUE(validation.allClose(oCpu, oGpu));
}

// V paged as [3 blocks, H=2, block 4, D=8] through its own [B, 1, 2, 1] table (ceil(6 / 4)).
const std::vector<int64_t> V_CONTAINER_DIMS = {3, 2, 4, 8};
const std::vector<int64_t> V_TABLE_DIMS = {2, 1, 2, 1};
// The unpaged side is dense [B, H, max_seq_len_kv = 6, D].
const std::vector<int64_t> DENSE_KV_DIMS = {2, 2, 6, 8};

// K blocks of 2 (batch 0 in 4, 0, 2; batch 1 in 1, 3); V blocks of 4 (batch 0 in 2, 0; batch 1
// in 1). Unused table slots point at block 0.
const std::vector<int32_t> K_BLOCKS = {4, 0, 2, 1, 3, 0};
const std::vector<int32_t> V_BLOCKS = {2, 0, 1, 0};

} // namespace

TEST(TestGpuSdpaPagedFwdPlan, ExecutorMatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    auto graph = makePagedGraph();

    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpuExecutor;
    ASSERT_TRUE(gpuExecutor.isApplicable(graph.GetBufferPointer(), graph.GetSize()));

    Tensor<float> q(Q_DIMS);
    Tensor<float> k(CONTAINER_DIMS);
    Tensor<float> v(CONTAINER_DIMS);
    q.fillWithRandomValues(-1.0F, 1.0F, 401);
    k.fillWithRandomValues(-1.0F, 1.0F, 402);
    v.fillWithRandomValues(-1.0F, 1.0F, 403);
    Tensor<int32_t> seqLenQ(LEN_DIMS);
    Tensor<int32_t> seqLenKv(LEN_DIMS);
    seqLenQ.setHostValue(3, 0, 0, 0, 0);
    seqLenQ.setHostValue(2, 1, 0, 0, 0);
    seqLenKv.setHostValue(6, 0, 0, 0, 0);
    seqLenKv.setHostValue(3, 1, 0, 0, 0);
    // Batch 0 keys 0..5 live in blocks 4, 0, 2; batch 1 keys 0..2 in blocks 1 and 3 (slot 0).
    Tensor<int32_t> pageTable(TABLE_DIMS);
    const std::array<std::array<int32_t, 3>, 2> blocks = {{{4, 0, 2}, {1, 3, 0}}};
    for(int64_t b = 0; b < 2; ++b)
    {
        for(int64_t t = 0; t < 3; ++t)
        {
            pageTable.setHostValue(
                blocks[static_cast<size_t>(b)][static_cast<size_t>(t)], b, 0, t, 0);
        }
    }
    // Block 3 slot 1 holds no live key: NaN there shows up in O if either reference reads it.
    for(int64_t h = 0; h < 2; ++h)
    {
        for(int64_t d = 0; d < 8; ++d)
        {
            k.setHostValue(std::numeric_limits<float>::quiet_NaN(), 3, h, 1, d);
            v.setHostValue(std::numeric_limits<float>::quiet_NaN(), 3, h, 1, d);
        }
    }

    Tensor<float> oCpu(Q_DIMS);
    Tensor<float> oGpu(Q_DIMS);
    oGpu.fillWithValue(std::numeric_limits<float>::quiet_NaN());
    const auto pack = [&](Tensor<float>& o, bool device) {
        const auto ptr = [device](auto& t) -> void* {
            return device ? t.memory().deviceData() : t.memory().hostData();
        };
        return std::unordered_map<int64_t, void*>{{1, ptr(q)},
                                                  {2, ptr(k)},
                                                  {3, ptr(v)},
                                                  {4, ptr(o)},
                                                  {6, ptr(seqLenQ)},
                                                  {7, ptr(seqLenKv)},
                                                  {8, ptr(pageTable)},
                                                  {9, ptr(pageTable)}};
    };
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpuExecutor;
    cpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(oCpu, false));
    gpuExecutor.execute(graph.GetBufferPointer(), graph.GetSize(), pack(oGpu, true));
    oGpu.memory().markDeviceModified();

    for(size_t i = 0; i < oCpu.elementCount(); ++i)
    {
        ASSERT_FALSE(std::isnan(oCpu.memory().hostData()[i])) << "CPU read an unused slot";
    }
    const hipdnn_test_sdk::utilities::CpuFpReferenceValidation<float> validation(1e-5F, 1e-5F);
    EXPECT_TRUE(validation.allClose(oCpu, oGpu));
    // Batch 1's padded query row is zero in both.
    for(int64_t h = 0; h < 2; ++h)
    {
        EXPECT_EQ(oGpu.getHostValue(1, h, 2, 0), 0.0F);
    }
}

// K and V are paged by different tables over containers with different block sizes: each side is
// addressed through its own table and its own block size.
TEST(TestGpuSdpaPagedFwdPlan, PerSideTablesAndBlockSizesMatchCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    PagedLayout layout;
    layout.vDims = V_CONTAINER_DIMS;
    layout.vTableDims = V_TABLE_DIMS;
    expectExecutorMatchesCpuExecutor(layout, K_BLOCKS, V_BLOCKS);
}

// Only K is paged; V is dense.
TEST(TestGpuSdpaPagedFwdPlan, OnlyKPagedMatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    PagedLayout layout;
    layout.pagedV = false;
    layout.vDims = DENSE_KV_DIMS;
    expectExecutorMatchesCpuExecutor(layout, K_BLOCKS, {});
}

// Only V is paged (blocks of 4); K is dense.
TEST(TestGpuSdpaPagedFwdPlan, OnlyVPagedMatchesCpuExecutor)
{
    SKIP_IF_NO_DEVICES();
    PagedLayout layout;
    layout.pagedK = false;
    layout.kDims = DENSE_KV_DIMS;
    layout.vDims = V_CONTAINER_DIMS;
    layout.vTableDims = V_TABLE_DIMS;
    expectExecutorMatchesCpuExecutor(layout, {}, V_BLOCKS);
}
