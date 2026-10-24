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

flatbuffers::FlatBufferBuilder makePagedGraph()
{
    const auto qStrides = hipdnn_data_sdk::utilities::generateStrides(Q_DIMS);
    const auto cStrides = hipdnn_data_sdk::utilities::generateStrides(CONTAINER_DIMS);
    const auto tableStrides = hipdnn_data_sdk::utilities::generateStrides(TABLE_DIMS);
    const auto lenStrides = hipdnn_data_sdk::utilities::generateStrides(LEN_DIMS);

    SdpaAttributesT attrs;
    attrs.q_tensor_uid = 1;
    attrs.k_tensor_uid = 2;
    attrs.v_tensor_uid = 3;
    attrs.o_tensor_uid = 4;
    attrs.seq_len_q_tensor_uid = 6;
    attrs.seq_len_kv_tensor_uid = 7;
    attrs.page_table_k_tensor_uid = 8;
    attrs.page_table_v_tensor_uid = 9;
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
    add(2, "K", DataType::FLOAT, cStrides, CONTAINER_DIMS);
    add(3, "V", DataType::FLOAT, cStrides, CONTAINER_DIMS);
    add(4, "O", DataType::FLOAT, qStrides, Q_DIMS);
    add(6, "SeqLenQ", DataType::INT32, lenStrides, LEN_DIMS);
    add(7, "SeqLenKv", DataType::INT32, lenStrides, LEN_DIMS);
    add(8, "PageTableK", DataType::INT32, tableStrides, TABLE_DIMS);
    add(9, "PageTableV", DataType::INT32, tableStrides, TABLE_DIMS);

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

} // namespace

TEST(TestGpuSdpaPagedFwdPlan, GpuExecutorMatchesCpuExecutor)
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
