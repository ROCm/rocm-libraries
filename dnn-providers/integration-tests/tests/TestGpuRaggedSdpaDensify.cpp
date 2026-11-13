// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Ragged SDPA graphs that no native ragged plan accepts (backward, or a forward with a bias) run
// through the ragged-as-dense rewrite in both executors; the GPU result must equal the CPU one.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/RaggedLayoutTestSupport.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using namespace hipdnn_test_sdk::utilities::ragged_densify_test;
using hipdnn_test_sdk::utilities::ragged_layout_test::stridedAuxBuffer;

namespace
{

// Runs `graph` on the CPU executor (host buffers) and the GPU executor (device copies of the same
// buffers) and compares every output buffer element by element. The token offsets are always
// bound; intInputs adds INT32 inputs (seq_len tensors, page tables). With auxStride > 1 the token
// offsets and seq_len values sit at that batch stride in both executors' buffers (the graph must
// declare it, see withAuxStrides), the slots between them poisoned.
void expectGpuMatchesCpu(std::vector<uint8_t> graph,
                         const std::unordered_map<int64_t, std::vector<float>>& inputs,
                         const std::unordered_map<int64_t, size_t>& outputSizes,
                         float tolerance,
                         std::unordered_map<int64_t, std::vector<int32_t>> intInputs = {},
                         int64_t auxStride = 1)
{
    intInputs.emplace(Q_OFFSETS, Q_TOKENS);
    intInputs.emplace(KV_OFFSETS, KV_TOKENS);
    for(const int64_t uid : {Q_OFFSETS, KV_OFFSETS, SEQ_Q, SEQ_KV})
    {
        const auto it = intInputs.find(uid);
        if(it == intInputs.end() || auxStride == 1)
        {
            continue;
        }
        it->second = stridedAuxBuffer(it->second, auxStride);
    }

    // CPU: host buffers.
    std::unordered_map<int64_t, std::vector<float>> cpuData = inputs;
    std::unordered_map<int64_t, void*> cpuPack;
    for(auto& [uid, values] : intInputs)
    {
        cpuPack[uid] = values.data();
    }
    for(auto& [uid, values] : cpuData)
    {
        cpuPack[uid] = values.data();
    }
    for(const auto& [uid, size] : outputSizes)
    {
        cpuData[uid].assign(size, std::numeric_limits<float>::quiet_NaN());
        cpuPack[uid] = cpuData[uid].data();
    }
    hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor cpu;
    ASSERT_TRUE(cpu.isApplicable(graph.data(), graph.size()));
    cpu.execute(graph.data(), graph.size(), cpuPack);

    // GPU: the same values in device memory.
    std::unordered_map<int64_t, Tensor<float>> gpuFloat;
    std::unordered_map<int64_t, void*> gpuPack;
    for(const auto& [uid, values] : inputs)
    {
        auto& t = gpuFloat.emplace(uid, Tensor<float>({static_cast<int64_t>(values.size())}))
                      .first->second;
        std::memcpy(t.memory().hostData(), values.data(), values.size() * sizeof(float));
        gpuPack[uid] = t.memory().deviceData();
    }
    for(const auto& [uid, size] : outputSizes)
    {
        auto& t = gpuFloat.emplace(uid, Tensor<float>({static_cast<int64_t>(size)})).first->second;
        t.fillWithValue(std::numeric_limits<float>::quiet_NaN());
        gpuPack[uid] = t.memory().deviceData();
    }
    std::unordered_map<int64_t, Tensor<int32_t>> gpuInt;
    for(const auto& [uid, values] : intInputs)
    {
        auto& t = gpuInt.emplace(uid, Tensor<int32_t>({static_cast<int64_t>(values.size())}))
                      .first->second;
        std::memcpy(t.memory().hostData(), values.data(), values.size() * sizeof(int32_t));
        gpuPack[uid] = t.memory().deviceData();
    }

    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpu;
    ASSERT_TRUE(gpu.isApplicable(graph.data(), graph.size()));
    gpu.execute(graph.data(), graph.size(), gpuPack);

    for(const auto& [uid, size] : outputSizes)
    {
        auto& t = gpuFloat.at(uid);
        t.memory().markDeviceModified();
        const auto* got = t.memory().hostData();
        const auto& want = cpuData.at(uid);
        for(size_t i = 0; i < size; ++i)
        {
            ASSERT_FALSE(std::isnan(want[i])) << "CPU left output " << uid << "[" << i << "]";
            EXPECT_NEAR(got[i], want[i], tolerance) << "output " << uid << "[" << i << "]";
        }
    }
}

// The forward graph with seq_len_q/kv bound to the node (INT32 [B, 1, 1, 1], uids SEQ_Q/SEQ_KV).
std::vector<uint8_t> withSeqLens(const std::vector<uint8_t>& graph)
{
    return editGraph(graph, [](GraphT& g) {
        for(const int64_t uid : {SEQ_Q, SEQ_KV})
        {
            g.tensors.push_back(tensor(uid, DataType::INT32, {B, 1, 1, 1}, {1, 1, 1, 1}));
        }
        auto* attrs = g.nodes.at(0)->attributes.AsSdpaAttributes();
        attrs->seq_len_q_tensor_uid = SEQ_Q;
        attrs->seq_len_kv_tensor_uid = SEQ_KV;
        attrs->padding_mask = true;
    });
}

// The graph with the token offsets and seq_len tensors (those it has) at batch stride `stride`.
std::vector<uint8_t> withAuxStrides(const std::vector<uint8_t>& graph, int64_t stride)
{
    return editGraph(graph, [&](GraphT& g) {
        for(auto& attrs : g.tensors)
        {
            if(attrs->uid == Q_OFFSETS || attrs->uid == KV_OFFSETS || attrs->uid == SEQ_Q
               || attrs->uid == SEQ_KV)
            {
                attrs->strides = {stride, 1, 1, 1};
            }
        }
    });
}

} // namespace

// seq_len shorter than the offset block: the live K/V rows are seq_len_kv, not the block, so the
// padding rows of each block must not reach the softmax. Bias keeps the graph on the densify
// path.
TEST(TestGpuRaggedSdpaDensify, SeqLenShorterThanBlockMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    std::vector<float> bias(static_cast<size_t>(SQ * SKV));
    for(size_t i = 0; i < bias.size(); ++i)
    {
        bias[i] = static_cast<float>(i % 5) * 0.25F - 0.5F;
    }
    expectGpuMatchesCpu(
        withSeqLens(raggedGraph(false, true)),
        {{Q, packedValues(10, H * D, 61)},
         {K, packedValues(13, H * D, 62)},
         {V, packedValues(13, H * D, 63)},
         {BIAS, bias}},
        {{O, static_cast<size_t>(10 * H * D)}, {STATS, static_cast<size_t>(10 * H)}},
        1e-5F,
        {{SEQ_Q, {4, 6}}, {SEQ_KV, {6, 3}}});
}

// The same, with the token offsets and seq_len tensors at batch stride 2 and the odd slots
// poisoned: both executors' densify paths must read element b at b * 2.
TEST(TestGpuRaggedSdpaDensify, StridedSeqLenMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    std::vector<float> bias(static_cast<size_t>(SQ * SKV));
    for(size_t i = 0; i < bias.size(); ++i)
    {
        bias[i] = static_cast<float>(i % 5) * 0.25F - 0.5F;
    }
    expectGpuMatchesCpu(
        withAuxStrides(withSeqLens(raggedGraph(false, true)), 2),
        {{Q, packedValues(10, H * D, 61)},
         {K, packedValues(13, H * D, 62)},
         {V, packedValues(13, H * D, 63)},
         {BIAS, bias}},
        {{O, static_cast<size_t>(10 * H * D)}, {STATS, static_cast<size_t>(10 * H)}},
        1e-5F,
        {{SEQ_Q, {4, 6}}, {SEQ_KV, {6, 3}}},
        2);
}

TEST(TestGpuRaggedSdpaDensify, ForwardWithBiasMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    std::vector<float> bias(static_cast<size_t>(SQ * SKV));
    for(size_t i = 0; i < bias.size(); ++i)
    {
        bias[i] = static_cast<float>(i % 7) * 0.25F - 0.75F;
    }
    expectGpuMatchesCpu(
        raggedGraph(false, true),
        {{Q, packedValues(10, H * D, 31)},
         {K, packedValues(13, H * D, 32)},
         {V, packedValues(13, H * D, 33)},
         {BIAS, bias}},
        {{O, static_cast<size_t>(10 * H * D)}, {STATS, static_cast<size_t>(10 * H)}},
        1e-5F);
}

// Philox dropout over packed operands, validated through the dense RNG dump (SEM-015): both
// executors apply the dump through the dense rewrite.
TEST(TestGpuRaggedSdpaDensify, PhiloxDropoutWithItsDumpMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    std::vector<float> dump(static_cast<size_t>(B * H * SQ * SKV));
    for(size_t i = 0; i < dump.size(); ++i)
    {
        dump[i] = (i * 7U + i / 5U) % 4U == 0U ? 0.0F : 1.0F;
    }
    // The INT64 seed and offset, each as two INT32 words (little-endian): bound, not read.
    expectGpuMatchesCpu(
        withPhiloxDropout(raggedGraph(false, false), 0.25F),
        {{Q, packedValues(10, H * D, 51)},
         {K, packedValues(13, H * D, 52)},
         {V, packedValues(13, H * D, 53)},
         {RNG_DUMP, dump}},
        {{O, static_cast<size_t>(10 * H * D)}, {STATS, static_cast<size_t>(10 * H)}},
        1e-5F,
        {{SEED, {1234, 0}}, {OFFSET, {8, 0}}});
}

TEST(TestGpuRaggedSdpaDensify, BackwardMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    // O and the LSE need not come from a forward pass for a CPU/GPU comparison; any finite values
    // exercise the same arithmetic in both executors.
    std::vector<float> lse(static_cast<size_t>(10 * H));
    for(size_t i = 0; i < lse.size(); ++i)
    {
        lse[i] = 1.0F + 0.1F * static_cast<float>(i % 5);
    }
    expectGpuMatchesCpu(raggedGraph(true, false),
                        {{Q, packedValues(10, H * D, 41)},
                         {K, packedValues(13, H * D, 42)},
                         {V, packedValues(13, H * D, 43)},
                         {O, packedValues(10, H * D, 44)},
                         {DO, packedValues(10, H * D, 45)},
                         {STATS, lse}},
                        {{DQ, static_cast<size_t>(10 * H * D)},
                         {DK, static_cast<size_t>(13 * H * D)},
                         {DV, static_cast<size_t>(13 * H * D)}},
                        1e-4F);
}

// vLLM's shape: ragged Q over a paged K/V cache, with and without a sink, through the rewrite.
// The unreferenced cache block holds NaN.
TEST(TestGpuRaggedSdpaDensify, RaggedQOverPagedKvMatchesCpu)
{
    SKIP_IF_NO_DEVICES();
    const auto cacheSize = static_cast<size_t>(NUM_BLOCKS * H * BLOCK * D);
    const auto cache = [&](unsigned seed) {
        auto values = packedValues(NUM_BLOCKS * BLOCK, H * D, seed);
        for(size_t i = 0; i < cacheSize; ++i)
        {
            if(static_cast<int64_t>(i) / (H * BLOCK * D) == 2)
            {
                values[i] = std::numeric_limits<float>::quiet_NaN();
            }
        }
        return values;
    };
    std::vector<int32_t> seqQ;
    std::vector<int32_t> seqKv;
    for(int64_t b = 0; b < B; ++b)
    {
        seqQ.push_back(static_cast<int32_t>(Q_LENS[static_cast<size_t>(b)]));
        seqKv.push_back(static_cast<int32_t>(KV_LENS[static_cast<size_t>(b)]));
    }
    for(const bool sink : {false, true})
    {
        SCOPED_TRACE(sink ? "with sink" : "no sink");
        std::unordered_map<int64_t, std::vector<float>> inputs
            = {{Q, packedValues(10, H * D, 51)}, {K, cache(52)}, {V, cache(53)}};
        if(sink)
        {
            inputs[SINK] = {0.7F, -0.4F};
        }
        expectGpuMatchesCpu(raggedQPagedKvGraph(sink),
                            inputs,
                            {{O, static_cast<size_t>(10 * H * D)}},
                            1e-5F,
                            {{SEQ_Q, seqQ},
                             {SEQ_KV, seqKv},
                             {PAGE_TABLE_K, PAGE_TABLE},
                             {PAGE_TABLE_V, PAGE_TABLE}});
    }
}

// A seq_len or offset table whose values lie off the batch axis would be copied from the device
// past its buffer; the executor declines such a graph instead of reading it.
TEST(TestGpuRaggedSdpaDensify, AuxOffTheBatchAxisIsNotApplicable)
{
    SKIP_IF_NO_DEVICES();
    const auto reshape = [](const std::vector<uint8_t>& graph, int64_t uid, int64_t count) {
        return editGraph(graph, [&](GraphT& g) {
            tensorOf(g, uid).dims = {1, count, 1, 1};
            tensorOf(g, uid).strides = {count, 1, 1, 1};
        });
    };
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor gpu;
    auto applicable
        = [&](std::vector<uint8_t> graph) { return gpu.isApplicable(graph.data(), graph.size()); };
    const auto withLens = withSeqLens(raggedGraph(false, true));
    EXPECT_TRUE(applicable(withLens));
    EXPECT_FALSE(applicable(reshape(withLens, SEQ_Q, B)));
    EXPECT_FALSE(applicable(reshape(raggedGraph(false, true), Q_OFFSETS, B + 1)));
}
