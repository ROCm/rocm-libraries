// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Ragged SDPA graphs that no native ragged plan accepts (backward, or a forward with a bias) run
// through the ragged-as-dense rewrite in both executors; the GPU result must equal the CPU one.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/RaggedSdpaDensifyTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using namespace hipdnn_test_sdk::utilities::ragged_densify_test;

namespace
{

// Runs `graph` on the CPU executor (host buffers) and the GPU executor (device copies of the same
// buffers) and compares every output buffer element by element. The token offsets are always
// bound; intInputs adds INT32 inputs (seq_len tensors, page tables).
void expectGpuMatchesCpu(std::vector<uint8_t> graph,
                         const std::unordered_map<int64_t, std::vector<float>>& inputs,
                         const std::unordered_map<int64_t, size_t>& outputSizes,
                         float tolerance,
                         std::unordered_map<int64_t, std::vector<int32_t>> intInputs = {})
{
    intInputs.emplace(Q_OFFSETS, Q_TOKENS);
    intInputs.emplace(KV_OFFSETS, KV_TOKENS);

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

} // namespace

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
