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
// buffers) and compares every output buffer element by element.
void expectGpuMatchesCpu(std::vector<uint8_t> graph,
                         const std::unordered_map<int64_t, std::vector<float>>& inputs,
                         const std::unordered_map<int64_t, size_t>& outputSizes,
                         float tolerance)
{
    std::vector<int32_t> qOffsets = Q_TOKENS;
    std::vector<int32_t> kvOffsets = KV_TOKENS;

    // CPU: host buffers.
    std::unordered_map<int64_t, std::vector<float>> cpuData = inputs;
    std::unordered_map<int64_t, void*> cpuPack
        = {{Q_OFFSETS, qOffsets.data()}, {KV_OFFSETS, kvOffsets.data()}};
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
    Tensor<int32_t> qOff({B + 1});
    Tensor<int32_t> kvOff({B + 1});
    std::memcpy(qOff.memory().hostData(), qOffsets.data(), qOffsets.size() * sizeof(int32_t));
    std::memcpy(kvOff.memory().hostData(), kvOffsets.data(), kvOffsets.size() * sizeof(int32_t));
    gpuPack[Q_OFFSETS] = qOff.memory().deviceData();
    gpuPack[KV_OFFSETS] = kvOff.memory().deviceData();

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
