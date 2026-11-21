// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// A ragged Q with seq_len_kv and no seq_len_q over non-paged K/V (SdpaSemanticsRegister.md
// SEM-028), run by the GPU reference graph executor as the CPU suite
// TestSdpaFwdPlanSeqLenKvOnly does: PyTorch's varlen seqused_k and vLLM's non-paged call. The
// query lengths come from Q's ragged offsets: bit-identical to seq_len_q = diff(offsets), and
// equal to the FlashAttention-terms oracle and to the CPU executor.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/SdpaVllmFormTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

#include "harness/gpu-graph-executor/GpuReferenceGraphExecutor.hpp"

using namespace hipdnn_test_sdk::utilities::sdpa_vllm_test;
using hipdnn_data_sdk::utilities::Tensor;

namespace
{

struct Outputs
{
    std::vector<float> o;
    std::vector<float> lse;
};

template <typename T>
std::unique_ptr<Tensor<T>> deviceCopy(const std::vector<T>& values)
{
    auto tensor = std::make_unique<Tensor<T>>(
        std::vector<int64_t>{static_cast<int64_t>(std::max<size_t>(values.size(), 1))});
    std::memcpy(tensor->memory().hostData(), values.data(), values.size() * sizeof(T));
    return tensor;
}

// One FLOAT run of the problem's frontend-built graph, on the GPU executor (device copies of
// the buffers) or the CPU one. Outputs start as NaN, so an element left unwritten shows up.
Outputs run(const VllmProblem& p, const VllmBuffers& buffers, bool gpu)
{
    auto graph = buildVllmGraph(p);
    auto q = deviceCopy(buffers.q);
    auto k = deviceCopy(buffers.kv);
    auto v = deviceCopy(buffers.v);
    auto o = deviceCopy(std::vector<float>(static_cast<size_t>(queryElements(p)),
                                           std::numeric_limits<float>::quiet_NaN()));
    auto lse = deviceCopy(std::vector<float>(static_cast<size_t>(statsElements(p)),
                                             std::numeric_limits<float>::quiet_NaN()));
    auto queryOffsets = deviceCopy(buffers.queryStartLoc);
    auto keyOffsets = deviceCopy(buffers.keyStartLoc);
    auto seqLenQ = deviceCopy(buffers.seqLenQ);
    auto seqLenKv = deviceCopy(buffers.seqLenKv);
    const auto data = [gpu](auto& t) -> void* {
        return gpu ? t->memory().deviceData() : t->memory().hostData();
    };
    std::unordered_map<int64_t, void*> pack = {{Q_UID, data(q)},
                                               {K_UID, data(k)},
                                               {V_UID, data(v)},
                                               {O_UID, data(o)},
                                               {STATS_UID, data(lse)},
                                               {Q_OFFSETS_UID, data(queryOffsets)},
                                               {SEQ_LEN_KV_UID, data(seqLenKv)}};
    if(p.raggedKv)
    {
        pack[KV_OFFSETS_UID] = data(keyOffsets);
    }
    if(p.explicitSeqLenQ)
    {
        pack[SEQ_LEN_Q_UID] = data(seqLenQ);
    }
    if(gpu)
    {
        hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
        o->memory().markDeviceModified();
        lse->memory().markDeviceModified();
    }
    else
    {
        hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor executor;
        EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
        executor.execute(graph.data(), graph.size(), pack);
    }
    Outputs out;
    out.o.assign(o->memory().hostData(), o->memory().hostData() + queryElements(p));
    out.lse.assign(lse->memory().hostData(), lse->memory().hostData() + statsElements(p));
    return out;
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expectNear(const std::vector<float>& actual, const std::vector<double>& expected)
{
    ASSERT_EQ(actual.size(), expected.size());
    for(size_t i = 0; i < expected.size(); ++i)
    {
        if(std::isinf(expected[i]))
        {
            ASSERT_EQ(actual[i], static_cast<float>(expected[i])) << "[" << i << "] keyless";
        }
        else
        {
            ASSERT_NEAR(actual[i], expected[i], 2e-5) << "[" << i << "]";
        }
    }
}

void expectSeqLenQIsDerived(const VllmProblem& p, float unseen)
{
    const auto buffers = makeBuffers(p, unseen);
    const auto derived = run(p, buffers, true);
    auto explicitLengths = p;
    explicitLengths.explicitSeqLenQ = true;
    const auto given = run(explicitLengths, buffers, true);
    EXPECT_TRUE(bitsEqual(derived.o, given.o));
    EXPECT_TRUE(bitsEqual(derived.lse, given.lse));
    const auto [o, lse] = faOracle(p, buffers);
    expectNear(derived.o, o);
    expectNear(derived.lse, lse);
    const auto cpu = run(p, buffers, false);
    const std::vector<double> cpuO(cpu.o.begin(), cpu.o.end());
    expectNear(derived.o, cpuO);
}

} // namespace

TEST(TestGpuSdpaFwdPlanSeqLenKvOnly, SequsedKNeedsNoSeqLenQ)
{
    SKIP_IF_NO_DEVICES();
    for(const auto window : {FaWindow{}, FaWindow{-1, 0}, FaWindow{2, 0}})
    {
        VllmProblem p;
        p.paged = false;
        p.raggedKv = true;
        p.batch = 4;
        p.qLens = {5, 0, 3, 4};
        p.kvLens = {13, 4, 9, 6};
        p.kvUsed = {9, 4, 0, 6};
        p.window = window;
        expectSeqLenQIsDerived(p, std::numeric_limits<float>::quiet_NaN());
    }
}

TEST(TestGpuSdpaFwdPlanSeqLenKvOnly, DenseKvUnderARaggedQueryNeedsNoSeqLenQ)
{
    SKIP_IF_NO_DEVICES();
    for(const auto window : {FaWindow{}, FaWindow{-1, 0}, FaWindow{3, 0}})
    {
        VllmProblem p;
        p.paged = false;
        p.window = window;
        expectSeqLenQIsDerived(p, std::numeric_limits<float>::infinity());
    }
}
