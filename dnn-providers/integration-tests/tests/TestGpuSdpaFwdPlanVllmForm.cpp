// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// vLLM's attention calls as hipDNN graphs (SdpaVllmFormTestGraphs.hpp), built through the
// frontend and run by the GPU reference graph executor, as the CPU suite
// TestSdpaFwdPlanVllmForm does: paged K/V under a ragged Q with seq_len_kv only (SEM-026), FA
// windows through their mapping (SEM-027), and keys no row sees holding NaN or infinity.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/SdpaVllmFormTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

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

template <typename T>
std::vector<T> storeAs(const std::vector<float>& values)
{
    std::vector<T> stored;
    stored.reserve(values.size());
    for(const auto value : values)
    {
        stored.push_back(static_cast<T>(value));
    }
    return stored;
}

// One GPU executor run of the problem's frontend-built graph on device copies of the buffers,
// stored in the problem's dtype. Outputs start as NaN, so an element the reference leaves
// unwritten shows up.
template <typename T>
Outputs runGpuAs(const VllmProblem& p, const VllmBuffers& buffers)
{
    auto graph = buildVllmGraph(p);
    auto q = deviceCopy(storeAs<T>(buffers.q));
    auto kv = deviceCopy(storeAs<T>(buffers.kv));
    auto v = deviceCopy(storeAs<T>(buffers.v));
    auto o = deviceCopy(std::vector<T>(static_cast<size_t>(queryElements(p)),
                                       static_cast<T>(std::numeric_limits<float>::quiet_NaN())));
    auto lse = deviceCopy(std::vector<float>(static_cast<size_t>(statsElements(p)),
                                             std::numeric_limits<float>::quiet_NaN()));
    auto queryOffsets = deviceCopy(buffers.queryStartLoc);
    auto keyOffsets = deviceCopy(buffers.keyStartLoc);
    auto seqLenQ = deviceCopy(buffers.seqLenQ);
    auto seqLenKv = deviceCopy(buffers.seqLenKv);
    auto table = deviceCopy(buffers.blockTable);

    std::unordered_map<int64_t, void*> pack
        = {{Q_UID, q->memory().deviceData()},
           {K_UID, kv->memory().deviceData()},
           {V_UID,
            p.interleavedKv
                ? static_cast<void*>(static_cast<T*>(kv->memory().deviceData()) + valueBase(p))
                : v->memory().deviceData()},
           {O_UID, o->memory().deviceData()}};
    if(p.withStats)
    {
        pack[STATS_UID] = lse->memory().deviceData();
    }
    if(p.raggedQuery)
    {
        pack[Q_OFFSETS_UID] = queryOffsets->memory().deviceData();
    }
    if(p.raggedKv)
    {
        pack[KV_OFFSETS_UID] = keyOffsets->memory().deviceData();
    }
    else
    {
        pack[SEQ_LEN_KV_UID] = seqLenKv->memory().deviceData();
        if(p.explicitSeqLenQ || !p.raggedQuery)
        {
            pack[SEQ_LEN_Q_UID] = seqLenQ->memory().deviceData();
        }
    }
    if(p.paged)
    {
        pack[PAGE_TABLE_K_UID] = table->memory().deviceData();
        pack[PAGE_TABLE_V_UID] = table->memory().deviceData();
    }
    hipdnn_integration_tests::gpu_graph_executor::GpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
    executor.execute(graph.data(), graph.size(), pack);
    o->memory().markDeviceModified();
    lse->memory().markDeviceModified();
    Outputs out;
    const auto* oData = o->memory().hostData();
    for(int64_t i = 0; i < queryElements(p); ++i)
    {
        out.o.push_back(static_cast<float>(oData[i]));
    }
    out.lse.assign(lse->memory().hostData(), lse->memory().hostData() + statsElements(p));
    return out;
}

Outputs runGpu(const VllmProblem& p, const VllmBuffers& buffers)
{
    return p.dataType == hipdnn_frontend::DataType::BFLOAT16
               ? runGpuAs<hipdnn_data_sdk::types::bfloat16>(p, buffers)
               : runGpuAs<float>(p, buffers);
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expectMatchesOracle(const VllmProblem& p, const VllmBuffers& buffers, const Outputs& out)
{
    const auto [o, lse] = faOracle(p, buffers);
    // bf16 O is rounded to 8 bits; the LSE is FLOAT either way.
    const double tolerance = p.dataType == hipdnn_frontend::DataType::BFLOAT16 ? 1e-2 : 2e-5;
    for(size_t i = 0; i < o.size(); ++i)
    {
        ASSERT_NEAR(out.o[i], o[i], tolerance) << "O[" << i << "]";
    }
    for(size_t i = 0; p.withStats && i < lse.size(); ++i)
    {
        if(std::isinf(lse[i]))
        {
            ASSERT_EQ(out.lse[i], static_cast<float>(lse[i])) << "LSE[" << i << "] keyless";
        }
        else
        {
            ASSERT_NEAR(out.lse[i], lse[i], tolerance) << "LSE[" << i << "]";
        }
    }
}

} // namespace

// Ragged Q over paged K/V with seq_len_kv only equals the same graph with an explicit
// seq_len_q bit for bit, and the oracle.
TEST(TestGpuSdpaFwdPlanVllmForm, PagedKvUnderARaggedQueryNeedsNoSeqLenQ)
{
    SKIP_IF_NO_DEVICES();
    for(const bool causal : {false, true})
    {
        SCOPED_TRACE(causal ? "causal" : "non-causal");
        VllmProblem p;
        p.window.right = causal ? 0 : -1;
        const auto buffers = makeBuffers(p);
        const auto derived = runGpu(p, buffers);
        auto explicitLengths = p;
        explicitLengths.explicitSeqLenQ = true;
        const auto given = runGpu(explicitLengths, buffers);
        EXPECT_TRUE(bitsEqual(derived.o, given.o));
        EXPECT_TRUE(bitsEqual(derived.lse, given.lse));
        expectMatchesOracle(p, buffers, derived);
    }
}

// FA / vLLM windows mapped to hipDNN's bounds (SEM-027) equal FA's definition.
TEST(TestGpuSdpaFwdPlanVllmForm, FaWindowsMapToHipdnnBounds)
{
    SKIP_IF_NO_DEVICES();
    const std::vector<std::pair<const char*, FaWindow>> windows = {{"decoder W=1", {0, 0}},
                                                                   {"decoder W=3", {2, 0}},
                                                                   {"encoder W=4", {3, 3}},
                                                                   {"non-causal w=2", {2, 2}},
                                                                   {"left only w=3", {3, -1}}};
    for(const auto& [name, window] : windows)
    {
        for(const bool paged : {true, false})
        {
            for(const bool ragged : {true, false})
            {
                SCOPED_TRACE(std::string(name) + (paged ? ", paged" : ", dense")
                             + (ragged ? ", ragged Q" : ", padded Q"));
                VllmProblem p;
                p.window = window;
                p.paged = paged;
                p.raggedQuery = ragged;
                const auto buffers = makeBuffers(p);
                expectMatchesOracle(p, buffers, runGpu(p, buffers));
            }
        }
    }
}

// NaN or infinity in every K/V element no row sees gives the same bits as zeros there;
// so do a block table with rows wider apart than its width and an interleaved cache.
TEST(TestGpuSdpaFwdPlanVllmForm, KeysNoRowSeesDoNotLeak)
{
    SKIP_IF_NO_DEVICES();
    const std::array<float, 2> poisons
        = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()};
    for(const bool paged : {true, false})
    {
        for(const auto& window : {FaWindow{-1, 0}, FaWindow{2, 0}, FaWindow{1, 1}})
        {
            SCOPED_TRACE(std::string(paged ? "paged" : "dense") + ", window ("
                         + std::to_string(window.left) + ", " + std::to_string(window.right) + ")");
            VllmProblem p;
            p.paged = paged;
            p.window = window;
            const auto clean = runGpu(p, makeBuffers(p, 0.0F));
            for(const float poison : poisons)
            {
                const auto buffers = makeBuffers(p, poison);
                const auto poisoned = runGpu(p, buffers);
                EXPECT_TRUE(bitsEqual(poisoned.o, clean.o)) << "unseen = " << poison;
                EXPECT_TRUE(bitsEqual(poisoned.lse, clean.lse)) << "unseen = " << poison;
                expectMatchesOracle(p, buffers, poisoned);
            }
        }
    }

    const float nan = std::numeric_limits<float>::quiet_NaN();
    VllmProblem compact;
    compact.window = {3, 0};
    const auto reference = runGpu(compact, makeBuffers(compact, 0.0F));
    VllmProblem strided = compact;
    strided.tableRowStride = 6;
    strided.blockTable = {3, 7, 1, 10, 4, 6, 5, 0, 0, 0, 8, 4, 9, 2, 11, 0, 6, 8};
    for(const bool interleaved : {false, true})
    {
        for(auto variant : {compact, strided})
        {
            variant.interleavedKv = interleaved;
            SCOPED_TRACE(std::string("row stride ") + std::to_string(variant.tableRowStride)
                         + (interleaved ? ", interleaved" : ""));
            const auto buffers = makeBuffers(variant, nan);
            const auto out = runGpu(variant, buffers);
            EXPECT_TRUE(bitsEqual(out.o, reference.o));
            EXPECT_TRUE(bitsEqual(out.lse, reference.lse));
            expectMatchesOracle(variant, buffers, out);
        }
    }
}

// Q, K and V all packed (vLLM's encoder call, AITER's extend): FLOAT through the dense rewrite,
// BFLOAT16 through the native ragged kernel; each window equals the oracle and NaN outside every
// row's window gives the same bits as zeros.
TEST(TestGpuSdpaFwdPlanVllmForm, CompactThdWindowsAndKeysNoRowSees)
{
    SKIP_IF_NO_DEVICES();
    for(const auto dataType :
        {hipdnn_frontend::DataType::FLOAT, hipdnn_frontend::DataType::BFLOAT16})
    {
        for(const auto& window : {FaWindow{-1, 0}, FaWindow{2, 0}, FaWindow{3, 3}})
        {
            SCOPED_TRACE(std::string(dataType == hipdnn_frontend::DataType::FLOAT ? "f32" : "bf16")
                         + ", window (" + std::to_string(window.left) + ", "
                         + std::to_string(window.right) + ")");
            VllmProblem p;
            p.dataType = dataType;
            p.paged = false;
            p.raggedKv = true;
            p.qLens = {5, 2, 3};
            p.window = window;
            const auto clean = runGpu(p, makeBuffers(p, 0.0F));
            const auto buffers = makeBuffers(p, std::numeric_limits<float>::quiet_NaN());
            const auto poisoned = runGpu(p, buffers);
            EXPECT_TRUE(bitsEqual(poisoned.o, clean.o));
            EXPECT_TRUE(bitsEqual(poisoned.lse, clean.lse));
            expectMatchesOracle(p, buffers, poisoned);
        }
    }
}
