// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// A ragged Q with seq_len_kv and no seq_len_q over non-paged K/V (SdpaSemanticsRegister.md
// SEM-028): PyTorch's varlen seqused_k (K/V packed through cu_seqlens_k, seqused_k their used
// keys) and vLLM's non-paged call (dense K/V with seq_lens). The graphs are built through the
// frontend (SdpaVllmFormTestGraphs.hpp) and run by the CPU reference graph executor, which takes
// the query lengths from Q's ragged offsets (the dense rewrite, RaggedSdpaDensify.hpp): the
// result is bit-identical to the same graph with seq_len_q = diff(offsets), and equal to an
// oracle written in FlashAttention's terms.

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_test_sdk/utilities/SdpaVllmFormTestGraphs.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/CpuReferenceGraphExecutor.hpp>

using namespace hipdnn_test_sdk::utilities::sdpa_vllm_test;
using hipdnn_test_sdk::utilities::CpuReferenceGraphExecutor;

namespace
{

struct Outputs
{
    std::vector<float> o;
    std::vector<float> lse;
};

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

// One CPU executor run of the problem's frontend-built graph, stored in the problem's dtype.
// Outputs start as NaN, so an element the reference leaves unwritten shows up.
template <typename T>
Outputs runCpuAs(const VllmProblem& p, VllmBuffers& buffers)
{
    auto graph = buildVllmGraph(p);
    auto q = storeAs<T>(buffers.q);
    auto k = storeAs<T>(buffers.kv);
    auto v = storeAs<T>(buffers.v);
    std::vector<T> o(static_cast<size_t>(queryElements(p)),
                     static_cast<T>(std::numeric_limits<float>::quiet_NaN()));
    std::vector<float> lse(static_cast<size_t>(statsElements(p)),
                           std::numeric_limits<float>::quiet_NaN());
    std::unordered_map<int64_t, void*> pack = {{Q_UID, q.data()},
                                               {K_UID, k.data()},
                                               {V_UID, v.data()},
                                               {O_UID, o.data()},
                                               {STATS_UID, lse.data()},
                                               {Q_OFFSETS_UID, buffers.queryStartLoc.data()},
                                               {SEQ_LEN_KV_UID, buffers.seqLenKv.data()}};
    if(p.raggedKv)
    {
        pack[KV_OFFSETS_UID] = buffers.keyStartLoc.data();
    }
    if(p.explicitSeqLenQ)
    {
        pack[SEQ_LEN_Q_UID] = buffers.seqLenQ.data();
    }
    CpuReferenceGraphExecutor executor;
    EXPECT_TRUE(executor.isApplicable(graph.data(), graph.size()));
    executor.execute(graph.data(), graph.size(), pack);
    Outputs out;
    for(const auto value : o)
    {
        out.o.push_back(static_cast<float>(value));
    }
    out.lse = lse;
    return out;
}

Outputs runCpu(const VllmProblem& p, VllmBuffers buffers)
{
    return p.dataType == hipdnn_frontend::DataType::BFLOAT16
               ? runCpuAs<hipdnn_data_sdk::types::bfloat16>(p, buffers)
               : runCpuAs<float>(p, buffers);
}

bool bitsEqual(const std::vector<float>& a, const std::vector<float>& b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void expectMatchesOracle(const VllmProblem& p, const VllmBuffers& buffers, const Outputs& out)
{
    const auto [o, lse] = faOracle(p, buffers);
    const double tolerance = p.dataType == hipdnn_frontend::DataType::BFLOAT16 ? 1e-2 : 2e-5;
    for(size_t i = 0; i < o.size(); ++i)
    {
        ASSERT_NEAR(out.o[i], o[i], tolerance) << "O[" << i << "]";
    }
    for(size_t i = 0; i < lse.size(); ++i)
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

// The derived form against the explicit one on the same buffers, and against the oracle.
void expectSeqLenQIsDerived(const VllmProblem& p, float unseen)
{
    const auto buffers = makeBuffers(p, unseen);
    const auto derived = runCpu(p, buffers);
    auto explicitLengths = p;
    explicitLengths.explicitSeqLenQ = true;
    const auto given = runCpu(explicitLengths, buffers);
    EXPECT_TRUE(bitsEqual(derived.o, given.o));
    EXPECT_TRUE(bitsEqual(derived.lse, given.lse));
    expectMatchesOracle(p, buffers, derived);
}

// PyTorch's varlen seqused_k: Q and K/V packed, each key slot's used keys seq_len_kv. A request
// with no query token, one with no used key, and a used length short of its slot.
VllmProblem sequsedK()
{
    VllmProblem p;
    p.paged = false;
    p.raggedKv = true;
    p.qLens = {5, 0, 3, 4};
    p.kvLens = {13, 4, 9, 6};
    p.kvUsed = {9, 4, 0, 6};
    p.batch = 4;
    return p;
}

// vLLM's non-paged call: Q packed, K/V dense [B, Hkv, max_kv, D] with seq_lens.
VllmProblem denseKv()
{
    VllmProblem p;
    p.paged = false;
    return p;
}

const char* windowName(const FaWindow& w)
{
    if(w.left < 0)
    {
        return w.right < 0 ? "no window" : "causal";
    }
    return w.right < 0 ? "left only" : "decoder window";
}

} // namespace

// Keys past seq_len_kv (each slot's unused tail) hold NaN: no row reads them.
TEST(TestSdpaFwdPlanSeqLenKvOnly, SequsedKNeedsNoSeqLenQ)
{
    for(const auto window : {FaWindow{}, FaWindow{-1, 0}, FaWindow{2, 0}})
    {
        SCOPED_TRACE(windowName(window));
        auto p = sequsedK();
        p.window = window;
        expectSeqLenQIsDerived(p, std::numeric_limits<float>::quiet_NaN());
    }
    auto bf16 = sequsedK();
    bf16.dataType = hipdnn_frontend::DataType::BFLOAT16;
    bf16.window.right = 0;
    expectSeqLenQIsDerived(bf16, 0.0F);
}

TEST(TestSdpaFwdPlanSeqLenKvOnly, DenseKvUnderARaggedQueryNeedsNoSeqLenQ)
{
    for(const auto window : {FaWindow{}, FaWindow{-1, 0}, FaWindow{3, 0}})
    {
        SCOPED_TRACE(windowName(window));
        auto p = denseKv();
        p.window = window;
        expectSeqLenQIsDerived(p, std::numeric_limits<float>::infinity());
    }
}

// Without the explicit lengths the graph differs from compact THD: the used keys, not the
// slots, are the keys (the comparison above is not vacuous).
TEST(TestSdpaFwdPlanSeqLenKvOnly, SeqLenKvShortensTheKeySlots)
{
    const auto p = sequsedK();
    const auto buffers = makeBuffers(p, 0.0F);
    auto compact = p;
    compact.kvUsed.clear();
    const auto withUsed = runCpu(p, buffers);
    const auto slots = runCpu(compact, buffers);
    EXPECT_FALSE(bitsEqual(withUsed.o, slots.o));
}
