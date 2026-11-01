// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// vLLM's attention calls as hipDNN graphs (SdpaVllmFormTestGraphs.hpp), built through the
// frontend and run by the CPU reference graph executor:
// - paged K/V under a ragged Q with seq_len_kv only (SEM-026);
// - FlashAttention / vLLM windows through their mapping to hipDNN's bounds (SEM-027), against an
//   oracle written in FA's own terms;
// - keys no row sees (null and unreferenced blocks, unused slots, keys past seq_len_kv, keys
//   outside the window) holding NaN or infinity change nothing, nor do a block table whose rows
//   are wider apart than its width and an interleaved K/V cache.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
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

// One CPU executor run of the problem's frontend-built graph on these buffers, stored in the
// problem's dtype. Outputs start as NaN, so an element the reference leaves unwritten shows up.
template <typename T>
Outputs runCpuAs(const VllmProblem& p, VllmBuffers& buffers)
{
    auto graph = buildVllmGraph(p);
    auto q = storeAs<T>(buffers.q);
    auto kv = storeAs<T>(buffers.kv);
    auto v = storeAs<T>(buffers.v);
    std::vector<T> o(static_cast<size_t>(queryElements(p)),
                     static_cast<T>(std::numeric_limits<float>::quiet_NaN()));
    std::vector<float> lse(static_cast<size_t>(statsElements(p)),
                           std::numeric_limits<float>::quiet_NaN());
    std::unordered_map<int64_t, void*> pack
        = {{Q_UID, q.data()},
           {K_UID, kv.data()},
           {V_UID,
            p.interleavedKv ? static_cast<void*>(kv.data() + valueBase(p))
                            : static_cast<void*>(v.data())},
           {O_UID, o.data()}};
    if(p.withStats)
    {
        pack[STATS_UID] = lse.data();
    }
    if(p.raggedQuery)
    {
        pack[Q_OFFSETS_UID] = buffers.queryStartLoc.data();
    }
    if(p.raggedKv)
    {
        pack[KV_OFFSETS_UID] = buffers.keyStartLoc.data();
    }
    else
    {
        pack[SEQ_LEN_KV_UID] = buffers.seqLenKv.data();
        if(p.explicitSeqLenQ || !p.raggedQuery)
        {
            pack[SEQ_LEN_Q_UID] = buffers.seqLenQ.data();
        }
    }
    if(p.paged)
    {
        pack[PAGE_TABLE_K_UID] = buffers.blockTable.data();
        pack[PAGE_TABLE_V_UID] = buffers.blockTable.data();
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

// vLLM's default problem with a wider block table: rows W = 6 apart, M = 4 used; the two
// trailing entries of each row name blocks no key uses (they must not be read).
VllmProblem stridedTable(VllmProblem p)
{
    p.tableRowStride = 6;
    p.blockTable = {3, 7, 1, 10, 4, 6, 5, 0, 0, 0, 8, 4, 9, 2, 11, 0, 6, 8};
    return p;
}

} // namespace

// Ragged Q (query_start_loc) over paged K/V with seq_len_kv only. The references take the
// query lengths from Q's offsets: the result is bit-identical to the same graph with an explicit
// seq_len_q = diff(query_start_loc), and equal to the oracle. A request with q_len 0, shuffled
// blocks and the null block 0 in unused entries are included.
TEST(TestSdpaFwdPlanVllmForm, PagedKvUnderARaggedQueryNeedsNoSeqLenQ)
{
    for(const bool causal : {false, true})
    {
        for(const bool maxSeqLenKv : {false, true})
        {
            SCOPED_TRACE(std::string(causal ? "causal" : "non-causal")
                         + (maxSeqLenKv ? ", max_seq_len_kv 16" : ""));
            VllmProblem p;
            p.window.right = causal ? 0 : -1;
            if(maxSeqLenKv)
            {
                p.maxSeqLenKv = 16;
            }
            const auto buffers = makeBuffers(p);
            const auto derived = runCpu(p, buffers);
            auto explicitLengths = p;
            explicitLengths.explicitSeqLenQ = true;
            const auto given = runCpu(explicitLengths, buffers);
            EXPECT_TRUE(bitsEqual(derived.o, given.o));
            EXPECT_TRUE(bitsEqual(derived.lse, given.lse));
            expectMatchesOracle(p, buffers, derived);
        }
    }
}

// Decode (q_len 1 per request) and a declared max_q longer than every request.
TEST(TestSdpaFwdPlanVllmForm, PagedKvUnderARaggedQueryDecodeAndLongerMaxQ)
{
    VllmProblem decode;
    decode.qLens = {1, 1, 1};
    decode.window.right = 0;
    const auto decodeBuffers = makeBuffers(decode);
    expectMatchesOracle(decode, decodeBuffers, runCpu(decode, decodeBuffers));

    VllmProblem longer;
    longer.maxQ = 8;
    longer.window.right = 0;
    const auto longerBuffers = makeBuffers(longer);
    expectMatchesOracle(longer, longerBuffers, runCpu(longer, longerBuffers));
}

// FA / vLLM windows mapped to hipDNN's bounds (SEM-027) equal FA's definition, dense and
// paged, ragged and padded: decoder sliding window W stored as (W - 1, 0), encoder (W - 1, W - 1),
// FA's non-causal (w, w), causal (-1, 0), a left-only window and none.
TEST(TestSdpaFwdPlanVllmForm, FaWindowsMapToHipdnnBounds)
{
    const std::vector<std::pair<const char*, FaWindow>> windows = {{"decoder W=1", {0, 0}},
                                                                   {"decoder W=3", {2, 0}},
                                                                   {"decoder W=6", {5, 0}},
                                                                   {"encoder W=4", {3, 3}},
                                                                   {"non-causal w=2", {2, 2}},
                                                                   {"non-causal w=4", {4, 4}},
                                                                   {"causal", {-1, 0}},
                                                                   {"left only w=3", {3, -1}},
                                                                   {"none", {-1, -1}}};
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
                expectMatchesOracle(p, buffers, runCpu(p, buffers));
            }
        }
    }
    // The table itself: (W - 1, 0) -> left W, right 0; (w, w) -> left w + 1, right w.
    EXPECT_EQ(faWindowToBounds({4, 0}).leftBound, 5);
    EXPECT_EQ(faWindowToBounds({4, 0}).rightBound, 0);
    EXPECT_EQ(faWindowToBounds({3, 3}).leftBound, 4);
    EXPECT_EQ(faWindowToBounds({3, 3}).rightBound, 3);
    EXPECT_FALSE(faWindowToBounds({-1, -1}).leftBound.has_value());
    EXPECT_FALSE(faWindowToBounds({-1, -1}).rightBound.has_value());
}

// NaN, +inf, -inf or a huge finite value in every K/V element no row sees -- the null
// block 0, blocks no table entry names, slots past seq_len_kv, keys outside the window, dense
// keys past seq_len_kv -- gives the same bits as zeros there.
TEST(TestSdpaFwdPlanVllmForm, KeysNoRowSeesDoNotLeak)
{
    const std::array<float, 4> poisons = {std::numeric_limits<float>::quiet_NaN(),
                                          std::numeric_limits<float>::infinity(),
                                          -std::numeric_limits<float>::infinity(),
                                          3.0e38F};
    for(const bool paged : {true, false})
    {
        for(const auto& window : {FaWindow{-1, 0}, FaWindow{2, 0}, FaWindow{1, 1}})
        {
            SCOPED_TRACE(std::string(paged ? "paged" : "dense") + ", window ("
                         + std::to_string(window.left) + ", " + std::to_string(window.right) + ")");
            VllmProblem p;
            p.paged = paged;
            p.window = window;
            const auto clean = runCpu(p, makeBuffers(p, 0.0F));
            for(const float poison : poisons)
            {
                SCOPED_TRACE("unseen = " + std::to_string(poison));
                const auto buffers = makeBuffers(p, poison);
                const auto poisoned = runCpu(p, buffers);
                EXPECT_TRUE(bitsEqual(poisoned.o, clean.o));
                EXPECT_TRUE(bitsEqual(poisoned.lse, clean.lse));
                expectMatchesOracle(p, buffers, poisoned);
            }
        }
    }
}

// A block table viewed with rows W = 6 apart and M = 4 entries used (vLLM's block_table
// sliced to the bucket's width), and vLLM's interleaved cache (one allocation, head stride 2D,
// V at +D): both equal the compact form bit for bit, with NaN in everything no row sees.
TEST(TestSdpaFwdPlanVllmForm, StridedBlockTableAndInterleavedCache)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();
    VllmProblem compact;
    compact.window = {3, 0};
    const auto reference = runCpu(compact, makeBuffers(compact, nan));

    const auto strided = stridedTable(compact);
    const auto stridedBuffers = makeBuffers(strided, nan);
    const auto stridedOut = runCpu(strided, stridedBuffers);
    EXPECT_TRUE(bitsEqual(stridedOut.o, reference.o));
    EXPECT_TRUE(bitsEqual(stridedOut.lse, reference.lse));
    expectMatchesOracle(strided, stridedBuffers, stridedOut);

    for(const auto& base : {compact, strided})
    {
        auto interleaved = base;
        interleaved.interleavedKv = true;
        const auto buffers = makeBuffers(interleaved, nan);
        const auto out = runCpu(interleaved, buffers);
        EXPECT_TRUE(bitsEqual(out.o, reference.o));
        EXPECT_TRUE(bitsEqual(out.lse, reference.lse));
        expectMatchesOracle(interleaved, buffers, out);
    }
}

// Windows and keys no row sees on vLLM's encoder call and AITER's extend: Q, K and V all packed (cu_seqlens_q and
// cu_seqlens_k, compact THD, no lengths), FLOAT through the dense rewrite and BFLOAT16 through the
// native ragged plan. Each FA window equals the oracle, and NaN in every key outside every row's
// window gives the same bits as zeros.
TEST(TestSdpaFwdPlanVllmForm, CompactThdWindowsAndKeysNoRowSees)
{
    for(const auto dataType :
        {hipdnn_frontend::DataType::FLOAT, hipdnn_frontend::DataType::BFLOAT16})
    {
        for(const auto& window :
            {FaWindow{-1, -1}, FaWindow{-1, 0}, FaWindow{2, 0}, FaWindow{3, 3}})
        {
            for(const bool equalLengths : {true, false})
            {
                SCOPED_TRACE(
                    std::string(dataType == hipdnn_frontend::DataType::FLOAT ? "f32" : "bf16")
                    + ", window (" + std::to_string(window.left) + ", "
                    + std::to_string(window.right) + ")"
                    + (equalLengths ? ", encoder" : ", extend"));
                VllmProblem p;
                p.dataType = dataType;
                p.paged = false;
                p.raggedKv = true;
                p.qLens
                    = equalLengths ? std::vector<int32_t>{7, 0, 4} : std::vector<int32_t>{5, 2, 3};
                p.kvLens
                    = equalLengths ? std::vector<int32_t>{7, 0, 4} : std::vector<int32_t>{13, 4, 9};
                p.window = window;
                const auto clean = runCpu(p, makeBuffers(p, 0.0F));
                const auto buffers = makeBuffers(p, std::numeric_limits<float>::quiet_NaN());
                const auto poisoned = runCpu(p, buffers);
                EXPECT_TRUE(bitsEqual(poisoned.o, clean.o));
                EXPECT_TRUE(bitsEqual(poisoned.lse, clean.lse));
                expectMatchesOracle(p, buffers, poisoned);
            }
        }
    }
}
