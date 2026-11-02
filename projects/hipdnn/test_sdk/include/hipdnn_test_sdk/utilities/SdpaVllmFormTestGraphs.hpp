// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

// SDPA forward graphs in the form vLLM calls attention (SdpaSemanticsRegister.md SEM-026 and
// SEM-027), built through the frontend Graph API and checked by Graph::validate, with their host
// buffers and an independent oracle written in FlashAttention's own terms. The CPU and GPU
// reference tests run the same problems through their graph executors.
//
// - Q and O: ragged, packed token by token with vLLM's query_start_loc as the ragged offset
//   (multiplier Hq * D), or dense [B, Hq, max_q, D] with per-batch seq_len_q.
// - K and V: paged containers [num_blocks, Hkv, block_size, D] read through page tables
//   [B, 1, M, 1] (vLLM's block_table, rows W >= M apart), contiguous or interleaved in one
//   allocation (vLLM's LBNHC cache: head stride 2D, V at +D); dense [B, Hkv, max_kv, D]; or
//   packed like Q through cu_seqlens_k (vLLM's encoder call, compact THD, no lengths), with
//   PyTorch's seqused_k as seq_len_kv when asked for (SEM-028).
// - seq_len_kv: vLLM's seq_lens. seq_len_q only when asked for (vLLM has none).
// - A window in FlashAttention's convention (window_size_left/right, -1 unbounded, anchored at
//   the bottom-right diagonal i + Skv - Sq), mapped to hipDNN's bounds by faWindowToBounds.
namespace hipdnn_test_sdk::utilities::sdpa_vllm_test
{

/// FlashAttention's window: keys [i + Skv - Sq - left, i + Skv - Sq + right] of query row i,
/// -1 unbounded on that side. vLLM stores sliding window W as (W - 1, 0) on decoder layers and
/// (W - 1, W - 1) on encoder layers; FA's non-causal (w, w).
struct FaWindow
{
    int64_t left = -1;
    int64_t right = -1;
};

/// hipDNN's band for an FA window (SEM-027): left_bound = left + 1 (cuDNN's left_bound counts
/// the diagonal), right_bound = right, BOTTOM_RIGHT; an unbounded side is unset.
struct HipdnnBounds
{
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound;
};

inline HipdnnBounds faWindowToBounds(const FaWindow& window)
{
    HipdnnBounds bounds;
    if(window.left >= 0)
    {
        bounds.leftBound = window.left + 1;
    }
    if(window.right >= 0)
    {
        bounds.rightBound = window.right;
    }
    return bounds;
}

/// One problem. Every length vector has `batch` entries.
struct VllmProblem
{
    int64_t batch = 3;
    int64_t qHeads = 4;
    int64_t kvHeads = 2;
    int64_t headDim = 8;
    float scale = 0.35F;
    std::vector<int32_t> qLens = {5, 0, 3};
    std::vector<int32_t> kvLens = {13, 4, 9};
    /// Ragged Q and O through query_start_loc; dense with seq_len_q otherwise.
    bool raggedQuery = true;
    /// Whether the graph carries seq_len_q (always when Q is dense).
    bool explicitSeqLenQ = false;
    /// The dense Q's (or ragged Q's declared) max query length; 0 means max(qLens).
    int64_t maxQ = 0;
    /// Paged K/V; dense [B, Hkv, maxKv, D] otherwise.
    bool paged = true;
    /// K and V packed token by token through cu_seqlens_k (kvLens), with Q ragged too and no
    /// seq_len tensors: compact THD (needs paged = false).
    bool raggedKv = false;
    /// With raggedKv: PyTorch's seqused_k, each sequence's used keys (at most its kvLens slot),
    /// as the graph's seq_len_kv; keys past it are no row's. Empty: compact THD.
    std::vector<int32_t> kvUsed;
    int64_t maxKv = 16;
    int64_t blockSize = 4;
    int64_t numBlocks = 12;
    /// Table width M and row stride W (W >= M; vLLM's block_table rows are max_num_blocks
    /// apart). Each batch's row, W entries; unused entries hold the null block 0.
    int64_t tableWidth = 4;
    int64_t tableRowStride = 4;
    std::vector<int32_t> blockTable = {3, 7, 1, 10, 5, 0, 0, 0, 9, 2, 11, 0};
    bool interleavedKv = false;
    std::optional<int32_t> maxSeqLenKv;
    FaWindow window;
    bool withStats = true;
    /// Q/K/V/O's dtype in the graph: FLOAT or BFLOAT16. The buffers are float either way; for
    /// BFLOAT16 every value is one bf16 holds exactly, and the tests store it as bf16.
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::FLOAT;
};

enum Uid : int64_t
{
    Q_UID = 1,
    K_UID,
    V_UID,
    O_UID,
    STATS_UID,
    Q_OFFSETS_UID,
    SEQ_LEN_Q_UID,
    SEQ_LEN_KV_UID,
    PAGE_TABLE_K_UID,
    PAGE_TABLE_V_UID,
    KV_OFFSETS_UID,
};

inline int64_t maxQueryLength(const VllmProblem& p)
{
    const auto longest = *std::max_element(p.qLens.begin(), p.qLens.end());
    return p.maxQ > 0 ? p.maxQ : std::max<int64_t>(longest, 1);
}

inline int64_t totalTokens(const VllmProblem& p)
{
    int64_t total = 0;
    for(const auto len : p.qLens)
    {
        total += len;
    }
    return total;
}

/// query_start_loc: [0, q0, q0 + q1, ...].
inline std::vector<int32_t> queryStartLoc(const VllmProblem& p)
{
    std::vector<int32_t> offsets = {0};
    for(const auto len : p.qLens)
    {
        offsets.push_back(offsets.back() + len);
    }
    return offsets;
}

/// cu_seqlens_k: [0, kv0, kv0 + kv1, ...].
inline std::vector<int32_t> keyStartLoc(const VllmProblem& p)
{
    std::vector<int32_t> offsets = {0};
    for(const auto len : p.kvLens)
    {
        offsets.push_back(offsets.back() + len);
    }
    return offsets;
}

/// The K (and V) container's strides: [Hkv * bs * D, bs * D, D, 1], or vLLM's interleaved LBNHC
/// cache [bs * Hkv * 2D, 2D, Hkv * 2D, 1] (V at +D in the same allocation).
inline std::vector<int64_t> kvStrides(const VllmProblem& p)
{
    if(p.raggedKv)
    {
        return {p.maxKv * p.kvHeads * p.headDim, p.headDim, p.kvHeads * p.headDim, 1};
    }
    if(!p.paged)
    {
        return {p.kvHeads * p.maxKv * p.headDim, p.maxKv * p.headDim, p.headDim, 1};
    }
    if(p.interleavedKv)
    {
        const auto twoD = 2 * p.headDim;
        return {p.blockSize * p.kvHeads * twoD, twoD, p.kvHeads * twoD, 1};
    }
    return {p.kvHeads * p.blockSize * p.headDim, p.blockSize * p.headDim, p.headDim, 1};
}

/// Elements of the K/V allocation (one allocation for both when interleaved).
inline int64_t kvElements(const VllmProblem& p)
{
    if(p.raggedKv)
    {
        return std::max<int64_t>(keyStartLoc(p).back(), 1) * p.kvHeads * p.headDim;
    }
    const auto count = p.paged ? p.numBlocks : p.batch;
    const auto perEntry = p.paged ? p.blockSize : p.maxKv;
    return count * p.kvHeads * perEntry * p.headDim * (p.interleavedKv ? 2 : 1);
}

/// V's first element within its allocation (D when interleaved with K).
inline int64_t valueBase(const VllmProblem& p)
{
    return p.interleavedKv ? p.headDim : 0;
}

/// Q (or O) elements: T * Hq * D packed, or B * max_q * Hq * D dense (BSHD).
inline int64_t queryElements(const VllmProblem& p)
{
    return (p.raggedQuery ? totalTokens(p) : p.batch * maxQueryLength(p)) * p.qHeads * p.headDim;
}

inline int64_t statsElements(const VllmProblem& p)
{
    return (p.raggedQuery ? totalTokens(p) : p.batch * maxQueryLength(p)) * p.qHeads;
}

/// Element of query row i of batch b, head h, column d in the Q (or O) buffer.
inline int64_t queryIndex(const VllmProblem& p, int64_t b, int64_t h, int64_t i, int64_t d)
{
    const auto token
        = p.raggedQuery ? queryStartLoc(p)[static_cast<size_t>(b)] + i : b * maxQueryLength(p) + i;
    return (token * p.qHeads + h) * p.headDim + d;
}

inline int64_t statsIndex(const VllmProblem& p, int64_t b, int64_t h, int64_t i)
{
    const auto token
        = p.raggedQuery ? queryStartLoc(p)[static_cast<size_t>(b)] + i : b * maxQueryLength(p) + i;
    return token * p.qHeads + h;
}

/// Element of logical key j of batch b, KV head h, column d in the K allocation (add
/// valueBase for V): through the page table when paged.
inline int64_t keyIndex(const VllmProblem& p, int64_t b, int64_t h, int64_t j, int64_t d)
{
    const auto strides = kvStrides(p);
    if(p.raggedKv)
    {
        return (keyStartLoc(p)[static_cast<size_t>(b)] + j) * strides[2] + h * strides[1] + d;
    }
    if(!p.paged)
    {
        return b * strides[0] + h * strides[1] + j * strides[2] + d;
    }
    const auto block = p.blockTable[static_cast<size_t>(b * p.tableRowStride + j / p.blockSize)];
    return block * strides[0] + h * strides[1] + (j % p.blockSize) * strides[2] + d;
}

/// Batch b's key length: its used keys (seqused_k) when given, else its kvLens.
inline int64_t usedKeys(const VllmProblem& p, int64_t b)
{
    const auto& lengths = p.kvUsed.empty() ? p.kvLens : p.kvUsed;
    return lengths[static_cast<size_t>(b)];
}

/// Whether query row i of batch b sees key j: j below seq_len_kv and inside FA's window around
/// the bottom-right diagonal i + kv_len - q_len.
inline bool faVisible(const VllmProblem& p, int64_t b, int64_t i, int64_t j)
{
    const auto qLen = static_cast<int64_t>(p.qLens[static_cast<size_t>(b)]);
    const auto kvLen = usedKeys(p, b);
    if(i >= qLen || j >= kvLen)
    {
        return false;
    }
    const auto diagonal = i + kvLen - qLen;
    return (p.window.left < 0 || j >= diagonal - p.window.left)
           && (p.window.right < 0 || j <= diagonal + p.window.right);
}

/// The host buffers of one problem.
struct VllmBuffers
{
    std::vector<float> q;
    std::vector<float> kv; // K's allocation (and V's when interleaved)
    std::vector<float> v; // V's allocation when not interleaved
    std::vector<int32_t> queryStartLoc;
    std::vector<int32_t> keyStartLoc;
    std::vector<int32_t> seqLenQ;
    std::vector<int32_t> seqLenKv;
    std::vector<int32_t> blockTable;
};

/// @p value as the problem's dtype stores it.
inline float storable(const VllmProblem& p, float value)
{
    return p.dataType == hipdnn_frontend::DataType::BFLOAT16
               ? static_cast<float>(hipdnn_data_sdk::types::bfloat16(value))
               : value;
}

/// A deterministic value in [-1, 1).
inline float hashValue(uint64_t i, uint64_t salt)
{
    uint64_t x = i * 0x9E3779B97F4A7C15ULL + salt * 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 31;
    x *= 0x94D049BB133111EBULL;
    x ^= x >> 29;
    return static_cast<float>(x % 2048) / 1024.0F - 1.0F;
}

/// Random Q, K and V. Every K/V element no live query row sees (unreferenced and null blocks,
/// unused slots, keys past seq_len_kv, keys outside every row's window, the interleaved
/// allocation's unused halves are all K or V) holds `unseen`.
inline VllmBuffers makeBuffers(const VllmProblem& p, float unseen = 0.0F)
{
    VllmBuffers buffers;
    buffers.q.resize(static_cast<size_t>(queryElements(p)));
    for(size_t i = 0; i < buffers.q.size(); ++i)
    {
        buffers.q[i] = storable(p, hashValue(i, 1));
    }
    buffers.kv.assign(static_cast<size_t>(kvElements(p)), unseen);
    if(!p.interleavedKv)
    {
        buffers.v.assign(buffers.kv.size(), unseen);
    }
    auto& vAlloc = p.interleavedKv ? buffers.kv : buffers.v;
    for(int64_t b = 0; b < p.batch; ++b)
    {
        for(int64_t j = 0; j < p.kvLens[static_cast<size_t>(b)]; ++j)
        {
            bool seen = false;
            for(int64_t i = 0; i < p.qLens[static_cast<size_t>(b)] && !seen; ++i)
            {
                seen = faVisible(p, b, i, j);
            }
            if(!seen)
            {
                continue;
            }
            for(int64_t h = 0; h < p.kvHeads; ++h)
            {
                for(int64_t d = 0; d < p.headDim; ++d)
                {
                    const auto at = keyIndex(p, b, h, j, d);
                    const auto salt = static_cast<uint64_t>((b * p.maxKv + j) * p.kvHeads + h);
                    buffers.kv[static_cast<size_t>(at)]
                        = storable(p, hashValue(static_cast<uint64_t>(d), 2 + 2 * salt));
                    vAlloc[static_cast<size_t>(at + valueBase(p))]
                        = storable(p, hashValue(static_cast<uint64_t>(d), 3 + 2 * salt));
                }
            }
        }
    }
    buffers.queryStartLoc = queryStartLoc(p);
    buffers.keyStartLoc = keyStartLoc(p);
    buffers.seqLenQ = p.qLens;
    buffers.seqLenKv = p.kvUsed.empty() ? p.kvLens : p.kvUsed;
    buffers.blockTable = p.blockTable;
    return buffers;
}

/// FlashAttention's forward in double, from the definition: per row, softmax over the keys it
/// sees of scale * q.k, times V. A row that sees no key (or a padded dense row) is O = 0,
/// LSE = -inf (SEM-010).
inline std::pair<std::vector<double>, std::vector<double>> faOracle(const VllmProblem& p,
                                                                    const VllmBuffers& buffers)
{
    std::vector<double> o(static_cast<size_t>(queryElements(p)), 0.0);
    std::vector<double> lse(static_cast<size_t>(statsElements(p)),
                            -std::numeric_limits<double>::infinity());
    const auto& vAlloc = p.interleavedKv ? buffers.kv : buffers.v;
    const auto group = p.qHeads / p.kvHeads;
    for(int64_t b = 0; b < p.batch; ++b)
    {
        const auto kvLen = static_cast<int64_t>(p.kvLens[static_cast<size_t>(b)]);
        for(int64_t h = 0; h < p.qHeads; ++h)
        {
            const auto hk = h / group;
            for(int64_t i = 0; i < p.qLens[static_cast<size_t>(b)]; ++i)
            {
                std::vector<double> scores;
                std::vector<int64_t> keys;
                for(int64_t j = 0; j < kvLen; ++j)
                {
                    if(!faVisible(p, b, i, j))
                    {
                        continue;
                    }
                    double dot = 0.0;
                    for(int64_t d = 0; d < p.headDim; ++d)
                    {
                        dot += static_cast<double>(
                                   buffers.q[static_cast<size_t>(queryIndex(p, b, h, i, d))])
                               * static_cast<double>(
                                   buffers.kv[static_cast<size_t>(keyIndex(p, b, hk, j, d))]);
                    }
                    scores.push_back(static_cast<double>(p.scale) * dot);
                    keys.push_back(j);
                }
                if(scores.empty())
                {
                    continue;
                }
                const double top = *std::max_element(scores.begin(), scores.end());
                double sum = 0.0;
                for(const auto s : scores)
                {
                    sum += std::exp(s - top);
                }
                lse[static_cast<size_t>(statsIndex(p, b, h, i))] = top + std::log(sum);
                for(int64_t d = 0; d < p.headDim; ++d)
                {
                    double acc = 0.0;
                    for(size_t n = 0; n < keys.size(); ++n)
                    {
                        acc += std::exp(scores[n] - top) / sum
                               * static_cast<double>(vAlloc[static_cast<size_t>(
                                   keyIndex(p, b, hk, keys[n], d) + valueBase(p))]);
                    }
                    o[static_cast<size_t>(queryIndex(p, b, h, i, d))] = acc;
                }
            }
        }
    }
    return {o, lse};
}

/// The problem's graph through the frontend Graph API, validated (the frontend's rules). Throws
/// if the frontend rejects it.
inline std::shared_ptr<hipdnn_frontend::graph::Graph> makeVllmGraph(const VllmProblem& p)
{
    using hipdnn_frontend::DataType;
    using hipdnn_frontend::graph::TensorAttributes;
    const auto make = [](int64_t uid,
                         const std::string& name,
                         DataType type,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& strides) {
        auto t = std::make_shared<TensorAttributes>();
        t->set_uid(uid).set_name(name).set_data_type(type).set_dim(dims).set_stride(strides);
        return t;
    };
    const auto maxQ = maxQueryLength(p);
    const auto qkStride = p.qHeads * p.headDim;
    const std::vector<int64_t> qDims = {p.batch, p.qHeads, maxQ, p.headDim};
    const std::vector<int64_t> qStrides = {maxQ * qkStride, p.headDim, qkStride, 1};
    const std::vector<int64_t> lenDims = {p.batch, 1, 1, 1};
    const std::vector<int64_t> lenStrides = {1, 1, 1, 1};

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("SdpaFwdVllmForm")
        .set_io_data_type(p.dataType)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);
    auto q = make(Q_UID, "Q", p.dataType, qDims, qStrides);
    std::shared_ptr<TensorAttributes> offsets;
    if(p.raggedQuery)
    {
        offsets = make(
            Q_OFFSETS_UID, "QueryStartLoc", DataType::INT32, {p.batch + 1, 1, 1, 1}, {1, 1, 1, 1});
        q->set_ragged_offset(offsets).set_ragged_offset_multiplier(qkStride);
    }
    const auto kvCount = p.paged ? p.numBlocks : p.batch;
    const auto kvSeq = p.paged ? p.blockSize : p.maxKv;
    const std::vector<int64_t> kvDims = {kvCount, p.kvHeads, kvSeq, p.headDim};
    auto k = make(K_UID, "K", p.dataType, kvDims, kvStrides(p));
    auto v = make(V_UID, "V", p.dataType, kvDims, kvStrides(p));
    if(p.raggedKv)
    {
        auto keyOffsets = make(
            KV_OFFSETS_UID, "KeyStartLoc", DataType::INT32, {p.batch + 1, 1, 1, 1}, {1, 1, 1, 1});
        k->set_ragged_offset(keyOffsets).set_ragged_offset_multiplier(p.kvHeads * p.headDim);
        v->set_ragged_offset(keyOffsets).set_ragged_offset_multiplier(p.kvHeads * p.headDim);
    }

    hipdnn_frontend::graph::SdpaAttributes attrs;
    attrs.set_name("SdpaFwdVllmForm")
        .set_attn_scale(p.scale)
        .set_generate_stats(p.withStats)
        .set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT);
    const auto bounds = faWindowToBounds(p.window);
    if(bounds.leftBound.has_value())
    {
        attrs.set_diagonal_band_left_bound(*bounds.leftBound);
    }
    if(bounds.rightBound.has_value())
    {
        attrs.set_diagonal_band_right_bound(*bounds.rightBound);
    }
    if(!p.raggedKv || !p.kvUsed.empty())
    {
        attrs.set_seq_len_kv(
            make(SEQ_LEN_KV_UID, "SeqLenKv", DataType::INT32, lenDims, lenStrides));
        if(p.explicitSeqLenQ || !p.raggedQuery)
        {
            attrs.set_seq_len_q(
                make(SEQ_LEN_Q_UID, "SeqLenQ", DataType::INT32, lenDims, lenStrides));
        }
        attrs.set_padding_mask(true);
    }
    if(p.paged)
    {
        const std::vector<int64_t> tableDims = {p.batch, 1, p.tableWidth, 1};
        const std::vector<int64_t> tableStrides = {p.tableRowStride, p.tableRowStride, 1, 1};
        attrs.set_paged_attention_k_table(
            make(PAGE_TABLE_K_UID, "BlockTableK", DataType::INT32, tableDims, tableStrides));
        attrs.set_paged_attention_v_table(
            make(PAGE_TABLE_V_UID, "BlockTableV", DataType::INT32, tableDims, tableStrides));
        if(p.maxSeqLenKv.has_value())
        {
            attrs.set_paged_attention_max_seq_len_kv(*p.maxSeqLenKv);
        }
    }

    auto [o, stats] = graph->sdpa(q, k, v, attrs);
    o->set_uid(O_UID)
        .set_name("O")
        .set_data_type(p.dataType)
        .set_dim(qDims)
        .set_stride(qStrides)
        .set_output(true);
    if(p.raggedQuery)
    {
        o->set_ragged_offset(offsets).set_ragged_offset_multiplier(qkStride);
    }
    if(p.withStats)
    {
        stats->set_uid(STATS_UID)
            .set_name("Stats")
            .set_data_type(DataType::FLOAT)
            .set_dim({p.batch, p.qHeads, maxQ, 1})
            .set_stride({maxQ * p.qHeads, 1, p.qHeads, 1})
            .set_output(true);
        if(p.raggedQuery)
        {
            stats->set_ragged_offset(offsets).set_ragged_offset_multiplier(p.qHeads);
        }
    }
    auto error = graph->validate();
    if(!error.is_good())
    {
        throw std::runtime_error("frontend rejects the vLLM-form graph: " + error.get_message());
    }
    return graph;
}

/// makeVllmGraph, serialized.
inline std::vector<uint8_t> buildVllmGraph(const VllmProblem& p)
{
    auto [binary, serializeError] = makeVllmGraph(p)->to_binary();
    if(!serializeError.is_good())
    {
        throw std::runtime_error("cannot serialize the vLLM-form graph: "
                                 + serializeError.get_message());
    }
    return binary;
}

} // namespace hipdnn_test_sdk::utilities::sdpa_vllm_test
