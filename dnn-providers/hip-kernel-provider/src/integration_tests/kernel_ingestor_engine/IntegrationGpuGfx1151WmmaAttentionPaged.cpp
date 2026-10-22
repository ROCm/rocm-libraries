// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

/**
 * @file IntegrationGpuGfx1151WmmaAttentionPaged.cpp
 * @brief hipkernel:Gfx1151WmmaAttention on the cuDNN paged-K/V SDPA graph, end to end
 *        through the frontend on an RDNA (gfx11 / gfx12) device: vLLM's uniform decode
 *        (dense Q) and mixed batches (ragged Q) over a shuffled, NaN-poisoned paged cache
 *        -- the interleaved TRITON_ATTN K|V layout or separate planes -- against a
 *        gather-then-dense float64 reference, and one plan captured into a HIP graph and
 *        replayed with different sequence lengths, block ids and data.
 *
 * Unused cache slots and padding are NaN, so a read of anything outside the used
 * page-table entries or past seq_len_kv shows up as a non-finite output. O starts NaN:
 * every live row must be written (zeros for a row with no visible key) and every row past
 * seq_len_q -- including the gaps between ragged sequences -- must be left untouched.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{
namespace
{

constexpr const char* ENGINE_NAME = "hipkernel:Gfx1151WmmaAttention";

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t LSE_UID = 5;
constexpr int64_t PAGE_TABLE_UID = 6;
constexpr int64_t SEQ_LEN_KV_UID = 7;
constexpr int64_t SEQ_LEN_Q_UID = 8;
constexpr int64_t Q_OFFSET_UID = 9;

constexpr uint16_t NAN_BITS = 0xFFFF; // a NaN in both FP16 and BF16
constexpr int32_t UNUSED_BLOCK = std::numeric_limits<int32_t>::max();

enum class QueryForm
{
    DENSE, // uniform decode: [B, Hq, q_len, D], every row live
    DENSE_SEQ_LEN_Q, // dense Q whose live rows per sequence are seq_len_q
    RAGGED // ragged Q / O through token-unit offsets, live rows seq_len_q
};

struct PagedCase
{
    const char* name;
    DataType dataType;
    QueryForm form;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t headSize;
    int64_t pageSize;
    std::vector<int32_t> seqLenKv;
    // Live query rows per sequence; DENSE needs them all equal.
    std::vector<int32_t> seqLenQ;
    // Untouched rows after each sequence's live rows (DENSE_SEQ_LEN_Q: up to Sq; RAGGED:
    // before the next sequence).
    std::vector<int32_t> gaps;
    int64_t leftBound = -1;
    bool stats = false;
    bool interleavedCache = true;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const PagedCase& paged, std::ostream* os)
{
    *os << paged.name;
}

/// Host-side bits of one batch: the build-time geometry and the device inputs.
struct Geometry
{
    int64_t batch = 0;
    int64_t queryRows = 0; // the graph's Sq: the longest span
    int64_t totalRows = 0; // rows of the Q / O buffer
    int64_t maxBlocksPerSeq = 0;
    int64_t numBlocks = 0;
};

Geometry geometryOf(const PagedCase& paged)
{
    Geometry g;
    g.batch = static_cast<int64_t>(paged.seqLenKv.size());
    int64_t blocks = 0;
    int64_t longest = 0;
    for(size_t b = 0; b < paged.seqLenKv.size(); ++b)
    {
        const int64_t span = int64_t{paged.seqLenQ[b]} + paged.gaps[b];
        g.queryRows = std::max(g.queryRows, span);
        g.totalRows += span;
        longest = std::max<int64_t>(longest, paged.seqLenKv[b]);
        blocks += (paged.seqLenKv[b] + paged.pageSize - 1) / paged.pageSize;
    }
    if(paged.form != QueryForm::RAGGED)
    {
        g.totalRows = g.batch * g.queryRows;
    }
    // One spare table entry per row and a few never-referenced blocks.
    g.maxBlocksPerSeq = (longest + paged.pageSize - 1) / paged.pageSize + 1;
    g.numBlocks = blocks + 5;
    return g;
}

uint16_t toBits(float value, DataType dataType)
{
    if(dataType == DataType::HALF)
    {
        const auto half = static_cast<_Float16>(value);
        uint16_t bits = 0;
        std::memcpy(&bits, &half, sizeof(bits));
        return bits;
    }
    uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    word += 0x7FFFU + ((word >> 16U) & 1U); // round to nearest even
    return static_cast<uint16_t>(word >> 16U);
}

double fromBits(uint16_t bits, DataType dataType)
{
    if(dataType == DataType::HALF)
    {
        _Float16 half = 0;
        std::memcpy(&half, &bits, sizeof(bits));
        return static_cast<double>(half);
    }
    const uint32_t word = uint32_t{bits} << 16U;
    float value = 0.0F;
    std::memcpy(&value, &word, sizeof(value));
    return static_cast<double>(value);
}

/// One batch's device-input contents for a given set of lengths.
struct Batch
{
    std::vector<int32_t> seqLenKv;
    std::vector<int32_t> seqLenQ;
    std::vector<int32_t> gaps;
    std::vector<int32_t> qOffsets; // token units, ragged only
    std::vector<uint16_t> q; // [totalRows, Hq, D]
    std::vector<uint16_t> k; // [numBlocks, Hkv, page, row]
    std::vector<uint16_t> v; // V's own plane; empty for the interleaved cache
    std::vector<int32_t> pageTable; // [B, M]
};

class PagedProblem
{
public:
    PagedProblem(const PagedCase& paged, const Geometry& geometry)
        : _case(paged)
        , _g(geometry)
    {
    }

    int64_t cacheRow() const
    {
        return _case.interleavedCache ? 2 * _case.headSize : _case.headSize;
    }

    int64_t tokenStride() const
    {
        return _case.queryHeads * _case.headSize;
    }

    /// Row r of sequence b: its offset in the Q / O buffer, in rows.
    int64_t rowOf(const Batch& batch, int64_t b, int64_t r) const
    {
        return _case.form == QueryForm::RAGGED ? batch.qOffsets[static_cast<size_t>(b)] + r
                                               : b * _g.queryRows + r;
    }

    Batch makeBatch(std::vector<int32_t> seqLenKv,
                    std::vector<int32_t> seqLenQ,
                    std::vector<int32_t> gaps,
                    unsigned int seed) const
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> uniform(-1.0F, 1.0F);
        Batch batch;
        batch.seqLenKv = std::move(seqLenKv);
        batch.seqLenQ = std::move(seqLenQ);
        batch.gaps = std::move(gaps);
        const auto dt = _case.dataType;
        const auto d = static_cast<size_t>(_case.headSize);

        batch.q.assign(static_cast<size_t>(_g.totalRows * tokenStride()), NAN_BITS);
        batch.qOffsets.assign(static_cast<size_t>(_g.batch + 1), 0);
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
            batch.qOffsets[bi + 1] = batch.qOffsets[bi] + batch.seqLenQ[bi] + batch.gaps[bi];
            for(int64_t r = 0; r < batch.seqLenQ[bi]; ++r)
            {
                const auto base = static_cast<size_t>(rowOf(batch, b, r) * tokenStride());
                for(size_t i = 0; i < static_cast<size_t>(tokenStride()); ++i)
                {
                    batch.q[base + i] = toBits(uniform(rng), dt);
                }
            }
        }

        // Shuffle the physical blocks; NaN-poison every slot no sequence owns.
        const auto cacheElems
            = static_cast<size_t>(_g.numBlocks * _case.kvHeads * _case.pageSize * cacheRow());
        batch.k.assign(cacheElems, NAN_BITS);
        if(!_case.interleavedCache)
        {
            batch.v.assign(cacheElems, NAN_BITS);
        }
        std::vector<int32_t> order(static_cast<size_t>(_g.numBlocks));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        batch.pageTable.assign(static_cast<size_t>(_g.batch * _g.maxBlocksPerSeq), UNUSED_BLOCK);
        size_t next = 0;
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
            const int64_t keys = batch.seqLenKv[bi];
            const int64_t pages = (keys + _case.pageSize - 1) / _case.pageSize;
            for(int64_t page = 0; page < pages; ++page)
            {
                const int32_t block = order[next++];
                batch.pageTable[static_cast<size_t>(b * _g.maxBlocksPerSeq + page)] = block;
                const int64_t live = std::min(_case.pageSize, keys - page * _case.pageSize);
                for(int64_t h = 0; h < _case.kvHeads; ++h)
                {
                    for(int64_t t = 0; t < live; ++t)
                    {
                        const auto row = static_cast<size_t>(
                            ((block * _case.kvHeads + h) * _case.pageSize + t) * cacheRow());
                        for(size_t i = 0; i < d; ++i)
                        {
                            batch.k[row + i] = toBits(uniform(rng), dt);
                            auto& vPlane = _case.interleavedCache ? batch.k : batch.v;
                            vPlane[row + (_case.interleavedCache ? d : 0) + i]
                                = toBits(uniform(rng), dt);
                        }
                    }
                }
            }
        }
        return batch;
    }

    /// Key t of sequence b at K/V head h, element i (V at the +D half when interleaved).
    double
        cacheValue(const Batch& batch, int64_t b, int64_t t, int64_t h, int64_t i, bool value) const
    {
        const int32_t block
            = batch.pageTable[static_cast<size_t>(b * _g.maxBlocksPerSeq + t / _case.pageSize)];
        const auto row = static_cast<size_t>(
            ((block * _case.kvHeads + h) * _case.pageSize + t % _case.pageSize) * cacheRow());
        if(value && !_case.interleavedCache)
        {
            return fromBits(batch.v[row + static_cast<size_t>(i)], _case.dataType);
        }
        const int64_t half = value ? _case.headSize : 0;
        return fromBits(batch.k[row + static_cast<size_t>(half + i)], _case.dataType);
    }

    /// The gather-then-dense float64 reference: O over every live row, and the
    /// natural-log LSE ([b][h][r]); rows with no visible key give O = 0, LSE = -inf.
    void reference(const Batch& batch, std::vector<double>& out, std::vector<double>& lse) const
    {
        const int64_t d = _case.headSize;
        const int64_t groups = _case.queryHeads / _case.kvHeads;
        const double scale = 1.0 / std::sqrt(static_cast<double>(d));
        out.assign(static_cast<size_t>(_g.totalRows * tokenStride()), 0.0);
        lse.assign(static_cast<size_t>(_g.batch * _case.queryHeads * _g.queryRows),
                   -std::numeric_limits<double>::infinity());
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
            const int64_t sk = batch.seqLenKv[bi];
            const int64_t sq = batch.seqLenQ[bi];
            const int64_t diag = sk - sq; // bottom-right alignment
            for(int64_t h = 0; h < _case.queryHeads; ++h)
            {
                const int64_t kvh = h / groups;
                for(int64_t r = 0; r < sq; ++r)
                {
                    const auto qBase
                        = static_cast<size_t>(rowOf(batch, b, r) * tokenStride() + h * d);
                    std::vector<double> logits;
                    std::vector<int64_t> keys;
                    for(int64_t t = 0; t < sk; ++t)
                    {
                        if(t > r + diag || (_case.leftBound >= 0 && t < r + diag - _case.leftBound))
                        {
                            continue;
                        }
                        double dot = 0.0;
                        for(int64_t i = 0; i < d; ++i)
                        {
                            dot += fromBits(batch.q[qBase + static_cast<size_t>(i)], _case.dataType)
                                   * cacheValue(batch, b, t, kvh, i, false);
                        }
                        logits.push_back(dot * scale);
                        keys.push_back(t);
                    }
                    if(logits.empty())
                    {
                        continue;
                    }
                    const double peak = *std::max_element(logits.begin(), logits.end());
                    double sum = 0.0;
                    for(auto& logit : logits)
                    {
                        logit = std::exp(logit - peak);
                        sum += logit;
                    }
                    for(int64_t i = 0; i < d; ++i)
                    {
                        double acc = 0.0;
                        for(size_t j = 0; j < keys.size(); ++j)
                        {
                            acc += logits[j] * cacheValue(batch, b, keys[j], kvh, i, true);
                        }
                        out[qBase + static_cast<size_t>(i)] = acc / sum;
                    }
                    lse[static_cast<size_t>((b * _case.queryHeads + h) * _g.queryRows + r)]
                        = peak + std::log(sum);
                }
            }
        }
    }

private:
    const PagedCase& _case;
    Geometry _g;
};

/// A device allocation freed on scope exit.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
    {
        EXPECT_EQ(hipMalloc(&_ptr, std::max<size_t>(bytes, 16)), hipSuccess);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;
    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_ptr));
    }
    void* get() const
    {
        return _ptr;
    }
    template <typename T>
    void upload(const std::vector<T>& host) const
    {
        ASSERT_EQ(hipMemcpy(_ptr, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
                  hipSuccess);
    }
    template <typename T>
    void download(std::vector<T>& host) const
    {
        ASSERT_EQ(hipMemcpy(host.data(), _ptr, host.size() * sizeof(T), hipMemcpyDeviceToHost),
                  hipSuccess);
    }

private:
    void* _ptr = nullptr;
};

std::shared_ptr<TensorAttributes> makeTensor(int64_t uid,
                                             const std::string& name,
                                             DataType dataType,
                                             const std::vector<int64_t>& dims,
                                             const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid).set_name(name).set_data_type(dataType).set_dim(dims).set_stride(strides);
    return tensor;
}

std::shared_ptr<TensorAttributes> lengthVector(int64_t uid, const std::string& name, int64_t n)
{
    return makeTensor(uid, name, DataType::INT32, {n, 1, 1, 1}, {1, 1, 1, 1});
}

/// The paged SDPA graph of @p paged: vLLM's view of the TRITON_ATTN cache when
/// interleaved (K / V the two halves of [num_blocks, Hkv, page, 2D]).
std::shared_ptr<Graph> buildPagedGraph(const PagedCase& paged, const Geometry& g)
{
    auto graph = std::make_shared<Graph>();
    graph->set_name(paged.name)
        .set_io_data_type(paged.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    const int64_t d = paged.headSize;
    const int64_t token = paged.queryHeads * d;
    const std::vector<int64_t> qDims{g.batch, paged.queryHeads, g.queryRows, d};
    const std::vector<int64_t> qStrides{g.queryRows * token, d, token, 1};
    auto q = makeTensor(Q_UID, "Q", paged.dataType, qDims, qStrides);
    const int64_t row = paged.interleavedCache ? 2 * d : d;
    const std::vector<int64_t> cacheDims{g.numBlocks, paged.kvHeads, paged.pageSize, d};
    const std::vector<int64_t> cacheStrides{
        paged.kvHeads * paged.pageSize * row, paged.pageSize * row, row, 1};
    auto k = makeTensor(K_UID, "K", paged.dataType, cacheDims, cacheStrides);
    auto v = makeTensor(V_UID, "V", paged.dataType, cacheDims, cacheStrides);
    auto table = makeTensor(PAGE_TABLE_UID,
                            "page_table",
                            DataType::INT32,
                            {g.batch, 1, g.maxBlocksPerSeq, 1},
                            {g.maxBlocksPerSeq, g.maxBlocksPerSeq, 1, 1});
    std::shared_ptr<TensorAttributes> offsets;
    if(paged.form == QueryForm::RAGGED)
    {
        offsets = lengthVector(Q_OFFSET_UID, "q_offsets", g.batch + 1);
        q->set_ragged_offset(offsets).set_ragged_offset_multiplier(token);
    }

    SdpaAttributes attributes;
    attributes.set_name(paged.name)
        .set_attn_scale(1.0F / std::sqrt(static_cast<float>(d)))
        .set_diagonal_alignment(DiagonalAlignment::BOTTOM_RIGHT)
        .set_diagonal_band_right_bound(0)
        .set_paged_attention_k_table(table)
        .set_paged_attention_v_table(table)
        .set_seq_len_kv(lengthVector(SEQ_LEN_KV_UID, "seq_len_kv", g.batch))
        .set_paged_attention_max_seq_len_kv(
            static_cast<int32_t>(g.maxBlocksPerSeq * paged.pageSize))
        .set_padding_mask(true)
        .set_generate_stats(paged.stats);
    if(paged.leftBound >= 0)
    {
        attributes.set_diagonal_band_left_bound(paged.leftBound);
    }
    if(paged.form != QueryForm::DENSE)
    {
        attributes.set_seq_len_q(lengthVector(SEQ_LEN_Q_UID, "seq_len_q", g.batch));
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto& o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(paged.dataType)
        .set_dim(qDims)
        .set_stride(qStrides);
    if(offsets != nullptr)
    {
        o->set_ragged_offset(offsets).set_ragged_offset_multiplier(token);
    }
    if(paged.stats)
    {
        outputs[1]
            ->set_uid(LSE_UID)
            .set_name("LSE")
            .set_output(true)
            .set_data_type(DataType::FLOAT)
            .set_dim({g.batch, paged.queryHeads, g.queryRows, 1})
            .set_stride({paged.queryHeads * g.queryRows, g.queryRows, 1, 1});
    }
    return graph;
}

/// A small dense SDPA graph, with a max_seq_len_kv but no page tables when
/// @p maxSeqLenKv: the frontend's paged K/V rules (cuDNN's) reject exactly that one.
std::shared_ptr<Graph> pagedRuleProbe(bool maxSeqLenKv)
{
    constexpr int64_t SEQ = 16;
    constexpr int64_t D = 64;
    auto graph = std::make_shared<Graph>();
    graph->set_name("paged_rule_probe")
        .set_io_data_type(DataType::BFLOAT16)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);
    const std::vector<int64_t> dims{1, 1, SEQ, D};
    const std::vector<int64_t> strides{SEQ * D, SEQ * D, D, 1};
    auto q = makeTensor(Q_UID, "Q", DataType::BFLOAT16, dims, strides);
    auto k = makeTensor(K_UID, "K", DataType::BFLOAT16, dims, strides);
    auto v = makeTensor(V_UID, "V", DataType::BFLOAT16, dims, strides);
    SdpaAttributes attributes;
    attributes.set_name("paged_rule_probe")
        .set_attn_scale(1.0F / std::sqrt(static_cast<float>(D)))
        .set_generate_stats(false);
    if(maxSeqLenKv)
    {
        attributes.set_paged_attention_max_seq_len_kv(static_cast<int32_t>(SEQ));
    }
    auto outputs = graph->sdpa(q, k, v, attributes);
    outputs[0]
        ->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(DataType::BFLOAT16)
        .set_dim(dims)
        .set_stride(strides);
    return graph;
}

std::vector<PagedCase> pagedCases()
{
    const auto bf16 = DataType::BFLOAT16;
    const auto fp16 = DataType::HALF;
    return {
        // vLLM uniform decode, GQA 32/8, CUDA-graph padding rows (no keys) included.
        {"decode_bf16_d128_p16_gqa",
         bf16,
         QueryForm::DENSE,
         32,
         8,
         128,
         16,
         {0, 1, 17, 200, 513, 64},
         {1, 1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0, 0}},
        // Speculative decode (k = 2) with a sliding window, separate K / V planes.
        {"specdecode_fp16_d64_p32_window",
         fp16,
         QueryForm::DENSE,
         8,
         2,
         64,
         32,
         {3, 70, 129, 0},
         {3, 3, 3, 3},
         {0, 0, 0, 0},
         31,
         false,
         false},
        // Dense Q with seq_len_q, stats, a head width served by the 96 bucket.
        {"denseq_seqlenq_bf16_d80_p16_lse",
         bf16,
         QueryForm::DENSE_SEQ_LEN_Q,
         8,
         1,
         80,
         16,
         {40, 90, 0, 33},
         {1, 4, 0, 2},
         {3, 0, 4, 2},
         -1,
         true},
        // A mixed batch: chunked prefill beside decodes, gaps between sequences.
        {"ragged_bf16_d128_p16_mixed",
         bf16,
         QueryForm::RAGGED,
         8,
         2,
         128,
         16,
         {20, 0, 300, 50, 130},
         {5, 0, 40, 1, 130},
         {0, 2, 0, 3, 0}},
        {"ragged_fp16_d64_p64_window",
         fp16,
         QueryForm::RAGGED,
         4,
         4,
         64,
         64,
         {150, 9, 70},
         {30, 9, 1},
         {0, 0, 0},
         15},
        {"ragged_bf16_d256_p16", bf16, QueryForm::RAGGED, 4, 2, 256, 16, {90, 60}, {40, 2}, {0, 1}},
    };
}

class IntegrationGpuGfx1151WmmaAttentionPaged
    : public test_utilities::IntegrationGraphVerificationHarness<float, PagedCase>
{
protected:
    using Harness = test_utilities::IntegrationGraphVerificationHarness<float, PagedCase>;

    void SetUp() override
    {
        Harness::SetUp();
        if(IsSkipped() || HasFatalFailure())
        {
            return;
        }
        const auto arch = hip_kernel_provider_common::getDeviceString(_stream);
        if(arch.rfind("gfx11", 0) != 0 && arch.rfind("gfx12", 0) != 0)
        {
            GTEST_SKIP() << ENGINE_NAME << " ships for RDNA3/3.5/4 only; this device is " << arch;
        }
        // A frontend without the paged K/V rules (SdpaFwdNode) also compares a paged
        // container's num_blocks against the batch, so it rejects every graph below before
        // any engine sees it. Probe for the rules rather than fail on that: the dense probe
        // must validate, and with a max_seq_len_kv but no page tables it must not.
        const auto denseProbe = pagedRuleProbe(false)->validate();
        ASSERT_EQ(denseProbe.code, ErrorCode::OK) << denseProbe.err_msg;
        if(pagedRuleProbe(true)->validate().code == ErrorCode::OK)
        {
            GTEST_SKIP() << "the hipDNN frontend does not validate paged K/V SDPA graphs (no "
                            "paged-attention rules in SdpaFwdNode), so it cannot build one";
        }
    }

    /// Builds the engine's plan for @p graph; a decline fails the case.
    void buildPinnedPlan(Graph& graph)
    {
        auto result = graph.build_operation_graph(_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.create_execution_plan_ext(
            hipdnn_data_sdk::utilities::engineNameToId(ENGINE_NAME), {});
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = graph.build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        int64_t workspace = -1;
        ASSERT_EQ(graph.get_workspace_size(workspace).code, ErrorCode::OK);
        ASSERT_EQ(workspace, 0) << "the paged prefill object needs no workspace";
    }
};

/// The device buffers of one plan; the variant pack points at them for every execute.
struct DeviceInputs
{
    DeviceInputs(const PagedCase& paged, const Geometry& g, const PagedProblem& problem)
        : q(static_cast<size_t>(g.totalRows * problem.tokenStride()) * sizeof(uint16_t))
        , o(static_cast<size_t>(g.totalRows * problem.tokenStride()) * sizeof(uint16_t))
        , k(static_cast<size_t>(g.numBlocks * paged.kvHeads * paged.pageSize * problem.cacheRow())
            * sizeof(uint16_t))
        , v(paged.interleavedCache ? 16 : k.size)
        , lse(static_cast<size_t>(g.batch * paged.queryHeads * g.queryRows) * sizeof(float))
        , table(static_cast<size_t>(g.batch * g.maxBlocksPerSeq) * sizeof(int32_t))
        , seqLenKv(static_cast<size_t>(g.batch) * sizeof(int32_t))
        , seqLenQ(static_cast<size_t>(g.batch) * sizeof(int32_t))
        , offsets(static_cast<size_t>(g.batch + 1) * sizeof(int32_t))
    {
        auto* cache = static_cast<char*>(k.buffer.get());
        // V is K's buffer at +D elements in the interleaved cache.
        const size_t vOffset = static_cast<size_t>(paged.headSize) * sizeof(uint16_t);
        void* vPointer
            = paged.interleavedCache ? static_cast<void*>(cache + vOffset) : v.buffer.get();
        variantPack = {{Q_UID, q.buffer.get()},
                       {K_UID, k.buffer.get()},
                       {V_UID, vPointer},
                       {O_UID, o.buffer.get()},
                       {PAGE_TABLE_UID, table.buffer.get()},
                       {SEQ_LEN_KV_UID, seqLenKv.buffer.get()}};
        if(paged.form != QueryForm::DENSE)
        {
            variantPack[SEQ_LEN_Q_UID] = seqLenQ.buffer.get();
        }
        if(paged.form == QueryForm::RAGGED)
        {
            variantPack[Q_OFFSET_UID] = offsets.buffer.get();
        }
        if(paged.stats)
        {
            variantPack[LSE_UID] = lse.buffer.get();
        }
    }

    struct Sized
    {
        explicit Sized(size_t bytes)
            : size(bytes)
            , buffer(bytes)
        {
        }
        size_t size;
        DeviceBuffer buffer;
    };

    /// Uploads @p batch and resets O / LSE to NaN; synchronous, outside any capture.
    void upload(const PagedCase& paged, const Batch& batch) const
    {
        q.buffer.upload(batch.q);
        k.buffer.upload(batch.k);
        if(!paged.interleavedCache)
        {
            v.buffer.upload(batch.v);
        }
        table.buffer.upload(batch.pageTable);
        seqLenKv.buffer.upload(batch.seqLenKv);
        seqLenQ.buffer.upload(batch.seqLenQ);
        offsets.buffer.upload(batch.qOffsets);
        ASSERT_EQ(hipMemset(o.buffer.get(), 0xFF, o.size), hipSuccess);
        ASSERT_EQ(hipMemset(lse.buffer.get(), 0xFF, lse.size), hipSuccess);
    }

    Sized q;
    Sized o;
    Sized k;
    Sized v;
    Sized lse;
    Sized table;
    Sized seqLenKv;
    Sized seqLenQ;
    Sized offsets;
    std::unordered_map<int64_t, void*> variantPack;
};

/// Compares O (and LSE) on the device with the reference of @p batch.
void expectMatchesReference(const PagedCase& paged,
                            const Geometry& g,
                            const PagedProblem& problem,
                            const Batch& batch,
                            const DeviceInputs& device,
                            const char* phase)
{
    std::vector<double> want;
    std::vector<double> wantLse;
    problem.reference(batch, want, wantLse);
    std::vector<uint16_t> got(static_cast<size_t>(g.totalRows * problem.tokenStride()));
    device.o.buffer.download(got);
    const double tolerance = paged.dataType == DataType::BFLOAT16 ? 4e-2 : 2e-2;
    const int64_t token = problem.tokenStride();
    int64_t mismatches = 0;
    for(int64_t b = 0; b < g.batch; ++b)
    {
        const auto bi = static_cast<size_t>(b);
        const int64_t span = int64_t{batch.seqLenQ[bi]} + batch.gaps[bi];
        for(int64_t r = 0; r < span; ++r)
        {
            const bool live = r < batch.seqLenQ[bi];
            const auto base = static_cast<size_t>(problem.rowOf(batch, b, r) * token);
            for(size_t i = 0; i < static_cast<size_t>(token); ++i)
            {
                const double value = fromBits(got[base + i], paged.dataType);
                const bool ok
                    = live ? std::isfinite(value) && std::abs(value - want[base + i]) <= tolerance
                           : std::isnan(value);
                if(!ok && mismatches++ < 8)
                {
                    ADD_FAILURE() << phase << ": sequence " << b << " row " << r << " element " << i
                                  << (live ? "" : " (past seq_len_q, must be untouched)")
                                  << ": got " << value << ", want "
                                  << (live ? want[base + i]
                                           : std::numeric_limits<double>::quiet_NaN());
                }
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << phase;

    if(!paged.stats)
    {
        return;
    }
    std::vector<float> gotLse(wantLse.size());
    device.lse.buffer.download(gotLse);
    for(int64_t b = 0; b < g.batch; ++b)
    {
        for(int64_t h = 0; h < paged.queryHeads; ++h)
        {
            for(int64_t r = 0; r < batch.seqLenQ[static_cast<size_t>(b)]; ++r)
            {
                const auto at = static_cast<size_t>((b * paged.queryHeads + h) * g.queryRows + r);
                if(std::isinf(wantLse[at]))
                {
                    EXPECT_TRUE(std::isinf(gotLse[at]) && gotLse[at] < 0)
                        << phase << ": LSE of an empty row, sequence " << b;
                }
                else
                {
                    EXPECT_NEAR(gotLse[at], wantLse[at], 1e-2 + 1e-5 * std::abs(wantLse[at]))
                        << phase << ": LSE, sequence " << b << " head " << h << " row " << r;
                }
            }
        }
    }
}

template <typename T>
std::vector<T> reversed(std::vector<T> values)
{
    std::reverse(values.begin(), values.end());
    return values;
}

/// One plan executes the case directly, then the same plan is captured into a HIP graph
/// and replayed twice with other lengths (the case's reversed, then rotated), other block
/// ids and other data: nothing the kernel reads comes from the host at execute.
TEST_P(IntegrationGpuGfx1151WmmaAttentionPaged, MatchesTheGatherThenDenseReferenceAndReplays)
{
    const auto& paged = GetParam();
    const auto g = geometryOf(paged);
    const PagedProblem problem(paged, g);
    auto graph = buildPagedGraph(paged, g);
    ASSERT_NO_FATAL_FAILURE(buildPinnedPlan(*graph));

    DeviceInputs device(paged, g, problem);
    const auto first = problem.makeBatch(paged.seqLenKv, paged.seqLenQ, paged.gaps, 1);
    ASSERT_NO_FATAL_FAILURE(device.upload(paged, first));
    auto result = graph->execute(_handle, device.variantPack, nullptr);
    ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
    ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
    expectMatchesReference(paged, g, problem, first, device, "direct execute");

    hipGraph_t captured = nullptr;
    ASSERT_EQ(hipStreamBeginCapture(_stream, hipStreamCaptureModeThreadLocal), hipSuccess);
    result = graph->execute(_handle, device.variantPack, nullptr);
    ASSERT_EQ(hipStreamEndCapture(_stream, &captured), hipSuccess);
    ASSERT_EQ(result.code, ErrorCode::OK) << "execute under capture: " << result.err_msg;
    hipGraphExec_t replay = nullptr;
    ASSERT_EQ(hipGraphInstantiate(&replay, captured, nullptr, nullptr, 0), hipSuccess);

    auto rotate = [](std::vector<int32_t> values) {
        std::rotate(values.begin(), values.begin() + 1, values.end());
        return values;
    };
    const std::vector<std::pair<const char*, Batch>> replays{
        {"replay with reversed lengths",
         problem.makeBatch(
             reversed(paged.seqLenKv), reversed(paged.seqLenQ), reversed(paged.gaps), 2)},
        {"replay with rotated lengths",
         problem.makeBatch(rotate(paged.seqLenKv), rotate(paged.seqLenQ), rotate(paged.gaps), 3)},
    };
    for(const auto& [phase, batch] : replays)
    {
        ASSERT_NO_FATAL_FAILURE(device.upload(paged, batch));
        ASSERT_EQ(hipGraphLaunch(replay, _stream), hipSuccess);
        ASSERT_EQ(hipStreamSynchronize(_stream), hipSuccess);
        expectMatchesReference(paged, g, problem, batch, device, phase);
    }
    EXPECT_EQ(hipGraphExecDestroy(replay), hipSuccess);
    EXPECT_EQ(hipGraphDestroy(captured), hipSuccess);
}

INSTANTIATE_TEST_SUITE_P(Quick,
                         IntegrationGpuGfx1151WmmaAttentionPaged,
                         ::testing::ValuesIn(pagedCases()),
                         [](const auto& info) { return std::string(info.param.name); });

} // namespace
} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
