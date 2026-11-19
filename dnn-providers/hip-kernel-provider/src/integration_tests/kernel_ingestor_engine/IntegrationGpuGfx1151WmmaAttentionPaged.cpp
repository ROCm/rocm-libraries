// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx1151WmmaAttentionCatalog.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

/**
 * @file IntegrationGpuGfx1151WmmaAttentionPaged.cpp
 * @brief hipkernel:Gfx1151WmmaAttention on the cuDNN paged-K/V SDPA graph, end to end
 *        through the frontend on an RDNA (gfx11 / gfx12) device: vLLM's uniform decode
 *        (dense Q) and mixed batches (ragged Q) over a shuffled, NaN-poisoned paged cache
 *        -- the interleaved TRITON_ATTN K|V layout or separate planes, paged through one
 *        page table or through separate K and V tables that place V's pages elsewhere --
 *        against hipDNN's CPU SDPA reference (CpuFpReferenceSdpa, double compute), and one
 *        plan captured into a HIP graph and replayed with different sequence lengths, block
 *        ids and data; with and without attention sinks (sink_token [1, Hq, 1, 1], FLOAT as
 *        hipDNN takes it, or in the I/O data type, upcast exactly to the reference's).
 *
 * Every graph carries seq_len_q and seq_len_kv (hipDNN's paged contract). Unused cache
 * slots and padding are NaN, so a read of anything outside the used page-table entries or
 * past seq_len_kv shows up as a non-finite output. O starts NaN: every live row must be
 * written (zeros for a row with no visible key); a dense Q's padding rows past seq_len_q,
 * up to Sq, must be written as O = 0 and LSE = -inf (SdpaSemanticsRegister SEM-008), with
 * or without a sink. A ragged Q follows RFC 0014: its origins are in reverse sequence
 * order, ragged_offset[B] is poisoned, and the rows of an offset block past seq_len_q
 * belong to no sequence and must keep their NaN.
 *
 * The Q-dtype sink cases need the frontend to accept a sink_token in the Q data type; a
 * frontend that takes only cuDNN's FLOAT sink (SEM-022) rejects them, and they skip with a
 * message. The FLOAT sink cases always run.
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
constexpr int64_t SINK_UID = 10;
constexpr int64_t PAGE_TABLE_V_UID = 11;

constexpr uint16_t NAN_BITS = 0xFFFF; // a NaN in both FP16 and BF16
constexpr int32_t UNUSED_BLOCK = std::numeric_limits<int32_t>::max();
constexpr int32_t RAGGED_OFFSET_END_POISON = -7; // ragged_offset[B], which is not read

enum class QueryForm
{
    DENSE, // uniform decode: [B, Hq, q_len, D], seq_len_q = q_len for every sequence
    DENSE_SEQ_LEN_Q, // dense Q whose live rows per sequence are seq_len_q, padding after
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
    // Rows after each sequence's live rows: padding up to Sq (DENSE_SEQ_LEN_Q), or rows of
    // its ragged offset block that belong to no sequence (RAGGED).
    std::vector<int32_t> gaps;
    // cuDNN's left_bound (SEM-001): L keys ending at the diagonal, the diagonal included.
    int64_t leftBound = -1;
    bool stats = false;
    bool interleavedCache = true;
    // Attention sinks: one logit per query head, in the I/O data type (floatSink: FLOAT).
    bool sinks = false;
    // cuDNN's separate page_table_v: V's pages through their own table, in another order.
    bool separateVTable = false;
    // V in its own plane with padded rows, heads and pages: none of V's page, head and
    // token strides is K's, so V addressed through K's strides reads the wrong rows.
    bool paddedV = false;
    // With sinks: sink_token is FLOAT (hipDNN's form) rather than in the I/O data type.
    bool floatSink = false;
};

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest customization point
void PrintTo(const PagedCase& paged, std::ostream* os)
{
    *os << paged.name;
}

/// V's element strides {page, head, token}: the +D half of the interleaved cache (K's
/// strides), K's layout in its own plane, or with paddedV its own padded layout.
std::array<int64_t, 3> vStridesOf(const PagedCase& paged)
{
    const int64_t pad = paged.paddedV ? 8 : 0;
    const int64_t row = (paged.interleavedCache ? 2 * paged.headSize : paged.headSize) + pad;
    const int64_t head = paged.pageSize * row + pad;
    return {paged.kvHeads * head + 2 * pad, head, row};
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

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;

uint16_t toBits(float value, DataType dataType)
{
    return dataType == DataType::HALF ? half(value).data : bfloat16(value).data;
}

double fromBits(uint16_t bits, DataType dataType)
{
    return dataType == DataType::HALF ? static_cast<double>(half::from_bits(bits))
                                      : static_cast<double>(bfloat16::from_bits(bits));
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
    std::vector<int32_t> pageTableV; // [B, M], separate V table only
    std::vector<uint16_t> sinks; // [Hq], Q-dtype attention sinks only
    std::vector<float> floatSinks; // [Hq], FLOAT attention sinks only
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
        // Ragged origins in reverse sequence order with ragged_offset[B] poisoned: with
        // seq_len_q bound, origins may come in any order and ragged_offset[B] is neither
        // used nor validated (RFC 0014).
        batch.qOffsets.assign(static_cast<size_t>(_g.batch + 1), RAGGED_OFFSET_END_POISON);
        int32_t origin = 0;
        for(int64_t b = _g.batch - 1; b >= 0; --b)
        {
            const auto bi = static_cast<size_t>(b);
            batch.qOffsets[bi] = origin;
            origin += batch.seqLenQ[bi] + batch.gaps[bi];
        }
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
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
            batch.v.assign(static_cast<size_t>(_g.numBlocks * vStridesOf(_case)[0]), NAN_BITS);
        }
        std::vector<int32_t> order(static_cast<size_t>(_g.numBlocks));
        std::iota(order.begin(), order.end(), 0);
        std::shuffle(order.begin(), order.end(), rng);
        // A separate V table places each V page in another block than its K page.
        std::vector<int32_t> orderV = order;
        if(_case.separateVTable)
        {
            std::shuffle(orderV.begin(), orderV.end(), rng);
        }
        batch.pageTable.assign(static_cast<size_t>(_g.batch * _g.maxBlocksPerSeq), UNUSED_BLOCK);
        if(_case.separateVTable)
        {
            batch.pageTableV = batch.pageTable;
        }
        size_t next = 0;
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
            const int64_t keys = batch.seqLenKv[bi];
            const int64_t pages = (keys + _case.pageSize - 1) / _case.pageSize;
            for(int64_t page = 0; page < pages; ++page)
            {
                const auto entry = static_cast<size_t>(b * _g.maxBlocksPerSeq + page);
                const int32_t block = order[next];
                const int32_t blockV = orderV[next];
                ++next;
                batch.pageTable[entry] = block;
                if(_case.separateVTable)
                {
                    batch.pageTableV[entry] = blockV;
                }
                const int64_t live = std::min(_case.pageSize, keys - page * _case.pageSize);
                for(int64_t h = 0; h < _case.kvHeads; ++h)
                {
                    for(int64_t t = 0; t < live; ++t)
                    {
                        const auto row = static_cast<size_t>(
                            ((block * _case.kvHeads + h) * _case.pageSize + t) * cacheRow());
                        const auto rowV = vElement(blockV, h, t);
                        auto& vPlane = _case.interleavedCache ? batch.k : batch.v;
                        for(size_t i = 0; i < d; ++i)
                        {
                            batch.k[row + i] = toBits(uniform(rng), dt);
                            vPlane[rowV + i] = toBits(uniform(rng), dt);
                        }
                    }
                }
            }
        }
        if(_case.sinks)
        {
            batch.sinks.resize(static_cast<size_t>(_case.queryHeads));
            batch.floatSinks.resize(batch.sinks.size());
            for(size_t h = 0; h < batch.sinks.size(); ++h)
            {
                const float logit = 2.0F * uniform(rng);
                batch.sinks[h] = toBits(logit, dt);
                batch.floatSinks[h] = logit;
            }
        }
        return batch;
    }

    /// hipDNN's CPU SDPA reference (CpuFpReferenceSdpa, double compute) over @p batch: K
    /// and V gathered through their page tables from views of the cache (V through its own
    /// strides), Q scattered into a dense [B, Hq, Sq, D] with seq_len_q / seq_len_kv
    /// padding, and O gathered back to the device O's rows ([rows, Hq, D]); the natural-log
    /// LSE is [b][h][r]. The sink upcasts exactly to the reference's FLOAT sink.
    void reference(Batch batch, std::vector<double>& out, std::vector<double>& lse) const
    {
        if(_case.dataType == DataType::BFLOAT16)
        {
            referenceAs<bfloat16>(batch, out, lse);
        }
        else
        {
            referenceAs<half>(batch, out, lse);
        }
    }

private:
    template <typename T>
    void referenceAs(Batch& batch, std::vector<double>& out, std::vector<double>& lse) const
    {
        using hipdnn_data_sdk::utilities::ShallowTensor;
        using hipdnn_data_sdk::utilities::Tensor;
        using hipdnn_test_sdk::utilities::CpuFpReferenceSdpa;
        using hipdnn_test_sdk::utilities::SdpaFwdOptions;
        const int64_t d = _case.headSize;
        const int64_t heads = _case.queryHeads;
        const int64_t rows = _g.queryRows;
        const int64_t token = tokenStride();

        Tensor<T> q({_g.batch, heads, rows, d});
        q.fillWithValue(T(0.0F));
        Tensor<int32_t> seqLenQ({_g.batch, 1, 1, 1});
        Tensor<int32_t> seqLenKv({_g.batch, 1, 1, 1});
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            const auto bi = static_cast<size_t>(b);
            seqLenQ.setHostValue(batch.seqLenQ[bi], std::vector<int64_t>{b, 0, 0, 0});
            seqLenKv.setHostValue(batch.seqLenKv[bi], std::vector<int64_t>{b, 0, 0, 0});
            for(int64_t r = 0; r < batch.seqLenQ[bi]; ++r)
            {
                for(int64_t h = 0; h < heads; ++h)
                {
                    const auto base = static_cast<size_t>(rowOf(batch, b, r) * token + h * d);
                    for(int64_t i = 0; i < d; ++i)
                    {
                        q.setHostValue(T::from_bits(batch.q[base + static_cast<size_t>(i)]),
                                       std::vector<int64_t>{b, h, r, i});
                    }
                }
            }
        }

        const std::vector<int64_t> cacheDims{_g.numBlocks, _case.kvHeads, _case.pageSize, d};
        const ShallowTensor<T> kCache(batch.k.data(),
                                      cacheDims,
                                      {_case.kvHeads * _case.pageSize * cacheRow(),
                                       _case.pageSize * cacheRow(),
                                       cacheRow(),
                                       1});
        const auto vS = vStridesOf(_case);
        uint16_t* vPlane = _case.interleavedCache ? batch.k.data() + d : batch.v.data();
        const ShallowTensor<T> vCache(vPlane, cacheDims, {vS[0], vS[1], vS[2], 1});
        const std::vector<int64_t> tableDims{_g.batch, 1, _g.maxBlocksPerSeq, 1};
        const std::vector<int64_t> tableStrides{_g.maxBlocksPerSeq, _g.maxBlocksPerSeq, 1, 1};
        const ShallowTensor<int32_t> table(batch.pageTable.data(), tableDims, tableStrides);
        const ShallowTensor<int32_t> tableV(_case.separateVTable ? batch.pageTableV.data()
                                                                 : batch.pageTable.data(),
                                            tableDims,
                                            tableStrides);
        const int64_t maxSeqKv = _g.maxBlocksPerSeq * _case.pageSize;
        const auto k = CpuFpReferenceSdpa::gatherPagedKv(kCache, table, seqLenKv, maxSeqKv);
        const auto v = CpuFpReferenceSdpa::gatherPagedKv(vCache, tableV, seqLenKv, maxSeqKv);

        Tensor<double> o({_g.batch, heads, rows, d});
        Tensor<float> stats({_g.batch, heads, rows, 1});
        Tensor<float> sink({1, heads, 1, 1});
        auto options = SdpaFwdOptions<double>{}
                           .setScale(1.0F / std::sqrt(static_cast<float>(d)))
                           .setBand(_case.leftBound, 0, false)
                           .setSeqLens(&seqLenQ, &seqLenKv)
                           .setLse(&stats);
        if(_case.sinks)
        {
            for(int64_t h = 0; h < heads; ++h)
            {
                const auto i = static_cast<size_t>(h);
                sink.setHostValue(
                    _case.floatSink ? batch.floatSinks[i]
                                    : static_cast<float>(fromBits(batch.sinks[i], _case.dataType)),
                    std::vector<int64_t>{0, h, 0, 0});
            }
            options.setSink(&sink);
        }
        CpuFpReferenceSdpa::forward<T, T, T, double, double>(q, k, v, o, options);

        out.assign(static_cast<size_t>(_g.totalRows * token), 0.0);
        lse.assign(static_cast<size_t>(_g.batch * heads * rows), 0.0);
        for(int64_t b = 0; b < _g.batch; ++b)
        {
            for(int64_t h = 0; h < heads; ++h)
            {
                for(int64_t r = 0; r < rows; ++r)
                {
                    lse[static_cast<size_t>((b * heads + h) * rows + r)]
                        = static_cast<double>(stats.getHostValue(std::vector<int64_t>{b, h, r, 0}));
                }
                for(int64_t r = 0; r < batch.seqLenQ[static_cast<size_t>(b)]; ++r)
                {
                    const auto base = static_cast<size_t>(rowOf(batch, b, r) * token + h * d);
                    for(int64_t i = 0; i < d; ++i)
                    {
                        out[base + static_cast<size_t>(i)]
                            = o.getHostValue(std::vector<int64_t>{b, h, r, i});
                    }
                }
            }
        }
    }

    /// The first element of V's row at token @p t of head @p h in page @p block.
    size_t vElement(int64_t block, int64_t h, int64_t t) const
    {
        const auto s = vStridesOf(_case);
        return static_cast<size_t>(block * s[0] + h * s[1] + t * s[2]
                                   + (_case.interleavedCache ? _case.headSize : 0));
    }

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
/// interleaved (K / V the two halves of [num_blocks, Hkv, page, 2D]). A sink case's
/// sink_token is in @p sinkType (the I/O data type when unset).
std::shared_ptr<Graph> buildPagedGraph(const PagedCase& paged,
                                       const Geometry& g,
                                       std::optional<DataType> sinkType = std::nullopt)
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
    const auto vS = vStridesOf(paged);
    auto v = makeTensor(V_UID, "V", paged.dataType, cacheDims, {vS[0], vS[1], vS[2], 1});
    const std::vector<int64_t> tableDims{g.batch, 1, g.maxBlocksPerSeq, 1};
    const std::vector<int64_t> tableStrides{g.maxBlocksPerSeq, g.maxBlocksPerSeq, 1, 1};
    auto table = makeTensor(PAGE_TABLE_UID, "page_table", DataType::INT32, tableDims, tableStrides);
    auto tableV
        = paged.separateVTable
              ? makeTensor(
                    PAGE_TABLE_V_UID, "page_table_v", DataType::INT32, tableDims, tableStrides)
              : table;
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
        .set_paged_attention_v_table(tableV)
        .set_seq_len_q(lengthVector(SEQ_LEN_Q_UID, "seq_len_q", g.batch))
        .set_seq_len_kv(lengthVector(SEQ_LEN_KV_UID, "seq_len_kv", g.batch))
        .set_paged_attention_max_seq_len_kv(
            static_cast<int32_t>(g.maxBlocksPerSeq * paged.pageSize))
        .set_padding_mask(true)
        .set_generate_stats(paged.stats);
    if(paged.sinks)
    {
        attributes.set_sink_token(
            makeTensor(SINK_UID,
                       "sink",
                       sinkType.value_or(paged.floatSink ? DataType::FLOAT : paged.dataType),
                       {1, paged.queryHeads, 1, 1},
                       {paged.queryHeads, 1, 1, 1}));
    }
    if(paged.leftBound >= 0)
    {
        attributes.set_diagonal_band_left_bound(paged.leftBound);
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

std::vector<PagedCase> pagedCases()
{
    const auto bf16 = DataType::BFLOAT16;
    const auto fp16 = DataType::HALF;
    std::vector<PagedCase> cases{
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
        // Speculative decode (k = 2) with a 32-key sliding window, separate K / V planes.
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
         32,
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
         16},
        {"ragged_bf16_d256_p16", bf16, QueryForm::RAGGED, 4, 2, 256, 16, {90, 60}, {40, 2}, {0, 1}},
        // gpt-oss: GQA 64/8 decode with sinks and a 128-key window, padding rows and a
        // sequence with no keys (O = 0, LSE = the sink).
        {"sinks_decode_bf16_d64_p16_window_lse",
         bf16,
         QueryForm::DENSE,
         64,
         8,
         64,
         16,
         {0, 1, 300, 129},
         {1, 1, 1, 1},
         {0, 0, 0, 0},
         128,
         true,
         true,
         true},
        {"sinks_denseq_seqlenq_fp16_d128_p32_lse",
         fp16,
         QueryForm::DENSE_SEQ_LEN_Q,
         8,
         2,
         128,
         32,
         {70, 0, 33},
         {4, 2, 1},
         {0, 2, 3},
         -1,
         true,
         false,
         true},
        {"sinks_ragged_bf16_d64_p64_mixed",
         bf16,
         QueryForm::RAGGED,
         8,
         1,
         64,
         64,
         {20, 0, 300, 140},
         {5, 3, 70, 1},
         {0, 1, 0, 2},
         -1,
         false,
         true,
         true},
        // Separate K and V page tables (cuDNN's page_table_k / page_table_v): a mixed
        // batch through per-row lookups (page 16) over separate planes, and decode with
        // stats through per-tile lookups (page 64) over the interleaved cache.
        {"split_tables_ragged_bf16_d128_p16_mixed",
         bf16,
         QueryForm::RAGGED,
         8,
         2,
         128,
         16,
         {20, 0, 300, 50},
         {5, 0, 40, 1},
         {0, 2, 0, 3},
         -1,
         false,
         false,
         false,
         true},
        {"split_tables_denseq_seqlenq_fp16_d64_p64_window_lse",
         fp16,
         QueryForm::DENSE_SEQ_LEN_Q,
         8,
         2,
         64,
         64,
         {3, 200, 129, 0},
         {3, 1, 2, 0},
         {0, 2, 1, 3},
         64,
         true,
         true,
         false,
         true},
        // V in its own padded plane: its page, head and token strides all differ from K's
        // (with a ragged Q, and with dense-Q decode, separate tables and stats).
        {"padded_v_ragged_fp16_d128_p32_mixed",
         fp16,
         QueryForm::RAGGED,
         8,
         2,
         128,
         32,
         {40, 0, 200, 65},
         {6, 0, 33, 1},
         {0, 1, 0, 2},
         -1,
         false,
         false,
         false,
         false,
         true},
        {"padded_v_split_tables_decode_bf16_d64_p16_lse",
         bf16,
         QueryForm::DENSE,
         16,
         4,
         64,
         16,
         {0, 33, 130},
         {1, 1, 1},
         {0, 0, 0},
         -1,
         true,
         false,
         false,
         true,
         true},
        // The 96 and 256 buckets at pages 32 and 64, and the 128 bucket at page 64.
        {"decode_fp16_d96_p32_gqa_lse",
         fp16,
         QueryForm::DENSE,
         16,
         4,
         96,
         32,
         {0, 1, 33, 97, 260},
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0},
         -1,
         true},
        {"ragged_bf16_d96_p64_mixed_planes",
         bf16,
         QueryForm::RAGGED,
         8,
         2,
         96,
         64,
         {150, 0, 65, 9},
         {24, 0, 2, 9},
         {0, 2, 1, 0},
         -1,
         false,
         false},
        {"ragged_fp16_d256_p32_window",
         fp16,
         QueryForm::RAGGED,
         4,
         2,
         256,
         32,
         {90, 33, 0},
         {12, 1, 0},
         {0, 2, 1},
         48},
        {"denseq_seqlenq_bf16_d256_p64_lse",
         bf16,
         QueryForm::DENSE_SEQ_LEN_Q,
         4,
         1,
         256,
         64,
         {130, 64, 0, 1},
         {3, 1, 0, 1},
         {0, 2, 3, 2},
         -1,
         true},
        {"decode_bf16_d128_p64_window_lse",
         bf16,
         QueryForm::DENSE,
         16,
         2,
         128,
         64,
         {0, 1, 64, 65, 300},
         {1, 1, 1, 1, 1},
         {0, 0, 0, 0, 0},
         100,
         true},
    };
    // Each sink case again with hipDNN's FLOAT sink_token, which needs no frontend change.
    const std::map<std::string, const char*> floatNames{
        {"sinks_decode_bf16_d64_p16_window_lse", "float_sinks_decode_bf16_d64_p16_window_lse"},
        {"sinks_denseq_seqlenq_fp16_d128_p32_lse", "float_sinks_denseq_seqlenq_fp16_d128_p32_lse"},
        {"sinks_ragged_bf16_d64_p64_mixed", "float_sinks_ragged_bf16_d64_p64_mixed"}};
    const size_t authored = cases.size();
    for(size_t i = 0; i < authored; ++i)
    {
        if(cases[i].sinks)
        {
            PagedCase floatCase = cases[i];
            floatCase.name = floatNames.at(cases[i].name);
            floatCase.floatSink = true;
            cases.push_back(floatCase);
        }
    }
    return cases;
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
        if(std::find(GFX1151_WMMA_ATTENTION_ARCHS.begin(), GFX1151_WMMA_ATTENTION_ARCHS.end(), arch)
           == GFX1151_WMMA_ATTENTION_ARCHS.end())
        {
            GTEST_SKIP() << ENGINE_NAME << " ships no object for this device, " << arch;
        }
        // The Q-dtype sink cases need the frontend to accept a sink_token in the Q data
        // type. One that takes only cuDNN's FLOAT sink rejects them before any engine sees
        // them.
        const auto& paged = GetParam();
        if(paged.sinks && !paged.floatSink)
        {
            const auto g = geometryOf(paged);
            const auto qTypeSink = buildPagedGraph(paged, g)->validate();
            if(qTypeSink.code != ErrorCode::OK
               && buildPagedGraph(paged, g, DataType::FLOAT)->validate().code == ErrorCode::OK)
            {
                GTEST_SKIP() << "the hipDNN frontend accepts only a FLOAT sink_token (cuDNN, "
                                "SdpaSemanticsRegister SEM-022); this case passes the sink in "
                                "the Q data type, so it needs a frontend that accepts a "
                                "Q-dtype sink (the float_sinks_ cases cover hipDNN's form): "
                             << qTypeSink.err_msg;
            }
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
        , v(paged.interleavedCache
                ? 16
                : static_cast<size_t>(g.numBlocks * vStridesOf(paged)[0]) * sizeof(uint16_t))
        , lse(static_cast<size_t>(g.batch * paged.queryHeads * g.queryRows) * sizeof(float))
        , table(static_cast<size_t>(g.batch * g.maxBlocksPerSeq) * sizeof(int32_t))
        , tableV(paged.separateVTable ? table.size : 16)
        , seqLenKv(static_cast<size_t>(g.batch) * sizeof(int32_t))
        , seqLenQ(static_cast<size_t>(g.batch) * sizeof(int32_t))
        , offsets(static_cast<size_t>(g.batch + 1) * sizeof(int32_t))
        , sinks(static_cast<size_t>(paged.queryHeads)
                * (paged.floatSink ? sizeof(float) : sizeof(uint16_t)))
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
                       {SEQ_LEN_Q_UID, seqLenQ.buffer.get()},
                       {SEQ_LEN_KV_UID, seqLenKv.buffer.get()}};
        if(paged.separateVTable)
        {
            variantPack[PAGE_TABLE_V_UID] = tableV.buffer.get();
        }
        if(paged.form == QueryForm::RAGGED)
        {
            variantPack[Q_OFFSET_UID] = offsets.buffer.get();
        }
        if(paged.stats)
        {
            variantPack[LSE_UID] = lse.buffer.get();
        }
        if(paged.sinks)
        {
            variantPack[SINK_UID] = sinks.buffer.get();
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
        if(paged.separateVTable)
        {
            tableV.buffer.upload(batch.pageTableV);
        }
        seqLenKv.buffer.upload(batch.seqLenKv);
        seqLenQ.buffer.upload(batch.seqLenQ);
        offsets.buffer.upload(batch.qOffsets);
        if(paged.sinks && paged.floatSink)
        {
            sinks.buffer.upload(batch.floatSinks);
        }
        else if(paged.sinks)
        {
            sinks.buffer.upload(batch.sinks);
        }
        ASSERT_EQ(hipMemset(o.buffer.get(), 0xFF, o.size), hipSuccess);
        ASSERT_EQ(hipMemset(lse.buffer.get(), 0xFF, lse.size), hipSuccess);
    }

    Sized q;
    Sized o;
    Sized k;
    Sized v;
    Sized lse;
    Sized table;
    Sized tableV;
    Sized seqLenKv;
    Sized seqLenQ;
    Sized offsets;
    Sized sinks;
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
    const bool ragged = paged.form == QueryForm::RAGGED;
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
                // A dense Q's padding past seq_len_q is written as zeros; a ragged Q's rows
                // past it belong to no sequence and keep O's NaN.
                bool ok = false;
                double expected = 0.0;
                const char* note = "";
                if(live)
                {
                    expected = want[base + i];
                    ok = std::isfinite(value) && std::abs(value - expected) <= tolerance;
                }
                else if(ragged)
                {
                    expected = std::numeric_limits<double>::quiet_NaN();
                    ok = std::isnan(value);
                    note = " (owned by no sequence, must stay unwritten)";
                }
                else
                {
                    ok = value == 0.0;
                    note = " (padding past seq_len_q, must be 0)";
                }
                if(!ok && mismatches++ < 8)
                {
                    ADD_FAILURE() << phase << ": sequence " << b << " row " << r << " element " << i
                                  << note << ": got " << value << ", want " << expected;
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
        const auto bi = static_cast<size_t>(b);
        const int64_t span = int64_t{batch.seqLenQ[bi]} + batch.gaps[bi];
        for(int64_t h = 0; h < paged.queryHeads; ++h)
        {
            // Padding rows past seq_len_q have LSE = -inf, as rows with no visible key
            // and no sink do.
            for(int64_t r = 0; r < span; ++r)
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
TEST_P(IntegrationGpuGfx1151WmmaAttentionPaged, MatchesTheHipdnnReferenceAndReplays)
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
