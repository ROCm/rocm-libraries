// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA) \
    && (defined(HKP_DETERMINISM_WMMA_PREFILL) || defined(HKP_DETERMINISM_WMMA_DECODE))

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismWmma.cpp
 * @brief The determinism rows of the RDNA rocKE WMMA engines hipkernel:Gfx1151WmmaAttention
 *        (prefill and paged) and hipkernel:Gfx1151WmmaAttentionDecode. CMake defines
 *        HKP_DETERMINISM_WMMA_PREFILL and HKP_DETERMINISM_WMMA_DECODE for each engine the
 *        product root offers, and each table below exists only under its definition.
 *
 * The catalogs ship generic code objects (gfx11-generic for gfx1100..gfx1153, gfx12-generic for
 * gfx1200 and gfx1201), so every row runs on gfx1100 (gfx11 objects) and gfx1201 (gfx12
 * objects); the two families differ for the wide-Q buckets and the paged head-96 objects, and a
 * row reaches both.
 *
 * What the rows have to reach (a source read of the two kernels): neither kernel has an atomic that
 * combines values, a scheduling-dependent reduction order or a second writer of an output
 * element. Prefill and paged prefill are one launch with one CTA per (head, q-tile, batch) and
 * a sequential in-CTA loop over the key tiles; the wave-pair buckets (Q or V wider than 128 on
 * gfx11) exchange partial scores through LDS between barriers. Decode is two launches: the
 * split-KV kernel writes one partial (O, LSE) per split and the merge kernel adds the splits
 * in index order; the D256 object splits the O columns over dv_split workgroups (2 on gfx11,
 * 4 on gfx12), which all write the same split LSE. The fused-merge arrival counter is never
 * reached: the native launches the merge kernel with WS_CNT null. Rows therefore have to reach
 * every distinct object family (bucket, dtype, bias kind, page size, query form, sinks) and
 * every launch structure (one launch, decode with one split, decode and merge with 2 to 64
 * splits, a split count not divisible by the merge unroll of 8, more than 8 splits, several
 * row tiles, dv_split slices).
 *
 * The routing is fixed by the device, not by the run: the engine takes a dense graph on the
 * prefill path or the decode path by rdnaAttentionUseDecode, a function of the problem and the
 * CU count, and the decode split count is min(keys / (4 waves x 4 tiles x 32 keys), WGP x 2 /
 * work groups, 64) (decodePlanFor, usesDecode). The rows
 * keep both far from their thresholds so every RDNA board of the two families routes them as
 * written: a prefill row has Sq 512 and 2 x 8 heads, 64 CTAs, so 4 x 64 CTAs exceed any RDNA
 * board's CU count (the rule declines decode when 4 x CTAs >= CUs); a decode row has Sq 1 to 4, so
 * both families route to decode (gfx12: Sq <= 32; gfx11: g x Sq < 512). A decode row with Skv 8192
 * has 256 key tiles, so by-work gives 16 splits and by-device (WGP x 2 / 2 to 8 work groups) gives
 * at least 2 on any board with 8 WGPs or more; the exact count depends on the board's CU
 * count, a property of the device and not of the run.
 *
 * Admission evidence, per row group (Gfx1151WmmaAttentionNative.cpp unless noted; CAT is the
 * same directory's Gfx1151WmmaAttentionCatalog.hpp; YAML is projects/hipdnn/tools/
 * IngestorGenerator/configs/gfx1151_wmma_attention.yaml and gfx1151_wmma_attention_decode.yaml):
 *  - every row: one SDPA node (graphMatchesFor); 16-bit Q/K/V/O with well-formed strides
 *    (isWellFormedOperand); head widths multiples of 8 up to 256 (isServedHeadDim); compute
 *    type FLOAT, no dropout, no FP8, no alibi (unservedFeature, bindCommon); mask: right bound
 *    0 or unbounded, left bound optional (a sliding window; the decode table below adds one
 *    windowed row) (maskFor, bindCommon); scale 1 / sqrt(D).
 *  - dense prefill rows (D1): the dense branch of graphMatchesFor: no seq_len, padding_mask or
 *    sink, rows aligned to 8 elements (hasAlignedRows), LSE FLOAT [B, Hq, Sq, 1] (lseFor), bias
 *    FLOAT or Q dtype [B|1, Hq|1, Sq|1, Skv] with unit key stride (biasFor); kernel match
 *    (prefillKernelMatches): dtype and bias kind equal, widths no wider than the object's
 *    bucket, and the dense graph is not one the dispatcher routes to decode (usesDecode), here
 *    guaranteed by Sq 512 above; the tightest bucket scores highest (scoreKernel), and the two
 *    knobs head_size and head_size_v pin the bucket to the graph's own widths (shipped buckets
 *    64/64, 96/96, 128/128, 256/256, 192/128 and 256/128, YAML shapes). The 192/128 and 256/128
 *    objects ship on both families (the wave-pair form on gfx11, the unsplit form on gfx12,
 *    YAML variants).
 *  - paged rows (P1): gfx1151WmmaAttentionPagedGraphMatches: page tables and seq_len_q,
 *    seq_len_kv present and device INT32, page a power of two in the catalog's set {16, 32, 64}
 *    (CAT), table width ceil(max_seq_len_kv / page) (frontend SdpaFwdNode.hpp), K and V
 *    containers of one block size and page count, a shared or a separate page_table_v of one
 *    width and row stride, ragged Q through token-unit offsets that Q and O share, no stats
 *    with a ragged Q, FLOAT sink with a unit head stride (sinkFor). The kernel match bakes the
 *    page size, the query form and sinks into the object. The shipped objects: pages 16, 32, 64
 *    for buckets 64, 128, 256; bucket 96 at page 16 on both families and at pages 32 and 64 as a
 *    distinct block_m 64 object on gfx11 and block_m 128 on gfx12 (YAML paged variants).
 *  - decode rows (C1): decodeKernelMatches and usesDecode: the dispatcher's decode rule and the
 *    bucket of decodeBucket (64/128/256 for the widths of the rows); the decode objects ship
 *    for buckets 64, 128 and 256 with bias none, f32 and q (decode YAML). A window narrows
 *    decodeKvRange's visible keys, which changes decodePlanFor's numSplits (below).
 *
 * Limits of the table, each a fact of this tree and not a choice:
 *  - A sink_token in the Q data type is rejected by the frontend (validateSinkToken,
 *    SdpaFwdNode.hpp: sink must be FLOAT), so the Q-dtype sink objects (sink_is_f32 = 0) are
 *    unreachable through the frontend; every sink row uses the FLOAT sink.
 *  - The harness binds K and V as two tensors, so the interleaved K|V rows give K and V the
 *    interleaved strides (row of 2 x D) over two buffers; the launch arguments, and so the
 *    code path, are the ones of one buffer holding both halves (the native reads the K and V
 *    strides separately, runtimeArguments in Gfx1151WmmaAttentionNative.cpp).
 *  - No row has a sequence with no key, so no LSE holds infinity.
 *  - hipkernel:Gfx1151WmmaAttentionDecode declines every paged graph
 *    (gfx1151WmmaAttentionDecodeGraphMatches, Gfx1151WmmaAttentionNative.cpp) and has no sink
 *    objects, so neither exists for it.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::integration::FrontendSdpaSpec;
using hipdnn_frontend::DataType;
namespace sdpa_detail = hip_kernel_provider::sdpa_determinism::detail;

constexpr DataType BF16 = DataType::BFLOAT16;
constexpr DataType FP16 = DataType::HALF;

/// The devices the rows run on: one gfx11 and one gfx12 device, which between them load both
/// code-object families.
std::vector<std::string> runArchs()
{
    return {"gfx1100", "gfx1201"};
}

std::string dtypeToken(DataType dataType)
{
    return dataType == FP16 ? "Fp16" : "Bf16";
}

/// One dense forward graph: Q [B, Hq, Sq, D] and K, V [B, Hkv, Skv, D(v)], all BSHD.
struct DenseShape
{
    DataType dataType;
    int64_t headSize;
    int64_t valueHeadSize;
    /// Right bound 0 anchored at the bottom-right corner; unbounded when false.
    bool causal;
    bool withStats;
    /// An additive bias [B, Hq, Sq, Skv] in this type.
    std::optional<DataType> bias;
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    int64_t seqQ;
    int64_t seqKv;
    /// A causal window of this many keys (left_bound, the frontend's band-inclusive count),
    /// bottom-right anchored; none when unset.
    std::optional<int64_t> window;
};

std::string denseToken(const DenseShape& shape)
{
    std::string token = dtypeToken(shape.dataType);
    token += "Hd" + std::to_string(shape.headSize);
    if(shape.valueHeadSize != shape.headSize)
    {
        token += "V" + std::to_string(shape.valueHeadSize);
    }
    token += shape.causal ? "Causal" : "NoMask";
    if(shape.window.has_value())
    {
        token += "Window" + std::to_string(*shape.window);
    }
    if(shape.withStats)
    {
        token += "Lse";
    }
    if(shape.bias.has_value())
    {
        token += *shape.bias == DataType::FLOAT ? "BiasF32" : "BiasQ";
    }
    token += "Hq" + std::to_string(shape.queryHeads) + "Kv" + std::to_string(shape.kvHeads) + "B"
             + std::to_string(shape.batch) + "S" + std::to_string(shape.seqQ) + "x"
             + std::to_string(shape.seqKv);
    return token;
}

FrontendSdpaSpec denseSpec(const std::string& name, const DenseShape& shape)
{
    FrontendSdpaSpec spec;
    spec.name = name;
    spec.dataType = shape.dataType;
    spec.batch = shape.batch;
    spec.queryHeads = shape.queryHeads;
    spec.kvHeads = shape.kvHeads;
    spec.seqQ = shape.seqQ;
    spec.seqKv = shape.seqKv;
    spec.headSize = shape.headSize;
    if(shape.valueHeadSize != shape.headSize)
    {
        spec.valueHeadSize = shape.valueHeadSize;
    }
    spec.leftBound.reset();
    spec.rightBound.reset();
    if(shape.causal)
    {
        spec.rightBound = 0;
    }
    if(shape.window.has_value())
    {
        spec.leftBound = shape.window;
    }
    spec.withStats = shape.withStats;
    spec.biasDataType = shape.bias;
    return spec;
}

void addDense(std::vector<DeterminismRow>& rows,
              const char* engine,
              KnobClass knobClass,
              KnobList knobs,
              const DenseShape& shape)
{
    const std::string token = denseToken(shape);
    rows.push_back(forwardRow(RowMeta{engine,
                                      token,
                                      runArchs(),
                                      Sku::ANY,
                                      Disposition::PROVEN,
                                      knobClass,
                                      std::move(knobs),
                                      false},
                              denseSpec(token, shape)));
}

#ifdef HKP_DETERMINISM_WMMA_PREFILL

constexpr const char* PREFILL_ENGINE = "hipkernel:Gfx1151WmmaAttention";

/// The query form of a paged graph: a batch-strided Q whose live rows per sequence are
/// seq_len_q with padding rows after them, the same with one row per sequence (vLLM's uniform
/// decode), or a ragged Q packed through token-unit offsets.
enum class PagedForm
{
    DENSE_PADDED,
    DENSE_DECODE,
    RAGGED,
};

struct PagedRow
{
    DataType dataType;
    int64_t headSize;
    int64_t pageSize;
    PagedForm form;
    /// K and V the two halves of one cache (rows of 2 x D), or two planes (rows of D).
    bool interleaved;
    /// V's pages through their own page_table_v, in another order than K's.
    bool separateVTable;
    /// A FLOAT attention sink per query head.
    bool sink;
    /// Binds the LSE; a ragged Q has none (the paged kernels write no stats for it).
    bool withStats;
};

constexpr int64_t PAGE_TABLE_UID = 21;
constexpr int64_t PAGE_TABLE_V_UID = 22;
constexpr int32_t UNUSED_BLOCK = std::numeric_limits<int32_t>::max();
/// ragged_offset[B] is neither read nor validated with seq_len_q bound (RFC 0014).
constexpr int32_t RAGGED_OFFSET_END_POISON = -7;

/// The sequences of a paged row: per sequence the live query rows, the rows after them
/// (padding of a dense Q, rows of the offset block that belong to no sequence for a ragged Q)
/// and the keys. Every live row sees at least one key (keys >= live rows, bottom-right causal),
/// so no LSE holds infinity, and sequence 0's last row sees many keys, so the sensitivity
/// element moves its output.
struct PagedGeometry
{
    int64_t batch;
    int64_t queryHeads;
    int64_t kvHeads;
    std::vector<int32_t> seqLenQ;
    std::vector<int32_t> gaps;
    std::vector<int32_t> seqLenKv;
};

PagedGeometry geometryOf(const PagedRow& row)
{
    const int64_t kvHeads = 2;
    const int64_t queryHeads = row.headSize == 256 ? 4 : 8;
    if(row.form == PagedForm::DENSE_PADDED)
    {
        return {3, queryHeads, kvHeads, {20, 7, 1}, {4, 17, 23}, {150, 70, 33}};
    }
    if(row.form == PagedForm::DENSE_DECODE)
    {
        return {3, queryHeads, kvHeads, {1, 1, 1}, {0, 0, 0}, {150, 70, 33}};
    }
    // Sequence 0 spans two q-tiles at block_m 128 and three at block_m 64.
    return {3, queryHeads, kvHeads, {130, 9, 1}, {0, 2, 3}, {300, 90, 50}};
}

std::string pagedToken(const PagedRow& row)
{
    std::string token = "Paged" + dtypeToken(row.dataType) + "Hd" + std::to_string(row.headSize)
                        + "P" + std::to_string(row.pageSize);
    if(row.form == PagedForm::DENSE_PADDED)
    {
        token += "DensePad";
    }
    else if(row.form == PagedForm::DENSE_DECODE)
    {
        token += "DenseOneRow";
    }
    else
    {
        token += "Ragged";
    }
    token += row.interleaved ? "Interleaved" : "Planes";
    if(row.separateVTable)
    {
        token += "TableV";
    }
    if(row.sink)
    {
        token += "Sink";
    }
    if(row.withStats)
    {
        token += "Lse";
    }
    return token;
}

std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    makeTensor(int64_t uid,
               const std::string& name,
               DataType dataType,
               const std::vector<int64_t>& dims,
               const std::vector<int64_t>& strides)
{
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_uid(uid).set_name(name).set_data_type(dataType).set_dim(dims).set_stride(strides);
    return tensor;
}

std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    lengthVector(int64_t uid, const std::string& name, int64_t count)
{
    return makeTensor(uid, name, DataType::INT32, {count, 1, 1, 1}, {1, 1, 1, 1});
}

/// A permutation of [0, count): Fisher-Yates over the raw output of a seeded mt19937, which the
/// standard specifies, so one build always lays the pages out the same way.
std::vector<int32_t> shuffledBlocks(int64_t count, uint32_t seed)
{
    std::vector<int32_t> order(static_cast<size_t>(count));
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    for(size_t i = order.size(); i > 1; --i)
    {
        std::swap(order[i - 1], order[static_cast<size_t>(rng() % i)]);
    }
    return order;
}

/// The cuDNN paged-K/V SDPA graph of @p row (the builder of IntegrationGpuGfx1151WmmaAttention
/// Paged.cpp in minimal form), the inputs the graph needs fixed (lengths, page tables, ragged
/// offsets) and the elements of O and LSE a launch writes. K, V and Q are bounded random
/// values, the sink a bounded logit.
PreparedGraph preparePaged(const PagedRow& row)
{
    using namespace hip_kernel_provider::kernel_ingestor_engine::integration::frontend_sdpa_detail;
    const auto geometry = geometryOf(row);
    const auto batchSize = static_cast<size_t>(geometry.batch);
    const int64_t headSize = row.headSize;
    const int64_t token = geometry.queryHeads * headSize;
    const bool ragged = row.form == PagedForm::RAGGED;
    const std::string name = pagedToken(row);

    int64_t queryRows = 0;
    int64_t longestKeys = 0;
    int64_t usedBlocks = 0;
    for(size_t b = 0; b < batchSize; ++b)
    {
        queryRows = std::max<int64_t>(queryRows, int64_t{geometry.seqLenQ[b]} + geometry.gaps[b]);
        longestKeys = std::max<int64_t>(longestKeys, geometry.seqLenKv[b]);
        usedBlocks += (geometry.seqLenKv[b] + row.pageSize - 1) / row.pageSize;
    }
    // One spare table entry per row and a few blocks no sequence references.
    const int64_t blocksPerSequence = (longestKeys + row.pageSize - 1) / row.pageSize + 1;
    const int64_t numBlocks = usedBlocks + 5;

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name(name)
        .set_io_data_type(row.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    const std::vector<int64_t> qDims{geometry.batch, geometry.queryHeads, queryRows, headSize};
    const std::vector<int64_t> qStrides{queryRows * token, headSize, token, 1};
    auto q = makeTensor(Q_UID, "Q", row.dataType, qDims, qStrides);
    const int64_t cacheRow = row.interleaved ? 2 * headSize : headSize;
    const std::vector<int64_t> cacheDims{numBlocks, geometry.kvHeads, row.pageSize, headSize};
    const std::vector<int64_t> cacheStrides{
        geometry.kvHeads * row.pageSize * cacheRow, row.pageSize * cacheRow, cacheRow, 1};
    auto k = makeTensor(K_UID, "K", row.dataType, cacheDims, cacheStrides);
    auto v = makeTensor(V_UID, "V", row.dataType, cacheDims, cacheStrides);
    const std::vector<int64_t> tableDims{geometry.batch, 1, blocksPerSequence, 1};
    const std::vector<int64_t> tableStrides{blocksPerSequence, blocksPerSequence, 1, 1};
    auto table = makeTensor(PAGE_TABLE_UID, "page_table", DataType::INT32, tableDims, tableStrides);
    auto tableV
        = row.separateVTable
              ? makeTensor(
                    PAGE_TABLE_V_UID, "page_table_v", DataType::INT32, tableDims, tableStrides)
              : table;

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> offsets;
    if(ragged)
    {
        offsets = lengthVector(OFFSETS_Q_UID, "q_offsets", geometry.batch + 1);
        q->set_ragged_offset(offsets).set_ragged_offset_multiplier(token);
    }

    hipdnn_frontend::graph::SdpaAttributes attributes;
    attributes.set_name(name)
        .set_attn_scale(1.0F / std::sqrt(static_cast<float>(headSize)))
        .set_diagonal_alignment(hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT)
        .set_diagonal_band_right_bound(0)
        .set_paged_attention_k_table(table)
        .set_paged_attention_v_table(tableV)
        .set_seq_len_q(lengthVector(SEQ_LEN_Q_UID, "seq_len_q", geometry.batch))
        .set_seq_len_kv(lengthVector(SEQ_LEN_KV_UID, "seq_len_kv", geometry.batch))
        .set_paged_attention_max_seq_len_kv(static_cast<int32_t>(blocksPerSequence * row.pageSize))
        .set_padding_mask(true)
        .set_generate_stats(row.withStats);
    if(row.sink)
    {
        attributes.set_sink_token(makeTensor(SINK_UID,
                                             "sink",
                                             DataType::FLOAT,
                                             {1, geometry.queryHeads, 1, 1},
                                             {geometry.queryHeads, 1, 1, 1}));
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    auto& o = outputs[0];
    o->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(row.dataType)
        .set_dim(qDims)
        .set_stride(qStrides);
    if(ragged)
    {
        o->set_ragged_offset(offsets).set_ragged_offset_multiplier(token);
    }
    if(row.withStats)
    {
        outputs[1]
            ->set_uid(STATS_UID)
            .set_name("LSE")
            .set_output(true)
            .set_data_type(DataType::FLOAT)
            .set_dim({geometry.batch, geometry.queryHeads, queryRows, 1})
            .set_stride({geometry.queryHeads * queryRows, queryRows, 1, 1});
    }

    // Ragged origins in reverse sequence order, as the engine's own paged suite lays them out.
    std::vector<int32_t> qOffsets(batchSize + 1, RAGGED_OFFSET_END_POISON);
    int32_t origin = 0;
    for(size_t b = batchSize; b-- > 0;)
    {
        qOffsets[b] = origin;
        origin += geometry.seqLenQ[b] + geometry.gaps[b];
    }

    // The page tables: each sequence's pages are blocks of one shuffled order, V's of another
    // when it has its own table; entries past a sequence's pages name no block.
    const auto order = shuffledBlocks(numBlocks, 0x5eed01U);
    const auto orderV = row.separateVTable ? shuffledBlocks(numBlocks, 0x5eed02U) : order;
    std::vector<int32_t> pageTable(batchSize * static_cast<size_t>(blocksPerSequence),
                                   UNUSED_BLOCK);
    std::vector<int32_t> pageTableV = pageTable;
    size_t next = 0;
    for(size_t b = 0; b < batchSize; ++b)
    {
        const int64_t pages = (geometry.seqLenKv[b] + row.pageSize - 1) / row.pageSize;
        for(int64_t page = 0; page < pages; ++page)
        {
            const auto entry
                = b * static_cast<size_t>(blocksPerSequence) + static_cast<size_t>(page);
            pageTable[entry] = order[next];
            pageTableV[entry] = orderV[next];
            ++next;
        }
    }

    PreparedGraph prepared;
    prepared.graph = graph;
    const auto liveRow = [&geometry](const std::vector<int64_t>& index) {
        return index[2] < geometry.seqLenQ[static_cast<size_t>(index[0])];
    };
    OutputSpec oSpec;
    oSpec.name = "O";
    oSpec.type = elementTypeOf(row.dataType);
    oSpec.live = ragged ? packedRuns(std::vector<int32_t>(qOffsets.begin(), qOffsets.end() - 1),
                                     geometry.seqLenQ,
                                     static_cast<size_t>(token))
                        : logicalRuns(qDims, qStrides, liveRow);
    oSpec.allowInfinity = false;
    prepared.outputs[O_UID] = std::move(oSpec);
    if(row.withStats)
    {
        OutputSpec lseSpec;
        lseSpec.name = "LSE";
        lseSpec.type = ElementType::FLOAT32;
        lseSpec.live = logicalRuns({geometry.batch, geometry.queryHeads, queryRows, 1},
                                   {geometry.queryHeads * queryRows, queryRows, 1, 1},
                                   liveRow);
        lseSpec.allowInfinity = false;
        prepared.outputs[STATS_UID] = std::move(lseSpec);
    }

    prepared.fixedInputs[SEQ_LEN_Q_UID] = sdpa_detail::bytesOf(geometry.seqLenQ);
    prepared.fixedInputs[SEQ_LEN_KV_UID] = sdpa_detail::bytesOf(geometry.seqLenKv);
    prepared.fixedInputs[PAGE_TABLE_UID] = sdpa_detail::bytesOf(pageTable);
    if(row.separateVTable)
    {
        prepared.fixedInputs[PAGE_TABLE_V_UID] = sdpa_detail::bytesOf(pageTableV);
    }
    if(ragged)
    {
        prepared.fixedInputs[OFFSETS_Q_UID] = sdpa_detail::bytesOf(qOffsets);
        prepared.extraInputs = {offsets};
    }

    // Sequence 0's last live query row, head 0, element 0.
    const int64_t firstRow = ragged ? int64_t{qOffsets[0]} : 0;
    prepared.sensitivityUid = Q_UID;
    prepared.sensitivityElement = static_cast<size_t>((firstRow + geometry.seqLenQ[0] - 1) * token);
    return prepared;
}

DeterminismRow pagedRow(const PagedRow& row)
{
    DeterminismRow result;
    result.meta = RowMeta{PREFILL_ENGINE,
                          pagedToken(row),
                          runArchs(),
                          Sku::ANY,
                          Disposition::PROVEN,
                          KnobClass::TILE_SCHEDULE,
                          {{"head_size", row.headSize}, {"head_size_v", row.headSize}},
                          false};
    result.name = sdpa_detail::rowName(result.meta);
    result.direction = Direction::FORWARD;
    result.build = [row](const DeterminismRow& /*self*/) { return preparePaged(row); };
    return result;
}

#endif // HKP_DETERMINISM_WMMA_PREFILL

std::vector<DeterminismRow> wmmaRows()
{
    std::vector<DeterminismRow> rows;
#ifdef HKP_DETERMINISM_WMMA_PREFILL
    {
        // Dense prefill, one launch, one CTA per (head, q-tile, batch). One row per shipped
        // (head_size, head_size_v) bucket x dtype x {causal, no mask}: 6 x 2 x 2 objects of
        // the "causal" class, which serves every mask by runtime bounds, so a causal row
        // (bottom-right band, GQA 8 / 2, LSE bound) and an unmasked row (MHA, no LSE) reach
        // the two mask arguments and the two LSE paths of every object. The 256-wide buckets
        // and 192/128 are the wave-pair classes on gfx11 (the LDS score exchange) and the
        // unsplit class on gfx12. Sq = Skv = 512 and B = 2 x Hq = 8 keep the rows on the
        // prefill path of every board (see the file comment). The pin is the bucket
        // (head_size, head_size_v), the knobs of the engine's UED; both knobs change the
        // compiled object and neither changes the launch set or a combine.
        constexpr std::array<std::array<int64_t, 2>, 6> BUCKETS{
            {{64, 64}, {96, 96}, {128, 128}, {256, 256}, {192, 128}, {256, 128}}};
        for(const auto dataType : {BF16, FP16})
        {
            for(const auto& bucket : BUCKETS)
            {
                for(const bool causal : {true, false})
                {
                    const DenseShape shape{dataType,
                                           bucket[0],
                                           bucket[1],
                                           causal,
                                           causal,
                                           std::nullopt,
                                           2,
                                           8,
                                           causal ? 2 : 8,
                                           512,
                                           512,
                                           std::nullopt};
                    addDense(rows,
                             PREFILL_ENGINE,
                             KnobClass::TILE_SCHEDULE,
                             {{"head_size", bucket[0]}, {"head_size_v", bucket[1]}},
                             shape);
                }
            }
        }
        // The bias objects (bias f32 and q, each dtype): the same launch with the bias
        // pointer and its three strides as arguments.
        for(const auto dataType : {BF16, FP16})
        {
            for(const auto biasType : {DataType::FLOAT, dataType})
            {
                const DenseShape shape{
                    dataType, 128, 128, false, false, biasType, 2, 8, 8, 512, 512, std::nullopt};
                addDense(rows,
                         PREFILL_ENGINE,
                         KnobClass::TILE_SCHEDULE,
                         {{"head_size", 128}, {"head_size_v", 128}},
                         shape);
            }
        }
    }
    {
        // Paged prefill, one launch, each page size x query form x dtype x bucket x sinks
        // object family; the table is a cover, not the product (no launch structure differs:
        // read-only pages and tables, one CTA per output tile, the sink only seeds the
        // softmax state). Per page size 16, 32 and 64: head 128 in each query form with
        // interleaved and separate K|V planes, a shared and a separate page_table_v; heads 64,
        // 96 (a distinct object at pages 32 and 64 on each family) and 256 (block_m 64) in each
        // query form; sinks on for each bucket. The DENSE_PADDED form has padding rows after
        // each sequence's live rows (written O = 0, LSE = -inf, not compared), DENSE_DECODE one
        // row per sequence, RAGGED a multi-tile sequence beside short ones.
        const auto add = [&rows](DataType dataType,
                                 int64_t headSize,
                                 int64_t pageSize,
                                 PagedForm form,
                                 bool interleaved,
                                 bool separateVTable,
                                 bool sink,
                                 bool withStats) {
            rows.push_back(pagedRow(PagedRow{
                dataType, headSize, pageSize, form, interleaved, separateVTable, sink, withStats}));
        };
        constexpr auto PAD = PagedForm::DENSE_PADDED;
        constexpr auto ONE = PagedForm::DENSE_DECODE;
        constexpr auto RAG = PagedForm::RAGGED;
        for(const int64_t page : {16, 32, 64})
        {
            add(BF16, 128, page, PAD, true, false, false, true);
            add(FP16, 128, page, PAD, false, true, false, true);
            add(BF16, 128, page, RAG, false, false, false, false);
            add(FP16, 128, page, RAG, true, true, false, false);
        }
        add(BF16, 64, 16, ONE, true, false, false, true);
        add(FP16, 64, 32, RAG, false, false, false, false);
        add(FP16, 64, 64, PAD, true, true, false, true);
        add(BF16, 96, 16, PAD, true, false, false, true);
        add(BF16, 96, 32, RAG, false, true, false, false);
        add(FP16, 96, 64, ONE, true, false, false, true);
        add(BF16, 256, 16, RAG, true, false, false, false);
        add(FP16, 256, 32, PAD, false, false, false, true);
        add(BF16, 256, 64, ONE, true, true, false, true);
        // Sinks (FLOAT; the frontend takes no other type).
        add(BF16, 64, 16, PAD, true, false, true, true);
        add(FP16, 64, 32, RAG, false, true, true, false);
        add(FP16, 64, 64, ONE, true, true, true, true);
        add(BF16, 128, 16, RAG, true, false, true, false);
        add(FP16, 128, 32, PAD, false, false, true, true);
        add(BF16, 128, 64, ONE, false, true, true, true);
        add(BF16, 96, 16, PAD, true, false, true, true);
        add(FP16, 96, 32, RAG, false, false, true, false);
        add(BF16, 96, 64, PAD, true, true, true, true);
        add(BF16, 256, 16, PAD, true, false, true, true);
        add(FP16, 256, 32, RAG, true, true, true, false);
        add(FP16, 256, 64, ONE, false, false, true, true);
    }
#endif
#ifdef HKP_DETERMINISM_WMMA_DECODE
    {
        // Dense decode: the split-KV kernel writes one (O, LSE) partial per split into the
        // workspace and the merge kernel adds them in split order, so every row has Sq <= 4 and
        // routes to decode on both families; all but the two Skv 512 rows take at least 2 splits. Skv 8192 is 256 key
        // tiles, so by-work gives 16 splits (a second group of the merge's unroll of 8) and
        // by-device at least 2 on a board of 8 WGPs; Skv 5120 gives 10 (not a multiple of 8);
        // Skv 2048 gives 4. The D256 object slices the O columns over dv_split workgroups (2 on
        // gfx11, 4 on gfx12) that write the same split LSE; its pin is the one STRUCTURE knob.
        // Sq 4 with 8 query heads per K/V head packs two 16-row tiles per head. A decode row
        // pins the bucket (head_size); bias f32 and q are distinct objects.
        const char* const engine = "hipkernel:Gfx1151WmmaAttentionDecode";
        const auto add = [&rows, engine](const DenseShape& shape) {
            const bool slicedO = shape.headSize == 256;
            addDense(rows,
                     engine,
                     slicedO ? KnobClass::STRUCTURE : KnobClass::TILE_SCHEDULE,
                     {{"head_size", shape.headSize}},
                     shape);
        };
        for(const int64_t headSize : {64, 128, 256})
        {
            add({BF16,
                 headSize,
                 headSize,
                 true,
                 true,
                 std::nullopt,
                 1,
                 8,
                 2,
                 1,
                 8192,
                 std::nullopt});
            add({FP16,
                 headSize,
                 headSize,
                 false,
                 false,
                 std::nullopt,
                 1,
                 8,
                 2,
                 1,
                 8192,
                 std::nullopt});
        }
        add({BF16, 128, 128, true, false, std::nullopt, 1, 8, 2, 1, 5120, std::nullopt});
        add({FP16, 128, 128, false, false, std::nullopt, 1, 8, 2, 1, 2048, std::nullopt});
        // One split: Skv 512 is 16 key tiles, so by-work gives 1 split on every board (a
        // decode launch that stores O and LSE directly, with no workspace and no merge launch).
        add({BF16, 128, 128, true, true, std::nullopt, 1, 8, 2, 1, 512, std::nullopt});
        add({FP16, 128, 128, false, false, std::nullopt, 1, 8, 2, 1, 512, std::nullopt});
        add({BF16, 128, 128, true, true, std::nullopt, 1, 8, 1, 4, 8192, std::nullopt});
        add({FP16, 256, 256, true, false, std::nullopt, 1, 8, 1, 4, 8192, std::nullopt});
        add({BF16, 128, 128, false, false, DataType::FLOAT, 1, 8, 2, 1, 8192, std::nullopt});
        add({FP16, 128, 128, false, false, DataType::HALF, 1, 8, 2, 1, 8192, std::nullopt});
        // A sliding window on the Skv 8192 causal shape above. The frontend's left_bound is
        // band-inclusive; the native's window_left (decodeKvRange) is left_bound - 1
        // (Gfx1151WmmaAttentionNative.cpp maskFor), so left_bound 2048 gives window_left 2047:
        // decodeKvRange keeps only the last 2048 keys visible (lo = 8191 - 2047 = 6144, hi =
        // 8192), so the tile count drops from 256 to 64 and decodePlanFor's by-work split count
        // drops from 16 (the unwindowed row above) to 4; every row still sees keys. The window
        // changes the host-computed launch set (numSplits), not the object, so this row pins
        // STRUCTURE rather than this block's usual TILE_SCHEDULE.
        addDense(rows,
                 engine,
                 KnobClass::STRUCTURE,
                 {{"head_size", 128}},
                 {BF16, 128, 128, true, true, std::nullopt, 1, 8, 2, 1, 8192, 2048});
    }
#endif
    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(wmmaRows());

} // namespace

#endif
