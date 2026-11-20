// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA) \
    && (defined(HKP_DETERMINISM_FLYDSL_BWD_MERGED) || defined(HKP_DETERMINISM_FLYDSL_BWD_SPLIT))

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "SdpaDeterminismHarness.hpp"

/**
 * @file IntegrationGpuSdpaDeterminismFlydslBwd.cpp
 * @brief The determinism rows of the FlyDSL backward engines hipkernel:flydsl_sdpa_bwd_merged
 *        and _bwd_split. CMake defines HKP_DETERMINISM_FLYDSL_BWD_MERGED and
 *        HKP_DETERMINISM_FLYDSL_BWD_SPLIT for each engine the product root ships FlyDSL content
 *        for, and each table below exists only under its definition.
 *
 * One row per launch-structure class and dtype (plan section 3.1, trace-TraceFlydslBwd). The
 * classes: the merged one-launch kernel (dK/dV and dQ roles in one grid); the split prep, dkdv
 * and dq launches; the GQA group fold (and the gqa_pack objects); dkv_shards; padded head dims
 * (an `padded` object, compared on live elements only, HKP-H34); the dBias in-place store and
 * the broadcast dBias reduce launch; the padding_mask tail-zero launch; Philox dropout (seed and
 * offset as INT64 device tensors and passed by value); causal and not; bf16, f16 and fp32; the
 * dkdv_u2 pair (V heads that differ from K's).
 *
 * Which graphs an engine admits is read from the shared matcher (FlydslSdpaBwdCommon.hpp), the
 * engine matchers and the objects each target ships (the SOURCE.md beside each descriptor
 * folder under descriptors/FlyDSL/sdpa_bwd_*):
 *  - Merged: matchSdpaBackward with default options (FlydslSdpaBwdMergedNative.cpp:288), so
 *    16-bit only (FlydslSdpaBwdCommon.hpp:726-735, acceptFloat off), no f32 bias (:1383-1397) and
 *    V heads equal to K's (:1360); head dims <= 32 (padded tile 32) or Dqk = Dv at 64 to 256 in
 *    steps of 32 (:307-321); dBias needs a store_db object (:461-463); the bias must satisfy the
 *    LDS staging contract (:322-332). gfx950 only.
 *  - Split: SPLIT_MATCH_OPTIONS (FlydslSdpaBwdSplitNative.cpp:467), so fp32, an f32 bias beside
 *    a 16-bit Q and Hv != Hk at D 64 and 128 (:403-451, FlydslSdpaBwdCommon.hpp:131-143);
 *    kernel_match :943-1024; head-dim bands :734-743. gfx950, gfx942, gfx90a and gfx11-generic
 *    (run on gfx1100); gfx942 and gfx90a fp32 reuse the gfx950 fp32 kernels.
 *  - Every graph: bias with a causal mask or padding_mask declines (FlydslSdpaBwdCommon.hpp:
 *    1438-1445), so the bias rows are dense and non-causal; dBias strides are multiples of 8
 *    elements (:1497-1509); a row-broadcast bias is left out (defects IMPL-12, 15643); padding_mask
 *    needs both lengths (:1262-1299); dropout needs seed and offset tensors of one form (:1597-1688);
 *    a head dim off 8 (off 4 for fp32) needs a row pitch of at least ceil8(D) (:894-906).
 *  - No row sets a sink or dSink: the frontend marks the backward sink tokens unsupported
 *    (SdpaBackwardAttributes.hpp:652-664) and the split prep refuses the sink twin
 *    (FlydslSdpaBwdSplitNative.cpp:957-964), so no graph reaches the dsink_reduce kernel.
 *
 * Plain 16-bit rows at head size 64 and 128 use the harness graph, whose O and LSE come from one
 * forward launch. Every other row builds its own graph in this file: O is bounded and finite
 * (it only enters rowsum(O * dO)) and the LSE is computed on the host exactly (hostLogsumexp:
 * the logsumexp over the keys of scale * QK^T plus the bias, under the row's causal mask and
 * padding; dropout leaves it unchanged), so P = exp(S - LSE) stays at most 1 as it does after a
 * forward launch. The forward builder takes 16-bit graphs only, so fp32 rows cannot use it.
 *
 * Rows that pin dkv_shards, gqa_pack or role_order are STRUCTURE (plan 3.1 names them structure classes);
 * rows that pin store_db or padded are TILE_SCHEDULE (they change the stores and the tail
 * handling only). Knobs are pinned only where the value is spelled in an object name of the
 * shipped catalog. The split
 * engine's knobs are per slot, so none is pinned there.
 */

namespace
{

using namespace hip_kernel_provider::sdpa_determinism;
using hip_kernel_provider::kernel_ingestor_engine::testing::SdpaLayout;
using hipdnn_frontend::DataType;
namespace sd = hip_kernel_provider::sdpa_determinism;
namespace fe = hipdnn_frontend;

[[maybe_unused]] constexpr auto BF16 = DataType::BFLOAT16;
[[maybe_unused]] constexpr auto F16 = DataType::HALF;
[[maybe_unused]] constexpr auto F32 = DataType::FLOAT;

/// The uids this file adds to the harness's bwd_uid set.
namespace case_uid
{
inline constexpr int64_t BIAS = 10;
inline constexpr int64_t DBIAS = 11;
inline constexpr int64_t SEQ_LEN_Q = 12;
inline constexpr int64_t SEQ_LEN_KV = 13;
inline constexpr int64_t SEED = 14;
inline constexpr int64_t OFFSET = 15;
} // namespace case_uid

/// Philox dropout with the seed and offset as INT64 [1, 1, 1, 1] tensors on the device or
/// passed by value at execute.
struct DropoutSpec
{
    float probability = 0.2F;
    int64_t seed = 0x5eed1234;
    int64_t offset = 0x40;
    bool onDevice = true;
};

/// An additive bias, packed row-major over its dims, and optionally its gradient.
struct BiasSpec
{
    DataType type;
    std::vector<int64_t> dims;
    bool gradient;
};

/// One backward graph: BSHD Q, K, V, O, dO with a row pitch of the head size rounded up to a
/// 16-byte vector (so an odd head size has a gap behind each row).
struct BwdCase
{
    std::string name;
    DataType dataType = BF16;
    int64_t batch = 2;
    int64_t queryHeads = 4;
    int64_t kvHeads = 4;
    /// V's own head count (Hk != Hv); K's when unset.
    std::optional<int64_t> valueHeads;
    int64_t seqQ = 128;
    int64_t seqKv = 128;
    int64_t headSize = 64;
    /// Causal, bottom-right aligned (right bound 0).
    bool causal = false;
    std::optional<BiasSpec> bias;
    /// Per-batch lengths of both Q and K with padding_mask; dense when empty.
    std::vector<int32_t> seqLens;
    std::optional<DropoutSpec> dropout;
};

[[maybe_unused]] BwdCase graphOf(DataType type, int64_t headSize)
{
    BwdCase graph;
    graph.dataType = type;
    graph.headSize = headSize;
    return graph;
}

[[maybe_unused]] BwdCase withCausal(BwdCase graph)
{
    graph.causal = true;
    return graph;
}

[[maybe_unused]] BwdCase withHeads(BwdCase graph, int64_t queryHeads, int64_t kvHeads)
{
    graph.queryHeads = queryHeads;
    graph.kvHeads = kvHeads;
    return graph;
}

[[maybe_unused]] BwdCase withValueHeads(BwdCase graph, int64_t valueHeads)
{
    graph.valueHeads = valueHeads;
    return graph;
}

[[maybe_unused]] BwdCase withSeq(BwdCase graph, int64_t seqQ, int64_t seqKv)
{
    graph.seqQ = seqQ;
    graph.seqKv = seqKv;
    return graph;
}

[[maybe_unused]] BwdCase withBatchHeads(BwdCase graph, int64_t batch, int64_t queryHeads)
{
    graph.batch = batch;
    graph.queryHeads = queryHeads;
    graph.kvHeads = queryHeads;
    return graph;
}

/// A bias over [B, Hq, Sq, Skv], with batch and/or head extents of 1 where broadcast; with its
/// gradient when @p gradient. Call after the shape is set.
[[maybe_unused]] BwdCase withBias(BwdCase graph,
                                  DataType type,
                                  bool gradient,
                                  bool batchBroadcast = false,
                                  bool headBroadcast = false)
{
    graph.bias = BiasSpec{type,
                          {batchBroadcast ? 1 : graph.batch,
                           headBroadcast ? 1 : graph.queryHeads,
                           graph.seqQ,
                           graph.seqKv},
                          gradient};
    return graph;
}

[[maybe_unused]] BwdCase withLens(BwdCase graph, std::vector<int32_t> lens)
{
    graph.seqLens = std::move(lens);
    return graph;
}

[[maybe_unused]] BwdCase withDropout(BwdCase graph, bool onDevice)
{
    DropoutSpec dropout;
    dropout.onDevice = onDevice;
    graph.dropout = dropout;
    return graph;
}

[[maybe_unused]] std::vector<int64_t> packedStrides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t axis = dims.size(); axis-- > 1;)
    {
        strides[axis - 1] = strides[axis] * dims[axis];
    }
    return strides;
}

/// The value of the element at @p element, of type @p type.
[[maybe_unused]] double decodeElement(ElementType type, const uint8_t* element)
{
    switch(type)
    {
    case ElementType::FLOAT32:
    {
        float value = 0.0F;
        std::memcpy(&value, element, sizeof(value));
        return static_cast<double>(value);
    }
    case ElementType::BFLOAT16:
    {
        uint16_t bits = 0;
        std::memcpy(&bits, element, sizeof(bits));
        const uint32_t widened = static_cast<uint32_t>(bits) << 16;
        float value = 0.0F;
        std::memcpy(&value, &widened, sizeof(value));
        return static_cast<double>(value);
    }
    case ElementType::HALF:
    {
        uint16_t bits = 0;
        std::memcpy(&bits, element, sizeof(bits));
        const int exponent = (bits >> 10) & 0x1F;
        const int mantissa = bits & 0x3FF;
        const double magnitude
            = exponent == 0
                  ? std::ldexp(static_cast<double>(mantissa), -24)
                  : std::ldexp(1.0 + static_cast<double>(mantissa) / 1024.0, exponent - 15);
        return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
    }
    default:
        throw std::invalid_argument("decodeElement: the backward rows have no such element type");
    }
}

/// The LSE the backward graph of @p c needs, from the Q, K and bias the launch will read:
/// log(sum_j exp(scale * q.k + bias)) over the keys row i sees (the causal band and the live
/// keys of padding_mask), in double. Rows past a batch's length get 0 (no kernel reads them).
[[maybe_unused]] std::string hostLogsumexp(const BwdCase& c,
                                           int64_t pitch,
                                           hipdnn_test_sdk::utilities::GraphTensorBundle& bundle)
{
    const auto type = elementTypeOf(c.dataType);
    const size_t size = elementBytes(type);
    const auto* q = static_cast<const uint8_t*>(bundle.getTensor(bwd_uid::Q).rawHostData());
    const auto* k = static_cast<const uint8_t*>(bundle.getTensor(bwd_uid::K).rawHostData());
    auto* lse = static_cast<float*>(bundle.getTensor(bwd_uid::STATS).rawHostData());
    const uint8_t* bias = nullptr;
    std::vector<int64_t> biasStrides;
    ElementType biasType = type;
    if(c.bias.has_value())
    {
        bias = static_cast<const uint8_t*>(bundle.getTensor(case_uid::BIAS).rawHostData());
        biasStrides = packedStrides(c.bias->dims);
        biasType = elementTypeOf(c.bias->type);
    }
    const double scale = 1.0 / std::sqrt(static_cast<double>(c.headSize));
    const int64_t group = c.queryHeads / c.kvHeads;
    const int64_t diagonal = c.seqLens.empty() ? c.seqKv - c.seqQ : 0;
    for(int64_t b = 0; b < c.batch; ++b)
    {
        const int64_t liveQ = c.seqLens.empty() ? c.seqQ : c.seqLens[static_cast<size_t>(b)];
        const int64_t liveKv = c.seqLens.empty() ? c.seqKv : c.seqLens[static_cast<size_t>(b)];
        for(int64_t h = 0; h < c.queryHeads; ++h)
        {
            const int64_t kvHead = h / group;
            for(int64_t i = 0; i < c.seqQ; ++i)
            {
                float& out = lse[(b * c.queryHeads + h) * c.seqQ + i];
                out = 0.0F;
                if(i >= liveQ)
                {
                    continue;
                }
                std::vector<double> scores;
                for(int64_t j = 0; j < liveKv; ++j)
                {
                    if(c.causal && j > i + diagonal)
                    {
                        break;
                    }
                    double dot = 0.0;
                    for(int64_t d = 0; d < c.headSize; ++d)
                    {
                        const auto qAt = static_cast<size_t>(
                            ((b * c.seqQ + i) * c.queryHeads + h) * pitch + d);
                        const auto kAt = static_cast<size_t>(
                            ((b * c.seqKv + j) * c.kvHeads + kvHead) * pitch + d);
                        dot += decodeElement(type, q + qAt * size)
                               * decodeElement(type, k + kAt * size);
                    }
                    double score = dot * scale;
                    if(bias != nullptr)
                    {
                        const auto& dims = c.bias->dims;
                        const int64_t at = (dims[0] == 1 ? 0 : b) * biasStrides[0]
                                           + (dims[1] == 1 ? 0 : h) * biasStrides[1]
                                           + (dims[2] == 1 ? 0 : i) * biasStrides[2] + j;
                        score += decodeElement(
                            biasType, bias + static_cast<size_t>(at) * elementBytes(biasType));
                    }
                    scores.push_back(score);
                }
                if(scores.empty())
                {
                    return "a live query row sees no key, so it has no LSE";
                }
                double largest = scores.front();
                for(const double score : scores)
                {
                    largest = std::max(largest, score);
                }
                double sum = 0.0;
                for(const double score : scores)
                {
                    sum += std::exp(score - largest);
                }
                out = static_cast<float>(largest + std::log(sum));
            }
        }
    }
    bundle.getTensor(bwd_uid::STATS).markHostModified();
    return {};
}

/// The frontend graph of @p graph, its outputs' live elements, its fixed inputs and its
/// sensitivity element. O is a bounded input and the LSE is computed on the host.
[[maybe_unused]] PreparedGraph prepareCase(const BwdCase& graphCase)
{
    const auto& c = graphCase;
    const int64_t valueHeads = c.valueHeads.value_or(c.kvHeads);
    const int64_t vector = c.dataType == DataType::FLOAT ? 4 : 8;
    const int64_t pitch = (c.headSize + vector - 1) / vector * vector;
    const auto tokenMajor = [pitch](int64_t heads, int64_t seq) {
        return std::vector<int64_t>{seq * heads * pitch, pitch, heads * pitch, 1};
    };
    const auto make = [](int64_t uid,
                         const char* name,
                         DataType type,
                         const std::vector<int64_t>& dims,
                         const std::vector<int64_t>& strides) {
        auto tensor = std::make_shared<fe::graph::TensorAttributes>();
        tensor->set_uid(uid).set_name(name).set_dim(dims).set_stride(strides).set_data_type(type);
        return tensor;
    };

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_name(c.name)
        .set_io_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT);

    const std::vector<int64_t> qDims = {c.batch, c.queryHeads, c.seqQ, c.headSize};
    const std::vector<int64_t> kDims = {c.batch, c.kvHeads, c.seqKv, c.headSize};
    const std::vector<int64_t> vDims = {c.batch, valueHeads, c.seqKv, c.headSize};
    const auto qStrides = tokenMajor(c.queryHeads, c.seqQ);
    const auto kStrides = tokenMajor(c.kvHeads, c.seqKv);
    const auto vStrides = tokenMajor(valueHeads, c.seqKv);
    const std::vector<int64_t> statsDims = {c.batch, c.queryHeads, c.seqQ, 1};
    const std::vector<int64_t> statsStrides = {c.queryHeads * c.seqQ, c.seqQ, 1, 1};
    const std::vector<int64_t> scalarDims = {1, 1, 1, 1};

    auto q = make(bwd_uid::Q, "Q", c.dataType, qDims, qStrides);
    auto k = make(bwd_uid::K, "K", c.dataType, kDims, kStrides);
    auto v = make(bwd_uid::V, "V", c.dataType, vDims, vStrides);
    auto o = make(bwd_uid::O, "O", c.dataType, qDims, qStrides);
    auto dO = make(bwd_uid::DO, "dO", c.dataType, qDims, qStrides);
    auto stats = make(bwd_uid::STATS, "LSE", DataType::FLOAT, statsDims, statsStrides);

    fe::graph::SdpaBackwardAttributes attributes;
    attributes.set_name(c.name);
    attributes.set_attn_scale(1.0F / std::sqrt(static_cast<float>(c.headSize)));
    if(c.causal)
    {
        attributes.set_diagonal_band_right_bound(0);
        attributes.set_diagonal_alignment(fe::DiagonalAlignment::BOTTOM_RIGHT);
    }

    PreparedGraph prepared;
    if(c.bias.has_value())
    {
        const auto& biasSpec = *c.bias;
        const auto biasStrides = packedStrides(biasSpec.dims);
        attributes.set_bias(
            make(case_uid::BIAS, "Bias", biasSpec.type, biasSpec.dims, biasStrides));
        if(biasSpec.gradient)
        {
            auto dBias = make(case_uid::DBIAS, "dBias", biasSpec.type, biasSpec.dims, biasStrides);
            dBias->set_output(true);
            attributes.set_dbias(dBias);
            prepared.outputs[case_uid::DBIAS] = {"dBias",
                                                 elementTypeOf(biasSpec.type),
                                                 logicalRuns(biasSpec.dims, biasStrides),
                                                 false};
        }
    }
    if(!c.seqLens.empty())
    {
        const std::vector<int64_t> lenDims = {c.batch, 1, 1, 1};
        attributes.set_seq_len_q(
            make(case_uid::SEQ_LEN_Q, "SeqLenQ", DataType::INT32, lenDims, scalarDims));
        attributes.set_seq_len_kv(
            make(case_uid::SEQ_LEN_KV, "SeqLenKv", DataType::INT32, lenDims, scalarDims));
        attributes.set_padding_mask(true);
        prepared.fixedInputs[case_uid::SEQ_LEN_Q] = sd::detail::bytesOf(c.seqLens);
        prepared.fixedInputs[case_uid::SEQ_LEN_KV] = sd::detail::bytesOf(c.seqLens);
    }
    if(c.dropout.has_value())
    {
        const auto& dropout = *c.dropout;
        attributes.set_dropout(dropout.probability,
                               ingestor::frontend_sdpa_detail::philoxTensor(
                                   case_uid::SEED, "Seed", dropout.seed, dropout.onDevice),
                               ingestor::frontend_sdpa_detail::philoxTensor(
                                   case_uid::OFFSET, "Offset", dropout.offset, dropout.onDevice));
        prepared.fixedInputs[case_uid::SEED] = sd::detail::bytesOf(dropout.seed);
        prepared.fixedInputs[case_uid::OFFSET] = sd::detail::bytesOf(dropout.offset);
    }

    auto [dq, dk, dv] = graph->sdpa_backward(q, k, v, o, dO, stats, std::move(attributes));
    dq->set_uid(bwd_uid::DQ)
        .set_name("dQ")
        .set_output(true)
        .set_data_type(c.dataType)
        .set_dim(qDims)
        .set_stride(qStrides);
    dk->set_uid(bwd_uid::DK)
        .set_name("dK")
        .set_output(true)
        .set_data_type(c.dataType)
        .set_dim(kDims)
        .set_stride(kStrides);
    dv->set_uid(bwd_uid::DV)
        .set_name("dV")
        .set_output(true)
        .set_data_type(c.dataType)
        .set_dim(vDims)
        .set_stride(vStrides);

    prepared.graph = graph;
    const auto type = elementTypeOf(c.dataType);
    prepared.outputs[bwd_uid::DQ] = {"dQ", type, logicalRuns(qDims, qStrides), false};
    prepared.outputs[bwd_uid::DK] = {"dK", type, logicalRuns(kDims, kStrides), false};
    prepared.outputs[bwd_uid::DV] = {"dV", type, logicalRuns(vDims, vStrides), false};
    // The last live query row of batch 0 sees keys (the first row of a causal graph sees one).
    const int64_t liveRows = c.seqLens.empty() ? c.seqQ : c.seqLens.front();
    prepared.sensitivityUid = bwd_uid::Q;
    prepared.sensitivityElement = static_cast<size_t>((liveRows - 1) * c.queryHeads * pitch);
    prepared.deriveInputs
        = [graphCase, pitch](hipdnnHandle_t /*handle*/,
                             hipStream_t /*stream*/,
                             hipdnn_test_sdk::utilities::GraphTensorBundle& bundle) {
              return hostLogsumexp(graphCase, pitch, bundle);
          };
    return prepared;
}

[[maybe_unused]] RowMeta metaFor(const char* engine,
                                 const char* caseName,
                                 const std::vector<std::string>& archs,
                                 KnobList knobs,
                                 KnobClass knobClass)
{
    return RowMeta{
        engine, caseName, archs, Sku::ANY, Disposition::PROVEN, knobClass, std::move(knobs)};
}

/// A row on the graph this file builds.
[[maybe_unused]] void addCase(std::vector<DeterminismRow>& rows,
                              const char* engine,
                              const char* caseName,
                              const std::vector<std::string>& archs,
                              BwdCase graph,
                              KnobList knobs = {},
                              KnobClass knobClass = KnobClass::STRUCTURE)
{
    graph.name = caseName;
    rows.push_back(backwardRowWith(
        metaFor(engine, caseName, archs, std::move(knobs), knobClass),
        [graph = std::move(graph)](const DeterminismRow& /*row*/) { return prepareCase(graph); }));
}

/// A 16-bit row on the harness graph (O and LSE from a forward launch).
[[maybe_unused]] void addPlain(std::vector<DeterminismRow>& rows,
                               const char* engine,
                               const char* caseName,
                               const std::vector<std::string>& archs,
                               DataType type,
                               int64_t queryHeads,
                               int64_t kvHeads,
                               int64_t headSize,
                               bool causal)
{
    DeterminismBwdSpec spec;
    spec.name = caseName;
    spec.dataType = type;
    spec.batch = 2;
    spec.queryHeads = queryHeads;
    spec.kvHeads = kvHeads;
    spec.seqQ = 128;
    spec.seqKv = 128;
    spec.headSize = headSize;
    spec.rightBound = causal ? std::optional<int64_t>(0) : std::nullopt;
    spec.layout = SdpaLayout::BSHD;
    rows.push_back(backwardRow(metaFor(engine, caseName, archs, {}, KnobClass::STRUCTURE), spec));
}

std::vector<DeterminismRow> flydslBackwardRows()
{
    std::vector<DeterminismRow> rows;
#ifdef HKP_DETERMINISM_FLYDSL_BWD_MERGED
    {
        constexpr const char* MERGED = "hipkernel:flydsl_sdpa_bwd_merged";
        const std::vector<std::string> archs = {"gfx950"};
        constexpr auto TILE = KnobClass::TILE_SCHEDULE;
        constexpr auto STRUCT = KnobClass::STRUCTURE;
        constexpr int64_t ROLE_AUTO = 0;
        constexpr int64_t ROLE_LPT = 1;
        // No row pins num_stream_buffers 1 or 2 on the merged engine: it is no knob of the merged UED
        // (flydsl_sdpa_bwd_merged.ued.json knobs) or KMD, and no merged object name carries nsb
        // (sdpa_bwd_merged/gfx950.SOURCE.md). The only nsb object is the split dkdv
        // sdpa_bwd_dkdv_bf16_d160_causal1_pad1_flg1_dfbk_nsb1 (dkdv/gfx950.SOURCE.md:140).

        // Head sizes (objects d32 to d256 ship; hd160 has no f16 object) and the dK/dV
        // geometry classes: the default objects, the dkv_shards 2 object (nw2_mr32_dks2, d128
        // bf16 plain), the dks1 mr16 objects (d64 and d192) and the 4-wave mr16 object (d160).
        addPlain(rows, MERGED, "Bf16Hd64", archs, BF16, 4, 4, 64, true);
        addPlain(rows, MERGED, "F16Hd64", archs, F16, 4, 4, 64, false);
        addPlain(rows, MERGED, "Bf16Hd128", archs, BF16, 4, 4, 128, false);
        addPlain(rows, MERGED, "F16Hd128Causal", archs, F16, 4, 4, 128, true);
        // role_order, pinned by ordinal (the index in the sorted set of the engine's values, `auto` and `lpt`;
        // no `concat` object ships): the block-id order of the dK/dV and dQ roles. `auto` objects carry no
        // name token, `lpt` ones `rolpt`. Both dtypes, on a shape with fewer keys than queries and a
        // grouped one with more.
        addCase(rows,
                MERGED,
                "Bf16Hd128RoleAuto",
                archs,
                withSeq(graphOf(BF16, 128), 136, 120),
                {{"role_order", ROLE_AUTO}},
                STRUCT);
        addCase(rows,
                MERGED,
                "Bf16Hd128RoleLpt",
                archs,
                withSeq(graphOf(BF16, 128), 136, 120),
                {{"role_order", ROLE_LPT}},
                STRUCT);
        addCase(rows,
                MERGED,
                "F16Hd128RoleAuto",
                archs,
                withSeq(graphOf(F16, 128), 136, 120),
                {{"role_order", ROLE_AUTO}},
                STRUCT);
        addCase(rows,
                MERGED,
                "F16Hd128RoleLpt",
                archs,
                withSeq(graphOf(F16, 128), 136, 120),
                {{"role_order", ROLE_LPT}},
                STRUCT);
        addCase(rows,
                MERGED,
                "Bf16Hd128GqaRoleAuto",
                archs,
                withSeq(withHeads(graphOf(BF16, 128), 8, 2), 120, 136),
                {{"role_order", ROLE_AUTO}},
                STRUCT);
        addCase(rows,
                MERGED,
                "Bf16Hd128GqaRoleLpt",
                archs,
                withSeq(withHeads(graphOf(BF16, 128), 8, 2), 120, 136),
                {{"role_order", ROLE_LPT}},
                STRUCT);
        addCase(rows,
                MERGED,
                "F16Hd128GqaRoleAuto",
                archs,
                withSeq(withHeads(graphOf(F16, 128), 8, 2), 120, 136),
                {{"role_order", ROLE_AUTO}},
                STRUCT);
        addCase(rows,
                MERGED,
                "F16Hd128GqaRoleLpt",
                archs,
                withSeq(withHeads(graphOf(F16, 128), 8, 2), 120, 136),
                {{"role_order", ROLE_LPT}},
                STRUCT);
        addCase(
            rows, MERGED, "Bf16Hd128Dks2", archs, graphOf(BF16, 128), {{"dkv_shards", 2}}, STRUCT);
        addCase(
            rows, MERGED, "Bf16Hd64Dks1", archs, graphOf(BF16, 64), {{"dkv_shards", 1}}, STRUCT);
        addCase(rows, MERGED, "Bf16Hd160", archs, withBatchHeads(graphOf(BF16, 160), 1, 2));
        addCase(rows,
                MERGED,
                "Bf16Hd192Causal",
                archs,
                withCausal(withBatchHeads(graphOf(BF16, 192), 1, 2)));
        addCase(rows, MERGED, "F16Hd96", archs, graphOf(F16, 96));
        addCase(rows, MERGED, "F16Hd256", archs, withBatchHeads(graphOf(F16, 256), 1, 2));
        addCase(rows,
                MERGED,
                "Bf16Hd256Causal",
                archs,
                withCausal(withBatchHeads(graphOf(BF16, 256), 1, 2)));
        // Sequence lengths that fill no tile.
        addCase(rows, MERGED, "Bf16Hd128Tail", archs, withSeq(graphOf(BF16, 128), 120, 136));
        addCase(rows,
                MERGED,
                "Bf16Hd64CausalTail",
                archs,
                withCausal(withSeq(graphOf(BF16, 64), 100, 100)));

        // Grouped query heads: the runtime group fold, and the gqa_pack objects (gp1: bf16 d64
        // causal, d128; f16 d128).
        addCase(rows, MERGED, "Bf16Hd128Gqa", archs, withHeads(graphOf(BF16, 128), 8, 2));
        addCase(rows,
                MERGED,
                "Bf16Hd128GqaPack",
                archs,
                withHeads(graphOf(BF16, 128), 8, 2),
                {{"gqa_pack", 1}},
                STRUCT);
        addCase(rows,
                MERGED,
                "Bf16Hd64GqaPackCausal",
                archs,
                withCausal(withHeads(graphOf(BF16, 64), 8, 2)),
                {{"gqa_pack", 1}},
                STRUCT);
        addCase(rows,
                MERGED,
                "F16Hd128GqaPack",
                archs,
                withHeads(graphOf(F16, 128), 8, 2),
                {{"gqa_pack", 1}},
                STRUCT);
        addCase(rows, MERGED, "F16Hd64Mqa", archs, withHeads(graphOf(F16, 64), 8, 1));

        // Padded head sizes: the padded tile-32 objects (pad1, d32) serve 1 to 32; 20 and 30
        // are off 8, so the 16-bit row pitch is 24 and 32 and the stores write a gap (HKP-H34).
        addCase(rows, MERGED, "Bf16Hd20Padded", archs, graphOf(BF16, 20), {{"padded", 1}}, TILE);
        addCase(rows, MERGED, "F16Hd30PaddedCausal", archs, withCausal(graphOf(F16, 30)));
        addCase(rows, MERGED, "Bf16Hd20PaddedDropout", archs, withDropout(graphOf(BF16, 20), true));

        // Bias and dBias (dense, non-causal): the in-place dB store (store_db objects), the
        // reduce launch for a bias broadcast over batch, heads or both, packed GQA with dBias,
        // and a bias with no dB store. A dBias needs a store_db object, so no d64 sdb0 row.
        addCase(rows, MERGED, "Bf16Hd64Bias", archs, withBias(graphOf(BF16, 64), BF16, false));
        addCase(rows, MERGED, "Bf16Hd128Dbias", archs, withBias(graphOf(BF16, 128), BF16, true));
        addCase(rows, MERGED, "F16Hd128Dbias", archs, withBias(graphOf(F16, 128), F16, true));
        addCase(rows,
                MERGED,
                "Bf16Hd128DbiasBatchBroadcast",
                archs,
                withBias(graphOf(BF16, 128), BF16, true, true, false));
        addCase(rows,
                MERGED,
                "Bf16Hd128DbiasHeadBroadcast",
                archs,
                withBias(graphOf(BF16, 128), BF16, true, false, true));
        addCase(rows,
                MERGED,
                "F16Hd128DbiasBothBroadcast",
                archs,
                withBias(graphOf(F16, 128), F16, true, true, true));
        addCase(rows,
                MERGED,
                "Bf16Hd128DbiasGqaPack",
                archs,
                withBias(withHeads(graphOf(BF16, 128), 8, 2), BF16, true),
                {{"gqa_pack", 1}},
                STRUCT);
        addCase(rows,
                MERGED,
                "Bf16Hd128BiasNoStoreDb",
                archs,
                withBias(graphOf(BF16, 128), BF16, false),
                {{"store_db", 0}},
                TILE);
        addCase(rows,
                MERGED,
                "Bf16Hd128DbiasDropout",
                archs,
                withDropout(withBias(graphOf(BF16, 128), BF16, true), true));

        // Philox dropout (the baked dform bk objects), both seed forms and packed GQA.
        addCase(rows, MERGED, "Bf16Hd128Dropout", archs, withDropout(graphOf(BF16, 128), true));
        addCase(
            rows, MERGED, "Bf16Hd64DropoutByValue", archs, withDropout(graphOf(BF16, 64), false));
        addCase(rows,
                MERGED,
                "F16Hd256Dropout",
                archs,
                withDropout(withBatchHeads(graphOf(F16, 256), 1, 2), true));
        addCase(rows,
                MERGED,
                "Bf16Hd128DropoutGqaPack",
                archs,
                withDropout(withHeads(graphOf(BF16, 128), 8, 2), true),
                {{"gqa_pack", 1}},
                STRUCT);

        // padding_mask: the tail-zero launch zeroes the rows past each length of dQ, dK and dV,
        // which are live elements of the dense outputs.
        addCase(rows, MERGED, "Bf16Hd128Padding", archs, withLens(graphOf(BF16, 128), {128, 77}));
        addCase(rows,
                MERGED,
                "F16Hd64PaddingCausal",
                archs,
                withLens(withCausal(graphOf(F16, 64)), {128, 77}));
        addCase(rows,
                MERGED,
                "Bf16Hd64PaddingDropout",
                archs,
                withDropout(withLens(graphOf(BF16, 64), {128, 77}), true));
        addCase(rows,
                MERGED,
                "Bf16Hd128PaddingGqa",
                archs,
                withLens(withHeads(graphOf(BF16, 128), 8, 2), {128, 77}));
    }
#endif
#ifdef HKP_DETERMINISM_FLYDSL_BWD_SPLIT
    {
        constexpr const char* SPLIT = "hipkernel:flydsl_sdpa_bwd_split";
        const std::vector<std::string> all = {"gfx950", "gfx942", "gfx90a", "gfx1100"};
        // Head size 512 ships for the CDNA targets only (gfx11-generic stops at 256).
        const std::vector<std::string> cdna = {"gfx950", "gfx942", "gfx90a"};
        const std::vector<std::string> gfx950 = {"gfx950"};

        // Plain: prep, dkdv and dq per dtype, causal and not, and the GQA group fold.
        addPlain(rows, SPLIT, "Bf16Hd64Causal", all, BF16, 4, 4, 64, true);
        addPlain(rows, SPLIT, "F16Hd128", all, F16, 4, 4, 128, false);
        addPlain(rows, SPLIT, "Bf16Hd64Gqa", all, BF16, 8, 2, 64, false);
        addPlain(rows, SPLIT, "F16Hd128GqaCausal", all, F16, 8, 2, 128, true);
        addCase(rows, SPLIT, "Bf16Hd256", all, withBatchHeads(graphOf(BF16, 256), 1, 2));
        addCase(rows, SPLIT, "Bf16Hd512", cdna, withBatchHeads(graphOf(BF16, 512), 1, 2));
        addCase(rows, SPLIT, "Bf16Hd64Tail", all, withSeq(graphOf(BF16, 64), 120, 136));

        // fp32 (the per-wave kernels and the D-split ones at 512; gfx942 and gfx90a reuse the
        // gfx950 fp32 kernels).
        addCase(rows, SPLIT, "F32Hd64", all, graphOf(F32, 64));
        addCase(rows, SPLIT, "F32Hd128Causal", all, withCausal(graphOf(F32, 128)));
        addCase(rows, SPLIT, "F32Hd256", all, withBatchHeads(graphOf(F32, 256), 1, 2));
        addCase(rows, SPLIT, "F32Hd512", cdna, withBatchHeads(graphOf(F32, 512), 1, 2));
        addCase(rows, SPLIT, "F32Hd64Gqa", all, withHeads(graphOf(F32, 64), 8, 2));
        // gfx11-generic bakes contraction_shards, split_head_dim and transposed_source per tile by
        // policy, not by a knob (the table rows leave them at policy: _instances.py:5558 tables,
        // knobs :2711-2735; no UED knob, KMD field or object-name token carries them), so a row
        // selects an object by its tile, which head_dim pins in both slots (an INT KMD field, a
        // UED knob of the dq and dkdv slots).
        //  - dq contraction_shards: 1 up to d96, 2 at d128 to d224, 4 at d256
        //    (fmha_tuning_bwd_dq_gfx11.py:57-71). Reached by existing rows: shards 2 by F16Hd128,
        //    F16Hd128DbiasBothBroadcast and the D100 padded rows (tile 128), shards 4 by Bf16Hd256.
        //  - dkdv: team split C1 (shards 1) at d96 and d128, C2 (shards 2) from d160
        //    (fmha_tuning_bwd_dkdv_gfx11.py:73-85). C1 is reached by F16Hd128, C2 by Bf16Hd256.
        //  - Not reached by an existing row: tiles d96, d160, d192 and d224 (the 8-wave and
        //    16-wave schedules differ), so the four rows below pin them.
        //  - transposed_source: gfx11-generic objects are all `tile` (:285, unpinned and no policy
        //    value); `derived` ships only for gfx942 and gfx90a (POLICY_MFMA16, :94-107), which
        //    every 16-bit split row on those targets reaches. No gfx1100 row for it.
        const std::vector<std::string> gfx1100 = {"gfx1100"};
        addCase(rows,
                SPLIT,
                "Bf16Hd96",
                gfx1100,
                withBatchHeads(graphOf(BF16, 96), 1, 2),
                {{"dq.head_dim", 96}, {"dkdv.head_dim", 96}},
                KnobClass::STRUCTURE);
        addCase(rows,
                SPLIT,
                "Bf16Hd160",
                gfx1100,
                withBatchHeads(graphOf(BF16, 160), 1, 2),
                {{"dq.head_dim", 160}, {"dkdv.head_dim", 160}},
                KnobClass::STRUCTURE);
        addCase(rows,
                SPLIT,
                "F16Hd192",
                gfx1100,
                withBatchHeads(graphOf(F16, 192), 1, 2),
                {{"dq.head_dim", 192}, {"dkdv.head_dim", 192}},
                KnobClass::STRUCTURE);
        addCase(rows,
                SPLIT,
                "Bf16Hd224Causal",
                gfx1100,
                withCausal(withBatchHeads(graphOf(BF16, 224), 1, 2)),
                {{"dq.head_dim", 224}, {"dkdv.head_dim", 224}},
                KnobClass::STRUCTURE);
        // Slot-qualified pins (dkdv.<knob>): the gp1 object (D96 and a group of 4) and the
        // dks1 4-wave object at D160 (nw4_mr16_dks1), both spelled in the gfx950 dkdv names.
        addCase(rows,
                SPLIT,
                "Bf16Hd96GqaPackPin",
                gfx950,
                withHeads(graphOf(BF16, 96), 8, 2),
                {{"dkdv.gqa_pack", 1}},
                KnobClass::STRUCTURE);
        addCase(rows,
                SPLIT,
                "Bf16Hd160Dks1Pin",
                gfx950,
                withBatchHeads(graphOf(BF16, 160), 1, 2),
                {{"dkdv.dkv_shards", 1}},
                KnobClass::STRUCTURE);
        // dkdv num_stream_buffers 1 (nsb1: the object sdpa_bwd_dkdv_bf16_d160_causal1_pad1_flg1_dfbk_nsb1,
        // dkdv/gfx950.SOURCE.md:140). Every other gfx950 dkdv row reaches the default of two
        // buffers (fmha_tuning_bwd_dkdv_gfx950.py:828-833: 2 unless 4 slots exceed the LDS cap),
        // for instance Bf16Hd64Causal and F16Hd128; the pin is an integer knob (the field
        // addresses itself, KernelIngestorStateManager.hpp:1631).
        addCase(rows,
                SPLIT,
                "Bf16Hd160CausalNsb1Pin",
                gfx950,
                withCausal(withBatchHeads(graphOf(BF16, 160), 1, 2)),
                {{"dkdv.num_stream_buffers", 1}},
                KnobClass::TILE_SCHEDULE);
        // Head size 512 with dBias on gfx942 and gfx90a: the dq d512 bias objects (sdb1 stores
        // dB, SOURCE.md d512_causal0_bias1) and the d512 bias dkdv objects.
        addCase(rows,
                SPLIT,
                "Bf16Hd512Dbias",
                std::vector<std::string>{"gfx942", "gfx90a"},
                withBias(withBatchHeads(graphOf(BF16, 512), 1, 2), BF16, true));

        // Padded head sizes: 100 is off 8 (row pitch 104) and 98 off 4 for fp32 (pitch 100).
        addCase(rows, SPLIT, "Bf16Hd100Padded", all, graphOf(BF16, 100));
        addCase(rows, SPLIT, "F16Hd100PaddedCausal", all, withCausal(graphOf(F16, 100)));
        addCase(rows, SPLIT, "F32Hd98Padded", all, graphOf(F32, 98));

        // Bias and dBias (dense, non-causal): the dB store of dq, the reduce launch for a
        // broadcast bias, an f32 bias beside a 16-bit Q (head sizes 64 and 128 only), fp32 bias.
        addCase(rows, SPLIT, "Bf16Hd64Bias", all, withBias(graphOf(BF16, 64), BF16, false));
        addCase(rows, SPLIT, "Bf16Hd64Dbias", all, withBias(graphOf(BF16, 64), BF16, true));
        addCase(rows,
                SPLIT,
                "F16Hd128DbiasBothBroadcast",
                all,
                withBias(graphOf(F16, 128), F16, true, true, true));
        addCase(rows,
                SPLIT,
                "Bf16Hd64DbiasBatchBroadcast",
                all,
                withBias(graphOf(BF16, 64), BF16, true, true, false));
        addCase(rows,
                SPLIT,
                "F16Hd64DbiasHeadBroadcast",
                all,
                withBias(graphOf(F16, 64), F16, true, false, true));
        addCase(rows, SPLIT, "F32Hd64Dbias", all, withBias(graphOf(F32, 64), F32, true));
        addCase(rows,
                SPLIT,
                "F32Hd64DbiasBothBroadcast",
                all,
                withBias(graphOf(F32, 64), F32, true, true, true));
        addCase(rows, SPLIT, "Bf16Hd64F32BiasDbias", all, withBias(graphOf(BF16, 64), F32, true));
        addCase(rows,
                SPLIT,
                "F16Hd128F32BiasDbiasBothBroadcast",
                all,
                withBias(graphOf(F16, 128), F32, true, true, true));
        addCase(rows,
                SPLIT,
                "Bf16Hd64DbiasGqa",
                all,
                withBias(withHeads(graphOf(BF16, 64), 8, 2), BF16, true));
        addCase(rows,
                SPLIT,
                "Bf16Hd64DbiasDropout",
                all,
                withDropout(withBias(graphOf(BF16, 64), BF16, true), true));
        // A 16-bit bias outside the LDS staging contract (Skv 100: the row stride is below
        // ceil8(Skv)) is served by a dq row that loads the bias directly; gfx950 ships
        // `bstdirect` rows for it, and a dBias would decline (strides off 8, BI-9).
        addCase(rows,
                SPLIT,
                "Bf16Hd64BiasUnstaged",
                gfx950,
                withBias(withSeq(graphOf(BF16, 64), 128, 100), BF16, false));

        // Dropout: the baked objects on every target, the runtime-flag ones too on gfx950.
        addCase(rows, SPLIT, "Bf16Hd64Dropout", all, withDropout(graphOf(BF16, 64), true));
        addCase(rows,
                SPLIT,
                "F16Hd128DropoutCausal",
                all,
                withDropout(withCausal(graphOf(F16, 128)), false));
        addCase(rows, SPLIT, "F32Hd64Dropout", all, withDropout(graphOf(F32, 64), true));
        addCase(rows,
                SPLIT,
                "Bf16Hd64DropoutGqa",
                all,
                withDropout(withHeads(graphOf(BF16, 64), 8, 2), true));

        // padding_mask: prep, dkdv and dq never read the rows past a length, and the tail-zero
        // launch writes them.
        addCase(rows, SPLIT, "Bf16Hd64Padding", all, withLens(graphOf(BF16, 64), {128, 77}));
        addCase(rows,
                SPLIT,
                "F16Hd128PaddingCausal",
                all,
                withLens(withCausal(graphOf(F16, 128)), {128, 77}));
        addCase(rows, SPLIT, "F32Hd64Padding", all, withLens(graphOf(F32, 64), {128, 77}));
        addCase(rows,
                SPLIT,
                "Bf16Hd64PaddingDropout",
                all,
                withDropout(withLens(graphOf(BF16, 64), {128, 77}), true));

        // V heads that differ from K's (head sizes 64 and 128, 16-bit, no bias): the dkdv_u2
        // pair (a dK-only and a dV-only launch) and the V-head tail-zero object.
        addCase(rows,
                SPLIT,
                "Bf16Hd64ValueHeads",
                all,
                withValueHeads(withHeads(graphOf(BF16, 64), 8, 2), 4));
        addCase(rows,
                SPLIT,
                "F16Hd128ValueHeadsCausal",
                all,
                withCausal(withValueHeads(withHeads(graphOf(F16, 128), 8, 2), 4)));
        addCase(rows,
                SPLIT,
                "Bf16Hd64ValueHeadsDropout",
                all,
                withDropout(withValueHeads(withHeads(graphOf(BF16, 64), 8, 2), 4), true));
        addCase(rows,
                SPLIT,
                "Bf16Hd64ValueHeadsPadding",
                all,
                withLens(withValueHeads(withHeads(graphOf(BF16, 64), 8, 2), 4), {128, 77}));
    }
#endif
    return rows;
}

[[maybe_unused]] const bool REGISTERED = registerRows(flydslBackwardRows());

} // namespace

#endif
