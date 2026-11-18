// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_test_sdk/utilities/ReferenceValidationInterface.hpp>
#include <hipdnn_test_sdk/utilities/ScopedTestCacheDir.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>

#include "../IntegrationGraphVerificationHarness.hpp"
#include "IngestorIntegrationSupport.hpp"

/**
 * @file FlydslSdpaFrontendGraphs.hpp
 * @brief The frontend-built test lane for the FlyDSL SDPA engines: an SDPA forward graph built
 *        through the public frontend Graph API (Graph::sdpa), checked by the frontend's own rules
 *        (Graph::validate()), lowered by the backend, served by a FlyDSL engine and compared with
 *        hipDNN's CPU reference graph executor on the same graph. The unit tests' raw flatbuffers
 *        never meet the frontend's rules; this lane does.
 *
 * Add a field to FrontendSdpaSpec for each served feature a test needs to reach, and a case to a
 * suite built on IntegrationGpuFlydslFrontendLane.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::integration
{

/// Philox dropout as a graph asks for it: a probability with a seed and an offset (INT64
/// [1, 1, 1, 1], on the device or passed by value at execute), and optionally
/// the RNG dump (FLOAT [B, Hq, Sq, Skv]) the reference reads back as its explicit mask (SEM-015).
struct FrontendDropout
{
    float probability = 0.2F;
    int64_t seed = 0x5eed1234;
    int64_t offset = 0x40;
    /// Device tensors (PyTorch-ROCm's placement); host scalars passed by value when false.
    bool onDevice = true;
    /// The hipDNN CPU reference serves Philox dropout only through the dump.
    bool withRngDump = true;
};

/// Where a case pinned to an engine (FrontendSdpaSpec::engine on engineArch) runs. The standard
/// tier runs it on engineArch, the architecture it exists for (pinnedCasesOnTheirArch); a
/// Comprehensive* instantiation runs it everywhere else (pinnedCasesElsewhere).
enum class PinnedArchRun
{
    EVERY_ARCH,
    ENGINE_ARCH_ONLY,
    OTHER_ARCHS_ONLY,
};

/// One SDPA forward graph: BSHD Q/K/V/O, an optional LSE (FLOAT [B, Hq, Sq, 1]), an optional
/// attention sink [1, Hq, 1, 1] (FLOAT, SEM-022), a diagonal band, per-batch lengths and dropout.
struct FrontendSdpaSpec
{
    std::string name = "flydsl_sdpa_frontend";
    hipdnn_frontend::DataType dataType = hipdnn_frontend::DataType::BFLOAT16;
    int64_t batch = 2;
    int64_t queryHeads = 8;
    int64_t kvHeads = 2;
    /// V's own head count (SEM-021's Hk != Hv); K's when unset.
    std::optional<int64_t> valueHeads;
    int64_t seqQ = 128;
    int64_t seqKv = 128;
    int64_t headSize = 64;
    /// V's and O's head size when it differs from Q/K's (MLA, a pair).
    std::optional<int64_t> valueHeadSize;
    /// The band's bounds (cuDNN's diagonal_band_{left,right}_bound); unset is unbounded. The
    /// default, right 0, is causal.
    std::optional<int64_t> leftBound;
    std::optional<int64_t> rightBound = 0;
    hipdnn_frontend::DiagonalAlignment alignment = hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT;
    /// Per-batch lengths (INT32 [B, 1, 1, 1] each, both or neither; seqLensKv alone over packed
    /// sequences, PyTorch's seqused_k, SEM-028); dense when empty.
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
    /// Whether the graph sets padding_mask beside the lengths (SEM-008: lengths alone pad too).
    bool paddingMask = true;
    std::optional<FrontendDropout> dropout;
    bool withStats = true;
    /// The sink's dtype; no sink when unset.
    std::optional<hipdnn_frontend::DataType> sinkDataType;
    /// An additive bias in this dtype, packed row-major over biasDims ([B, Hq, Sq, Skv] when
    /// empty; an extent of 1 broadcasts); none when unset.
    std::optional<hipdnn_frontend::DataType> biasDataType;
    std::vector<int64_t> biasDims;
    /// The scale; 1 / sqrt(headSize) when unset.
    std::optional<float> scale;
    /// Packed THD (RFC 0014): Q and O through one token-offset table, K and V through another,
    /// each INT32 [B + 1, 1, 1, 1] with the tensor's token stride as its multiplier, and the LSE
    /// through Q's (token-major [T, Hq]). These are each batch's slot lengths, at most seqQ and
    /// seqKv; with seqLensQ/Kv too, those are the live rows and keys of each slot (cuDNN's
    /// ragged + padding form). Dense when empty; both or neither.
    std::vector<int32_t> packedLensQ;
    std::vector<int32_t> packedLensKv;
    /// Packed with that side's per-batch lengths: each batch's first token in Q's (O's, the
    /// LSE's) or K's (V's) table, in any order and with gaps (cuDNN's per-batch origins, RFC
    /// 0014), its packedLens* region starting there and within the tensor's B * S tokens; the
    /// table's last entry is then 0, which bounds no sequence. End to end when empty.
    std::vector<int32_t> packedOriginsQ;
    std::vector<int32_t> packedOriginsKv;
    /// A value written into every K and V element no live query row sees (keys past a batch's
    /// key length, keys outside every row's band), in the engine's and the reference's inputs
    /// alike; random like the rest when unset.
    std::optional<float> unseenKeyValue;
    /// Which of K and V unseenKeyValue goes into.
    bool unseenInK = true;
    bool unseenInV = true;
    /// An engine to serve the case instead of the first-ranked FlyDSL one, by descriptor name
    /// (one outside FLYDSL_SDPA_ENGINES, such as gfx950's dual-wave kernel); on engineArch the
    /// backend must offer it, elsewhere the first-ranked FlyDSL engine serves.
    std::optional<std::string> engine;
    std::string engineArch;
    /// Which architectures run a pinned case; ignored without engine and engineArch.
    PinnedArchRun pinnedArchRun = PinnedArchRun::EVERY_ARCH;
    /// Every Q, K and V element this value, in the engine's and the reference's inputs alike
    /// (AOTriton's test_large_bf16_nan_values fills them with 133,120); random when unset.
    std::optional<float> inputValue;
    /// Q, K and V drawn from N(0, 1) instead of the harness's uniform values (AOTriton's
    /// test_sm_scale_magnitude), in the engine's and the reference's inputs alike.
    bool normalInputs = false;
    /// O's absolute tolerance, with no relative one, over the dtype's forward SDPA tolerance
    /// (an external bound such as AOTriton's on test_sm_scale_magnitude).
    std::optional<float> outputAbsoluteTolerance;
};

/// The graph and its tensors, by role.
struct FrontendSdpaGraph
{
    std::shared_ptr<hipdnn_frontend::graph::Graph> graph;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> output;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> stats;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> sink;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> seqLenQ;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> seqLenKv;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> seed;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> offset;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> rngDump;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> offsetsQ;
    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> offsetsKv;
    /// What the lane fills them with.
    std::vector<int32_t> seqLensQ;
    std::vector<int32_t> seqLensKv;
    std::optional<FrontendDropout> dropout;
    /// The spec the graph was built from.
    FrontendSdpaSpec spec;
};

namespace frontend_sdpa_detail
{

inline constexpr int64_t Q_UID = 1;
inline constexpr int64_t K_UID = 2;
inline constexpr int64_t V_UID = 3;
inline constexpr int64_t O_UID = 4;
inline constexpr int64_t STATS_UID = 5;
inline constexpr int64_t SINK_UID = 6;
inline constexpr int64_t SEQ_LEN_Q_UID = 7;
inline constexpr int64_t SEQ_LEN_KV_UID = 8;
inline constexpr int64_t SEED_UID = 9;
inline constexpr int64_t OFFSET_UID = 10;
inline constexpr int64_t RNG_DUMP_UID = 11;
inline constexpr int64_t BIAS_UID = 12;
inline constexpr int64_t OFFSETS_Q_UID = 13;
inline constexpr int64_t OFFSETS_KV_UID = 14;

/// The FlyDSL SDPA forward engines (their descriptors' names); whichever the backend ranks first
/// serves a lane case.
inline constexpr std::array<const char*, 5> FLYDSL_SDPA_ENGINES
    = {"hipkernel:flydsl_sdpa_rdna",
       "hipkernel:flydsl_sdpa_rdna_decode",
       "hipkernel:flydsl_sdpa_aiw",
       "hipkernel:flydsl_sdpa_cdna",
       "hipkernel:flydsl_sdpa_cdna_decode"};

#ifndef HIPKERNELPROVIDER_FLYDSL_SDPA_ARCHS
// integration_tests/CMakeLists.txt defines it for this binary; a build with no FlyDSL content
// has no architecture.
#define HIPKERNELPROVIDER_FLYDSL_SDPA_ARCHS ""
#endif

/// The architectures the FlyDSL SDPA forward engines have objects for, from the descriptors'
/// content list as CMake expands it (gfx11-generic and gfx12-generic into their member
/// architectures). Exact names, not prefixes: an architecture FlyDSL ships nothing for, such as
/// gfx1250, is not on it.
inline std::vector<std::string> flydslSdpaArches()
{
    // Through a pointer: with no FlyDSL content the macro is "", and initializing a string
    // from that literal directly is a redundant-string-init error under clang-tidy.
    const char* const raw = HIPKERNELPROVIDER_FLYDSL_SDPA_ARCHS;
    const std::string list = raw;
    std::vector<std::string> arches;
    size_t begin = 0;
    while(begin < list.size())
    {
        auto end = list.find(',', begin);
        if(end == std::string::npos)
        {
            end = list.size();
        }
        if(end > begin)
        {
            arches.push_back(list.substr(begin, end - begin));
        }
        begin = end + 1;
    }
    return arches;
}

/// Whether FlyDSL ships SDPA forward objects for @p arch (a device's bare gfx name).
inline bool flydslSdpaServes(const std::string& arch)
{
    const auto arches = flydslSdpaArches();
    return std::find(arches.begin(), arches.end(), arch) != arches.end();
}

/// A tensor of the given dims, packed row-major.
inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    packedTensor(int64_t uid,
                 const std::string& name,
                 hipdnn_frontend::DataType dataType,
                 const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for(size_t i = dims.size(); i-- > 1;)
    {
        strides[i - 1] = strides[i] * dims[i];
    }
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
    tensor->set_uid(uid).set_name(name).set_dim(dims).set_stride(strides).set_data_type(dataType);
    return tensor;
}

/// A Philox seed or offset: INT64 [1, 1, 1, 1] on the device, or an INT64 scalar passed by
/// value at execute (a host pointer in the variant pack, RFC 0016).
inline std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    philoxTensor(int64_t uid, const std::string& name, int64_t value, bool onDevice)
{
    if(onDevice)
    {
        return packedTensor(uid, name, hipdnn_frontend::DataType::INT64, {1, 1, 1, 1});
    }
    auto tensor = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(
        value, hipdnn_frontend::graph::ScalarType::RUNTIME_PARAM);
    tensor->set_uid(uid).set_name(name).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1});
    return tensor;
}

/// A packed tensor's token offsets: [0, l0, l0 + l1, ...].
inline std::vector<int32_t> tokenOffsets(const std::vector<int32_t>& lengths)
{
    std::vector<int32_t> offsets = {0};
    for(const auto length : lengths)
    {
        offsets.push_back(offsets.back() + length);
    }
    return offsets;
}

/// A packed tensor's ragged-offset table: @p origins and a last entry of 0 (per-batch origins),
/// or tokenOffsets(@p lengths) when there are none.
inline std::vector<int32_t> tokenTable(const std::vector<int32_t>& origins,
                                       const std::vector<int32_t>& lengths)
{
    if(origins.empty())
    {
        return tokenOffsets(lengths);
    }
    auto table = origins;
    table.push_back(0);
    return table;
}

/// Batch b's live query rows and keys: its seq_len values, else its packed slot, else the extent.
inline std::pair<int64_t, int64_t> liveLengths(const FrontendSdpaSpec& spec, int64_t b)
{
    const auto at = static_cast<size_t>(b);
    const auto live = [at](const std::vector<int32_t>& lengths,
                           const std::vector<int32_t>& packed,
                           int64_t extent) -> int64_t {
        if(!lengths.empty())
        {
            return lengths[at];
        }
        if(!packed.empty())
        {
            return packed[at];
        }
        return extent;
    };
    return {live(spec.seqLensQ, spec.packedLensQ, spec.seqQ),
            live(spec.seqLensKv, spec.packedLensKv, spec.seqKv)};
}

/// Whether some live query row of batch b sees key j: below the key length and inside the band
/// (cuDNN's bounds around the diagonal i + offset, SEM-001/004).
inline bool someRowSeesKey(const FrontendSdpaSpec& spec, int64_t b, int64_t j)
{
    const auto [qLen, kvLen] = liveLengths(spec, b);
    if(j >= kvLen)
    {
        return false;
    }
    const int64_t offset
        = spec.alignment == hipdnn_frontend::DiagonalAlignment::BOTTOM_RIGHT ? kvLen - qLen : 0;
    for(int64_t i = 0; i < qLen; ++i)
    {
        if((!spec.leftBound.has_value() || j > i + offset - *spec.leftBound)
           && (!spec.rightBound.has_value() || j <= i + offset + *spec.rightBound))
        {
            return true;
        }
    }
    return false;
}

/// Writes @p value, in the query's 16-bit dtype, into every element of a BSHD (or packed) K or
/// V tensor at a key no live row sees.
inline void writeUnseenKeys(hipdnn_data_sdk::utilities::ITensor& tensor,
                            const FrontendSdpaSpec& spec,
                            int64_t heads,
                            int64_t headSize,
                            float value)
{
    uint16_t bits = 0;
    if(spec.dataType == hipdnn_frontend::DataType::HALF)
    {
        const hipdnn_data_sdk::types::half converted(value);
        std::memcpy(&bits, &converted, sizeof(bits));
    }
    else
    {
        const hipdnn_data_sdk::types::bfloat16 converted(value);
        std::memcpy(&bits, &converted, sizeof(bits));
    }
    ASSERT_EQ(tensor.elementSize(), sizeof(bits));
    auto* data = static_cast<uint16_t*>(tensor.rawHostData());
    const auto offsets = tokenTable(spec.packedOriginsKv, spec.packedLensKv);
    const int64_t tokenStride = heads * headSize;
    for(int64_t b = 0; b < spec.batch; ++b)
    {
        const int64_t slot = spec.packedLensKv.empty()
                                 ? spec.seqKv
                                 : static_cast<int64_t>(spec.packedLensKv[static_cast<size_t>(b)]);
        const int64_t first = spec.packedLensKv.empty()
                                  ? b * spec.seqKv
                                  : static_cast<int64_t>(offsets[static_cast<size_t>(b)]);
        for(int64_t j = 0; j < slot; ++j)
        {
            if(someRowSeesKey(spec, b, j))
            {
                continue;
            }
            for(int64_t e = 0; e < tokenStride; ++e)
            {
                data[static_cast<size_t>((first + j) * tokenStride + e)] = bits;
            }
        }
    }
    tensor.markHostModified();
}

/// Overwrites every element of the 16-bit operand @p tensor with the next value of @p next.
inline void writeInputValues(hipdnn_data_sdk::utilities::ITensor& tensor,
                             hipdnn_frontend::DataType dataType,
                             const std::function<float()>& next)
{
    ASSERT_EQ(tensor.elementSize(), sizeof(uint16_t));
    auto* data = static_cast<uint16_t*>(tensor.rawHostData());
    for(size_t i = 0; i < tensor.elementSpace(); ++i)
    {
        const float value = next();
        if(dataType == hipdnn_frontend::DataType::HALF)
        {
            const hipdnn_data_sdk::types::half converted(value);
            std::memcpy(&data[i], &converted, sizeof(uint16_t));
        }
        else
        {
            const hipdnn_data_sdk::types::bfloat16 converted(value);
            std::memcpy(&data[i], &converted, sizeof(uint16_t));
        }
    }
    tensor.markHostModified();
}

/// A standard normal draw source (Box-Muller over a 64-bit LCG seeded with @p seed), or the
/// constant @p value when it is set.
inline std::function<float()> inputValueSource(std::optional<float> value, uint64_t seed)
{
    if(value.has_value())
    {
        return [constant = *value]() { return constant; };
    }
    struct Normal
    {
        uint64_t state;
        std::optional<float> spare;

        double uniform()
        {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            // 53 random bits in (0, 1]: never 0, so the log below is finite.
            return (static_cast<double>(state >> 11) + 1.0) * 0x1.0p-53;
        }

        float operator()()
        {
            if(spare.has_value())
            {
                const float out = *spare;
                spare.reset();
                return out;
            }
            const double radius = std::sqrt(-2.0 * std::log(uniform()));
            const double angle = 6.283185307179586 * uniform();
            spare = static_cast<float>(radius * std::sin(angle));
            return static_cast<float>(radius * std::cos(angle));
        }
    };
    return Normal{seed * 0x9E3779B97F4A7C15ULL + 1ULL, std::nullopt};
}

} // namespace frontend_sdpa_detail

/// Builds @p spec's graph through the frontend Graph API (Graph::sdpa). The caller validates
/// and builds it.
inline FrontendSdpaGraph buildFrontendSdpaGraph(const FrontendSdpaSpec& spec)
{
    using hipdnn_frontend::DataType;
    using namespace frontend_sdpa_detail;
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name(spec.name)
        .set_io_data_type(spec.dataType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    auto q = detail::makeSdpaTensor(Q_UID,
                                    "Q",
                                    spec.dataType,
                                    spec.batch,
                                    spec.queryHeads,
                                    spec.seqQ,
                                    spec.headSize,
                                    SdpaLayout::BSHD);
    auto k = detail::makeSdpaTensor(K_UID,
                                    "K",
                                    spec.dataType,
                                    spec.batch,
                                    spec.kvHeads,
                                    spec.seqKv,
                                    spec.headSize,
                                    SdpaLayout::BSHD);
    const int64_t valueHeadSize = spec.valueHeadSize.value_or(spec.headSize);
    auto v = detail::makeSdpaTensor(V_UID,
                                    "V",
                                    spec.dataType,
                                    spec.batch,
                                    spec.valueHeads.value_or(spec.kvHeads),
                                    spec.seqKv,
                                    valueHeadSize,
                                    SdpaLayout::BSHD);

    FrontendSdpaGraph result;
    result.spec = spec;
    const bool packed = !spec.packedLensQ.empty();
    if(packed)
    {
        result.offsetsQ
            = packedTensor(OFFSETS_Q_UID, "OffsetsQ", DataType::INT32, {spec.batch + 1, 1, 1, 1});
        result.offsetsKv
            = packedTensor(OFFSETS_KV_UID, "OffsetsKv", DataType::INT32, {spec.batch + 1, 1, 1, 1});
        q->set_ragged_offset(result.offsetsQ)
            .set_ragged_offset_multiplier(spec.queryHeads * spec.headSize);
        k->set_ragged_offset(result.offsetsKv)
            .set_ragged_offset_multiplier(spec.kvHeads * spec.headSize);
        v->set_ragged_offset(result.offsetsKv)
            .set_ragged_offset_multiplier(spec.valueHeads.value_or(spec.kvHeads) * valueHeadSize);
    }

    hipdnn_frontend::graph::SdpaAttributes attributes;
    attributes.set_name(spec.name);
    attributes.set_attn_scale(
        spec.scale.value_or(1.0F / std::sqrt(static_cast<float>(spec.headSize))));
    if(spec.leftBound.has_value())
    {
        attributes.set_diagonal_band_left_bound(*spec.leftBound);
    }
    if(spec.rightBound.has_value())
    {
        attributes.set_diagonal_band_right_bound(*spec.rightBound);
    }
    attributes.set_diagonal_alignment(spec.alignment);
    attributes.set_generate_stats(spec.withStats);

    result.seqLensQ = spec.seqLensQ;
    result.seqLensKv = spec.seqLensKv;
    result.dropout = spec.dropout;
    if(!spec.seqLensQ.empty() || !spec.seqLensKv.empty())
    {
        if(!spec.seqLensQ.empty())
        {
            result.seqLenQ
                = packedTensor(SEQ_LEN_Q_UID, "SeqLenQ", DataType::INT32, {spec.batch, 1, 1, 1});
            attributes.set_seq_len_q(result.seqLenQ);
        }
        result.seqLenKv
            = packedTensor(SEQ_LEN_KV_UID, "SeqLenKv", DataType::INT32, {spec.batch, 1, 1, 1});
        attributes.set_seq_len_kv(result.seqLenKv);
        attributes.set_padding_mask(spec.paddingMask);
    }
    if(spec.biasDataType.has_value())
    {
        const auto dims
            = spec.biasDims.empty()
                  ? std::vector<int64_t>{spec.batch, spec.queryHeads, spec.seqQ, spec.seqKv}
                  : spec.biasDims;
        attributes.set_bias(packedTensor(BIAS_UID, "Bias", *spec.biasDataType, dims));
    }
    if(spec.dropout.has_value())
    {
        const auto& dropout = *spec.dropout;
        result.seed = philoxTensor(SEED_UID, "Seed", dropout.seed, dropout.onDevice);
        result.offset = philoxTensor(OFFSET_UID, "Offset", dropout.offset, dropout.onDevice);
        attributes.set_dropout(dropout.probability, result.seed, result.offset);
        if(dropout.withRngDump)
        {
            result.rngDump = packedTensor(RNG_DUMP_UID,
                                          "RngDump",
                                          DataType::FLOAT,
                                          {spec.batch, spec.queryHeads, spec.seqQ, spec.seqKv});
            result.rngDump->set_output(true);
            attributes.set_rng_dump(result.rngDump);
        }
    }
    if(spec.sinkDataType.has_value())
    {
        // One logit per query head, [1, Hq, 1, 1]: a model's 1-D [Hq] parameter, described
        // without a copy.
        result.sink = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        result.sink->set_uid(SINK_UID)
            .set_name("Sink")
            .set_dim({1, spec.queryHeads, 1, 1})
            .set_stride({spec.queryHeads, 1, 1, 1})
            .set_data_type(*spec.sinkDataType);
        attributes.set_sink_token(result.sink);
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    result.graph = graph;
    result.output = outputs[0];
    result.output->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(spec.dataType)
        .set_dim({spec.batch, spec.queryHeads, spec.seqQ, valueHeadSize})
        .set_stride(
            detail::sdpaStrides(SdpaLayout::BSHD, spec.queryHeads, spec.seqQ, valueHeadSize));
    if(packed)
    {
        result.output->set_ragged_offset(result.offsetsQ)
            .set_ragged_offset_multiplier(spec.queryHeads * valueHeadSize);
    }
    if(spec.withStats)
    {
        result.stats = outputs[1];
        result.stats->set_uid(STATS_UID)
            .set_name("Stats")
            .set_output(true)
            .set_data_type(DataType::FLOAT)
            .set_dim({spec.batch, spec.queryHeads, spec.seqQ, 1})
            .set_stride({spec.queryHeads * spec.seqQ, spec.seqQ, 1, 1});
        if(packed)
        {
            // Token-major [T, Hq] through Q's table.
            result.stats->set_stride({spec.seqQ * spec.queryHeads, 1, spec.queryHeads, 1})
                .set_ragged_offset(result.offsetsQ)
                .set_ragged_offset_multiplier(spec.queryHeads);
        }
    }
    return result;
}

/// Whether a suite's cases need a FlyDSL engine, so the lane skips on an architecture FlyDSL
/// ships no SDPA objects for. A suite whose cases each pin another engine (the ASM group-mode
/// suite) specializes it to false and skips by that engine's own architectures.
template <typename CaseType>
struct LaneNeedsFlydsl : std::true_type
{
};

/// Whether @p spec pins an engine to an architecture.
inline bool pinsAnEngine(const FrontendSdpaSpec& spec)
{
    return spec.engine.has_value() && !spec.engineArch.empty();
}

/// The standard-tier copy of @p cases: each case that pins an engine runs on its engineArch
/// only, the rest everywhere.
template <typename CaseType>
std::vector<CaseType> pinnedCasesOnTheirArch(std::vector<CaseType> cases)
{
    for(auto& testCase : cases)
    {
        if(pinsAnEngine(testCase.spec))
        {
            testCase.spec.pinnedArchRun = PinnedArchRun::ENGINE_ARCH_ONLY;
        }
    }
    return cases;
}

/// The Comprehensive* copy of @p cases: only the cases that pin an engine, on every architecture
/// but their engineArch, where the first-ranked FlyDSL engine serves them.
template <typename CaseType>
std::vector<CaseType> pinnedCasesElsewhere(const std::vector<CaseType>& cases)
{
    std::vector<CaseType> elsewhere;
    for(auto testCase : cases)
    {
        if(pinsAnEngine(testCase.spec))
        {
            testCase.spec.pinnedArchRun = PinnedArchRun::OTHER_ARCHS_ONLY;
            elsewhere.push_back(testCase);
        }
    }
    return elsewhere;
}

/// The lane's fixture: skips where no FlyDSL SDPA objects ship, seeds each tensor apart, and
/// runs a frontend-built graph through validate(), the backend, the first-ranked FlyDSL engine and
/// the CPU reference graph executor.
template <typename CaseType>
class IntegrationGpuFlydslFrontendLane
    : public hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float,
                                                                                      CaseType>
{
protected:
    using Harness
        = hip_kernel_provider::test_utilities::IntegrationGraphVerificationHarness<float, CaseType>;

    void SetUp() override
    {
        Harness::SetUp();
        if(Harness::IsSkipped() || Harness::HasFatalFailure())
        {
            return;
        }
        _arch = hip_kernel_provider_common::getDeviceString(this->_stream);
        if(LaneNeedsFlydsl<CaseType>::value && !frontend_sdpa_detail::flydslSdpaServes(_arch))
        {
            GTEST_SKIP() << "no FlyDSL SDPA objects for " << _arch;
        }
    }

    /// Different data per tensor: the seed is offset by uid. The per-batch lengths and the
    /// Philox seed and offset hold the case's values.
    void initializeBundle(const hipdnn_frontend::graph::Graph& /*graph*/,
                          hipdnn_test_sdk::utilities::GraphTensorBundle& bundle,
                          unsigned int seed) override
    {
        using namespace frontend_sdpa_detail;
        // A packed graph's offset tables are no node's operands, so the harness does not make
        // them; they join the bundle here.
        for(const auto* table : {_current != nullptr ? &_current->offsetsQ : nullptr,
                                 _current != nullptr ? &_current->offsetsKv : nullptr})
        {
            if(table != nullptr && *table != nullptr
               && bundle.tensors.count((*table)->get_uid()) == 0)
            {
                bundle.addTensor(**table,
                                 hipdnn_test_sdk::utilities::createTensorFromAttribute(**table));
            }
        }
        for(auto& [uid, tensor] : bundle.tensors)
        {
            if(uid == SEQ_LEN_Q_UID && _current != nullptr)
            {
                tensor->fillWithData(_current->seqLensQ.data(),
                                     _current->seqLensQ.size() * sizeof(int32_t));
            }
            else if(uid == SEQ_LEN_KV_UID && _current != nullptr)
            {
                tensor->fillWithData(_current->seqLensKv.data(),
                                     _current->seqLensKv.size() * sizeof(int32_t));
            }
            else if((uid == OFFSETS_Q_UID || uid == OFFSETS_KV_UID) && _current != nullptr)
            {
                const auto& spec = _current->spec;
                const auto offsets = uid == OFFSETS_Q_UID
                                         ? tokenTable(spec.packedOriginsQ, spec.packedLensQ)
                                         : tokenTable(spec.packedOriginsKv, spec.packedLensKv);
                tensor->fillWithData(offsets.data(), offsets.size() * sizeof(int32_t));
            }
            else if((uid == SEED_UID || uid == OFFSET_UID) && _current != nullptr
                    && _current->dropout.has_value())
            {
                const int64_t value
                    = uid == SEED_UID ? _current->dropout->seed : _current->dropout->offset;
                tensor->fillWithData(&value, sizeof(value));
            }
            else
            {
                bundle.randomizeTensor(uid,
                                       Harness::DEFAULT_MIN,
                                       Harness::DEFAULT_MAX,
                                       seed + static_cast<unsigned int>(uid));
            }
        }
        // A constant, or N(0, 1) draws, in Q, K and V: before the unseen keys, which overwrite
        // theirs.
        if(_current != nullptr
           && (_current->spec.inputValue.has_value() || _current->spec.normalInputs))
        {
            for(const auto uid : {Q_UID, K_UID, V_UID})
            {
                writeInputValues(*bundle.tensors.at(uid),
                                 _current->spec.dataType,
                                 inputValueSource(_current->spec.inputValue,
                                                  seed + static_cast<unsigned int>(uid)));
            }
        }
        if(_current != nullptr && _current->spec.unseenKeyValue.has_value())
        {
            const auto& spec = _current->spec;
            if(spec.unseenInK)
            {
                writeUnseenKeys(*bundle.tensors.at(K_UID),
                                spec,
                                spec.kvHeads,
                                spec.headSize,
                                *spec.unseenKeyValue);
            }
            if(spec.unseenInV)
            {
                writeUnseenKeys(*bundle.tensors.at(V_UID),
                                spec,
                                spec.valueHeads.value_or(spec.kvHeads),
                                spec.valueHeadSize.value_or(spec.headSize),
                                *spec.unseenKeyValue);
            }
        }
    }

    /// Philox dropout: the reference runs on the engine's dumped keep mask (SEM-015), as
    /// hipDNN's bundle harness hands it over (RngDumpHandoff.hpp).
    void prepareReferenceInputs(hipdnn_test_sdk::utilities::GraphTensorBundle& gpuBundle,
                                hipdnn_test_sdk::utilities::GraphTensorBundle& cpuBundle) override
    {
        if(_current == nullptr || _current->rngDump == nullptr)
        {
            return;
        }
        const auto uid = frontend_sdpa_detail::RNG_DUMP_UID;
        auto& engineDump = *gpuBundle.tensors.at(uid);
        auto& referenceDump = *cpuBundle.tensors.at(uid);
        engineDump.markDeviceModified();
        const auto bytes = engineDump.elementSpace() * engineDump.elementSize();
        ASSERT_EQ(bytes, referenceDump.elementSpace() * referenceDump.elementSize());
        referenceDump.fillWithData(engineDump.rawHostData(), bytes);
    }

    /// An RNG dump is the reference's input, not a result: every element 0 or 1, the keep rate
    /// within 6 sigma + 1/n of 1 - p (hipDNN's harness's checkRngDump), and the reference's copy
    /// unchanged.
    class RngDumpValidator : public hipdnn_test_sdk::utilities::IReferenceValidation
    {
    public:
        explicit RngDumpValidator(float probability)
            : _probability(probability)
        {
        }

        bool allClose(hipdnn_data_sdk::utilities::ITensor& reference,
                      hipdnn_data_sdk::utilities::ITensor& implementation) const override
        {
            const auto count = implementation.elementSpace();
            if(implementation.elementSize() != sizeof(float) || count == 0
               || reference.elementSpace() != count)
            {
                ADD_FAILURE() << "rng_dump must be a non-empty FLOAT tensor";
                return false;
            }
            const auto* values = static_cast<const float*>(implementation.rawHostData());
            const auto* handed = static_cast<const float*>(reference.rawHostData());
            size_t kept = 0;
            for(size_t i = 0; i < count; ++i)
            {
                if(values[i] != 0.0F && values[i] != 1.0F)
                {
                    ADD_FAILURE() << "rng_dump element " << i << " is " << values[i];
                    return false;
                }
                if(handed[i] != values[i])
                {
                    ADD_FAILURE() << "the reference's rng_dump differs at element " << i;
                    return false;
                }
                kept += values[i] == 1.0F ? 1 : 0;
            }
            const double keep = 1.0 - static_cast<double>(_probability);
            const auto n = static_cast<double>(count);
            const double rate = static_cast<double>(kept) / n;
            const double bound = 6.0 * std::sqrt(keep * (1.0 - keep) / n) + 1.0 / n;
            EXPECT_LE(std::abs(rate - keep), bound) << "rng_dump keep rate " << rate;
            return std::abs(rate - keep) <= bound;
        }

    private:
        float _probability;
    };

    /// The first FlyDSL SDPA engine the backend ranks for @p graph, or 0 if none does; what the
    /// backend ranked goes to @p ranking for the failure message.
    static int64_t firstRankedFlydslEngine(hipdnn_frontend::graph::Graph& graph,
                                           std::string& ranking)
    {
        std::vector<int64_t> ranked;
        const auto result = graph.get_ranked_engine_ids(ranked);
        ranking = result.code == hipdnn_frontend::ErrorCode::OK
                      ? "ranked:"
                      : "ranking failed: " + result.err_msg;
        for(const auto id : ranked)
        {
            ranking += " " + std::to_string(id);
            for(const auto* name : frontend_sdpa_detail::FLYDSL_SDPA_ENGINES)
            {
                if(id == hipdnn_data_sdk::utilities::engineNameToId(name))
                {
                    return id;
                }
            }
        }
        return 0;
    }

    /// The frontend's rules, the backend, a FlyDSL engine, then O (and the LSE) against the CPU
    /// reference graph executor at the forward SDPA tolerance of the query's dtype.
    void validateServeAndVerify(FrontendSdpaGraph& sdpa, hipdnn_frontend::DataType dataType)
    {
        using hipdnn_frontend::ErrorCode;
        if(pinsAnEngine(sdpa.spec))
        {
            const bool onEngineArch = _arch == sdpa.spec.engineArch;
            if(sdpa.spec.pinnedArchRun == PinnedArchRun::ENGINE_ARCH_ONLY && !onEngineArch)
            {
                GTEST_SKIP() << "pinned to " << sdpa.spec.engineArch
                             << "; its Comprehensive* copy runs it on " << _arch;
            }
            if(sdpa.spec.pinnedArchRun == PinnedArchRun::OTHER_ARCHS_ONLY && onEngineArch)
            {
                GTEST_SKIP() << "the standard-tier copy runs it on " << _arch;
            }
        }
        auto result = sdpa.graph->validate();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = sdpa.graph->build_operation_graph(this->_handle);
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        std::string ranking;
        int64_t engineId = firstRankedFlydslEngine(*sdpa.graph, ranking);
        if(sdpa.spec.engine.has_value())
        {
            const auto pinned = hipdnn_data_sdk::utilities::engineNameToId(*sdpa.spec.engine);
            std::vector<int64_t> ranked;
            static_cast<void>(sdpa.graph->get_ranked_engine_ids(ranked));
            if(std::find(ranked.begin(), ranked.end(), pinned) != ranked.end())
            {
                engineId = pinned;
            }
            else
            {
                ASSERT_NE(_arch, sdpa.spec.engineArch)
                    << *sdpa.spec.engine << " not offered to serve " << sdpa.graph->get_name()
                    << " on " << _arch << " (" << ranking << ")";
            }
        }
        ASSERT_NE(engineId, 0) << "no FlyDSL SDPA engine offered to serve "
                               << sdpa.graph->get_name() << " on " << _arch << " (" << ranking
                               << ")";
        result = sdpa.graph->create_execution_plan_ext(engineId, {});
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = sdpa.graph->check_support();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;
        result = sdpa.graph->build_plans();
        ASSERT_EQ(result.code, ErrorCode::OK) << result.err_msg;

        hip_kernel_provider::test_utilities::GraphVerificationContext context(*sdpa.graph);
        _current = &sdpa;
        // FLOAT is exact f32 on every family that serves it, so f32's own tolerance.
        float tolerance
            = hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::bfloat16>();
        if(dataType == hipdnn_frontend::DataType::HALF)
        {
            tolerance
                = hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<hipdnn_data_sdk::types::half>();
        }
        else if(dataType == hipdnn_frontend::DataType::FLOAT)
        {
            tolerance = hipdnn_test_sdk::utilities::sdpa::getToleranceFwd<float>();
        }
        if(sdpa.spec.outputAbsoluteTolerance.has_value())
        {
            this->registerValidator(context, sdpa.output, *sdpa.spec.outputAbsoluteTolerance, 0.0F);
        }
        else
        {
            this->registerValidator(context, sdpa.output, tolerance, tolerance);
        }
        if(sdpa.stats != nullptr)
        {
            // A row no key reaches (a padded row, or one the band leaves keyless) has an LSE of
            // -inf on both sides.
            this->registerValidator(
                context,
                sdpa.stats,
                hipdnn_test_sdk::utilities::createAllCloseMatchingInfinitiesValidator(
                    hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT, tolerance, tolerance));
        }
        if(sdpa.rngDump != nullptr)
        {
            this->registerValidator(context,
                                    sdpa.rngDump,
                                    std::make_unique<RngDumpValidator>(sdpa.dropout->probability));
        }
        this->verifyBuiltGraph(context, /*seed=*/0);
        _current = nullptr;
    }

    std::string _arch;
    /// The case being verified, for the bundle's lengths, seed and offset.
    const FrontendSdpaGraph* _current = nullptr;
    /// No winner record another case measured.
    hipdnn_test_sdk::utilities::ScopedTestCacheDir _cacheDir{
        "flydsl-sdpa-frontend-case", hipdnn_test_sdk::utilities::ScopedTestCacheDir::Scope::TEST};
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::integration

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
