// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <hipdnn_data_sdk/types.hpp>

#include "SdpaDeterminismCompare.hpp"

/**
 * @file SdpaDeterminismGraphs.hpp
 * @brief The input half of the SDPA determinism suite: the element encoders and the bounded,
 *        layout-independent input filler, including the FP8 E4M3 FNUZ conversion (host-only,
 *        always compiled), and the forward graph builder for the rows buildFrontendSdpaGraph
 *        cannot build (compiled with the ingestor and SDPA).
 */
namespace hip_kernel_provider::sdpa_determinism
{

/// The value of an FP8 E4M3 FNUZ byte: bias 8, no infinities, no negative zero, NaN at 0x80,
/// largest magnitude 240.
inline float fp8E4M3FnuzToFloat(uint8_t bits)
{
    if(bits == 0x80)
    {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const int exponent = (bits >> 3) & 0xF;
    const int mantissa = bits & 0x7;
    const float magnitude
        = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -10)
                        : std::ldexp(1.0F + static_cast<float>(mantissa) / 8.0F, exponent - 8);
    return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

/// The FP8 E4M3 FNUZ byte nearest @p value, ties to the even code, saturating at the largest
/// magnitude. Never 0x80: a result that rounds to zero is 0x00.
inline uint8_t fp8E4M3FnuzFromFloat(float value)
{
    // The 127 nonzero magnitudes in increasing order, by code 1..127.
    static const std::array<float, 127> s_magnitudes = [] {
        std::array<float, 127> table{};
        for(size_t code = 1; code <= 127; ++code)
        {
            table[code - 1] = fp8E4M3FnuzToFloat(static_cast<uint8_t>(code));
        }
        return table;
    }();

    if(std::isnan(value))
    {
        throw std::invalid_argument("fp8E4M3FnuzFromFloat: NaN has no finite encoding");
    }
    const float magnitude = std::min(std::abs(value), s_magnitudes.back());
    // The first magnitude not below the value; the code before it is the one just below.
    const auto upper = std::lower_bound(s_magnitudes.begin(), s_magnitudes.end(), magnitude);
    const auto upperCode = static_cast<size_t>(upper - s_magnitudes.begin()) + 1;
    const size_t lowerCode = upperCode - 1;
    const float lowerValue = lowerCode == 0 ? 0.0F : s_magnitudes[lowerCode - 1];
    const float upperValue = s_magnitudes[upperCode - 1];

    size_t code = upperCode;
    const float belowGap = magnitude - lowerValue;
    const float aboveGap = upperValue - magnitude;
    if(belowGap < aboveGap || (belowGap == aboveGap && lowerCode % 2 == 0))
    {
        code = lowerCode;
    }
    // A negative value that rounds to zero is +0: 0x80 is the NaN.
    const bool negative = std::signbit(value) && code != 0;
    return static_cast<uint8_t>(code | (negative ? 0x80U : 0U));
}

/// Writes @p value in @p type's encoding at @p destination.
inline void encodeElement(ElementType type, float value, void* destination)
{
    switch(type)
    {
    case ElementType::FLOAT32:
        std::memcpy(destination, &value, sizeof(value));
        return;
    case ElementType::BFLOAT16:
    {
        const hipdnn_data_sdk::types::bfloat16 converted(value);
        std::memcpy(destination, &converted, sizeof(uint16_t));
        return;
    }
    case ElementType::HALF:
    {
        const hipdnn_data_sdk::types::half converted(value);
        std::memcpy(destination, &converted, sizeof(uint16_t));
        return;
    }
    case ElementType::FP8_E4M3_FNUZ:
        *static_cast<uint8_t*>(destination) = fp8E4M3FnuzFromFloat(value);
        return;
    default:
        throw std::invalid_argument("encodeElement: unhandled ElementType value");
    }
}

/// Writes @p count finite values, uniform in [-bound, bound] and fixed by @p seed, in
/// @p type's encoding at @p data.
inline void fillBounded(ElementType type, void* data, size_t count, uint64_t seed, float bound)
{
    auto* bytes = static_cast<uint8_t*>(data);
    uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1ULL;
    for(size_t i = 0; i < count; ++i)
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double unit = static_cast<double>(state >> 11) * 0x1.0p-53;
        encodeElement(
            type, static_cast<float>((2.0 * unit - 1.0) * static_cast<double>(bound)), bytes);
        bytes += elementBytes(type);
    }
}

/// Writes the next value of a fixed sequence, finite and uniform in [-bound, bound], into
/// every logical element of a tensor of @p dims and @p strides at @p storage, in @p type's
/// encoding. The sequence runs in logical (row-major index) order, so one seed gives every
/// layout of the same tensor the same logical values; storage no logical element addresses is
/// left as it is.
inline void fillLogical(ElementType type,
                        void* storage,
                        const std::vector<int64_t>& dims,
                        const std::vector<int64_t>& strides,
                        uint64_t seed,
                        float bound)
{
    if(dims.size() != strides.size())
    {
        throw std::invalid_argument("fillLogical: dims and strides differ in rank");
    }
    size_t total = 1;
    for(const auto extent : dims)
    {
        total *= static_cast<size_t>(std::max<int64_t>(extent, 0));
    }
    auto* bytes = static_cast<uint8_t*>(storage);
    const size_t size = elementBytes(type);
    uint64_t state = seed * 0x9E3779B97F4A7C15ULL + 1ULL;
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t n = 0; n < total; ++n)
    {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        const double unit = static_cast<double>(state >> 11) * 0x1.0p-53;
        int64_t offset = 0;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            offset += index[axis] * strides[axis];
        }
        encodeElement(type,
                      static_cast<float>((2.0 * unit - 1.0) * static_cast<double>(bound)),
                      bytes + static_cast<size_t>(offset) * size);
        for(size_t axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
}

/// Copies every logical element of a tensor of @p dims from @p source (laid out by
/// @p sourceStrides) to @p destination (laid out by @p destinationStrides).
inline void copyLogical(size_t elementSize,
                        const void* source,
                        const std::vector<int64_t>& sourceStrides,
                        void* destination,
                        const std::vector<int64_t>& destinationStrides,
                        const std::vector<int64_t>& dims)
{
    if(dims.size() != sourceStrides.size() || dims.size() != destinationStrides.size())
    {
        throw std::invalid_argument("copyLogical: dims and strides differ in rank");
    }
    size_t total = 1;
    for(const auto extent : dims)
    {
        total *= static_cast<size_t>(std::max<int64_t>(extent, 0));
    }
    const auto* from = static_cast<const uint8_t*>(source);
    auto* to = static_cast<uint8_t*>(destination);
    std::vector<int64_t> index(dims.size(), 0);
    for(size_t n = 0; n < total; ++n)
    {
        int64_t fromOffset = 0;
        int64_t toOffset = 0;
        for(size_t axis = 0; axis < dims.size(); ++axis)
        {
            fromOffset += index[axis] * sourceStrides[axis];
            toOffset += index[axis] * destinationStrides[axis];
        }
        std::memcpy(to + static_cast<size_t>(toOffset) * elementSize,
                    from + static_cast<size_t>(fromOffset) * elementSize,
                    elementSize);
        for(size_t axis = dims.size(); axis-- > 0;)
        {
            if(++index[axis] < dims[axis])
            {
                break;
            }
            index[axis] = 0;
        }
    }
}

} // namespace hip_kernel_provider::sdpa_determinism

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENABLE_SDPA)

#include <memory>
#include <optional>
#include <string>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

#include "../kernel_ingestor_engine/FlydslSdpaFrontendGraphs.hpp"

namespace hip_kernel_provider::sdpa_determinism
{

namespace ingestor = hip_kernel_provider::kernel_ingestor_engine::integration;

/// The three scalar descale tensors of an FP8 graph.
inline constexpr int64_t DESCALE_Q_UID = 15;
inline constexpr int64_t DESCALE_K_UID = 16;
inline constexpr int64_t DESCALE_V_UID = 17;

/// The element encoding a frontend data type has in the suite.
inline ElementType elementTypeOf(hipdnn_frontend::DataType dataType)
{
    switch(dataType)
    {
    case hipdnn_frontend::DataType::FLOAT:
        return ElementType::FLOAT32;
    case hipdnn_frontend::DataType::BFLOAT16:
        return ElementType::BFLOAT16;
    case hipdnn_frontend::DataType::HALF:
        return ElementType::HALF;
    case hipdnn_frontend::DataType::FP8_E4M3_FNUZ:
        return ElementType::FP8_E4M3_FNUZ;
    default:
        break;
    }
    throw std::invalid_argument("elementTypeOf: the suite has no encoding for this data type");
}

/// The data types of a forward graph: Q, K and V in qkvType, O in oType, and, when
/// descaleType is set, the three per-tensor descales of an FP8 graph (one element each).
struct DeterminismDtypes
{
    hipdnn_frontend::DataType qkvType = hipdnn_frontend::DataType::BFLOAT16;
    hipdnn_frontend::DataType oType = hipdnn_frontend::DataType::BFLOAT16;
    std::optional<hipdnn_frontend::DataType> descaleType;

    DeterminismDtypes() = default;

    /// The parameters are the members in order; the descale type may be omitted.
    DeterminismDtypes(hipdnn_frontend::DataType qkv,
                      hipdnn_frontend::DataType output,
                      std::optional<hipdnn_frontend::DataType> descale = std::nullopt)
        : qkvType(qkv)
        , oType(output)
        , descaleType(descale)
    {
    }
};

/// A forward graph the 16-bit builder cannot build. `shape.dataType` is ignored: the
/// dtypes decide. The descale values are distinct, fixed and positive.
struct DeterminismFwdSpec
{
    ingestor::FrontendSdpaSpec shape;
    DeterminismDtypes dtypes;
    std::array<float, 3> descales = {0.5F, 0.75F, 1.25F};
};

struct DeterminismFwdGraph
{
    ingestor::FrontendSdpaGraph sdpa;
    /// Q, K and V descales; null without descaleType.
    std::array<std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>, 3> descales;
};

/// Builds @p determinism's graph through the frontend Graph API as buildFrontendSdpaGraph
/// does, with the dtype contract of the FP8 forward rows (FP8 E4M3 FNUZ Q, K, V with BFLOAT16 O
/// and three scalar FLOAT descales, mma_core_mode unset, no scale_s or descale_s) and of the
/// FLOAT rows. The caller validates and builds it.
inline DeterminismFwdGraph buildDeterminismFwdGraph(const DeterminismFwdSpec& determinism)
{
    using hipdnn_frontend::DataType;
    using namespace ingestor::frontend_sdpa_detail;
    const auto& spec = determinism.shape;
    const auto& dtypes = determinism.dtypes;
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name(spec.name)
        .set_io_data_type(dtypes.oType)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    using ingestor::detail::makeSdpaTensor;
    using Layout = kernel_ingestor_engine::testing::SdpaLayout;
    auto q = makeSdpaTensor(Q_UID,
                            "Q",
                            dtypes.qkvType,
                            spec.batch,
                            spec.queryHeads,
                            spec.seqQ,
                            spec.headSize,
                            Layout::BSHD);
    auto k = makeSdpaTensor(K_UID,
                            "K",
                            dtypes.qkvType,
                            spec.batch,
                            spec.kvHeads,
                            spec.seqKv,
                            spec.headSize,
                            Layout::BSHD);
    const int64_t valueHeadSize = spec.valueHeadSize.value_or(spec.headSize);
    auto v = makeSdpaTensor(V_UID,
                            "V",
                            dtypes.qkvType,
                            spec.batch,
                            spec.valueHeads.value_or(spec.kvHeads),
                            spec.seqKv,
                            valueHeadSize,
                            Layout::BSHD);

    DeterminismFwdGraph result;
    result.sdpa.spec = spec;
    const bool packed = !spec.packedLensQ.empty();
    if(packed)
    {
        result.sdpa.offsetsQ
            = packedTensor(OFFSETS_Q_UID, "OffsetsQ", DataType::INT32, {spec.batch + 1, 1, 1, 1});
        result.sdpa.offsetsKv
            = packedTensor(OFFSETS_KV_UID, "OffsetsKv", DataType::INT32, {spec.batch + 1, 1, 1, 1});
        q->set_ragged_offset(result.sdpa.offsetsQ)
            .set_ragged_offset_multiplier(spec.queryHeads * spec.headSize);
        k->set_ragged_offset(result.sdpa.offsetsKv)
            .set_ragged_offset_multiplier(spec.kvHeads * spec.headSize);
        v->set_ragged_offset(result.sdpa.offsetsKv)
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

    result.sdpa.seqLensQ = spec.seqLensQ;
    result.sdpa.seqLensKv = spec.seqLensKv;
    result.sdpa.dropout = spec.dropout;
    if(!spec.seqLensQ.empty() || !spec.seqLensKv.empty())
    {
        if(!spec.seqLensQ.empty())
        {
            result.sdpa.seqLenQ
                = packedTensor(SEQ_LEN_Q_UID, "SeqLenQ", DataType::INT32, {spec.batch, 1, 1, 1});
            attributes.set_seq_len_q(result.sdpa.seqLenQ);
        }
        result.sdpa.seqLenKv
            = packedTensor(SEQ_LEN_KV_UID, "SeqLenKv", DataType::INT32, {spec.batch, 1, 1, 1});
        attributes.set_seq_len_kv(result.sdpa.seqLenKv);
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
        result.sdpa.seed = philoxTensor(SEED_UID, "Seed", dropout.seed, dropout.onDevice);
        result.sdpa.offset = philoxTensor(OFFSET_UID, "Offset", dropout.offset, dropout.onDevice);
        attributes.set_dropout(dropout.probability, result.sdpa.seed, result.sdpa.offset);
        if(dropout.withRngDump)
        {
            result.sdpa.rngDump
                = packedTensor(RNG_DUMP_UID,
                               "RngDump",
                               DataType::FLOAT,
                               {spec.batch, spec.queryHeads, spec.seqQ, spec.seqKv});
            result.sdpa.rngDump->set_output(true);
            attributes.set_rng_dump(result.sdpa.rngDump);
        }
    }
    if(spec.sinkDataType.has_value())
    {
        result.sdpa.sink = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        result.sdpa.sink->set_uid(SINK_UID)
            .set_name("Sink")
            .set_dim({1, spec.queryHeads, 1, 1})
            .set_stride({spec.queryHeads, 1, 1, 1})
            .set_data_type(*spec.sinkDataType);
        attributes.set_sink_token(result.sdpa.sink);
    }
    if(dtypes.descaleType.has_value())
    {
        const std::array<int64_t, 3> uids = {DESCALE_Q_UID, DESCALE_K_UID, DESCALE_V_UID};
        const std::array<const char*, 3> names = {"DescaleQ", "DescaleK", "DescaleV"};
        for(size_t i = 0; i < uids.size(); ++i)
        {
            result.descales[i] = packedTensor(uids[i], names[i], *dtypes.descaleType, {1, 1, 1, 1});
        }
        attributes.set_descale_q(result.descales[0]);
        attributes.set_descale_k(result.descales[1]);
        attributes.set_descale_v(result.descales[2]);
    }

    auto outputs = graph->sdpa(q, k, v, attributes);
    result.sdpa.graph = graph;
    result.sdpa.output = outputs[0];
    result.sdpa.output->set_uid(O_UID)
        .set_name("O")
        .set_output(true)
        .set_data_type(dtypes.oType)
        .set_dim({spec.batch, spec.queryHeads, spec.seqQ, valueHeadSize})
        .set_stride(kernel_ingestor_engine::testing::sdpaStrides(
            Layout::BSHD, spec.queryHeads, spec.seqQ, valueHeadSize));
    if(packed)
    {
        result.sdpa.output->set_ragged_offset(result.sdpa.offsetsQ)
            .set_ragged_offset_multiplier(spec.queryHeads * valueHeadSize);
    }
    if(spec.withStats)
    {
        result.sdpa.stats = outputs[1];
        result.sdpa.stats->set_uid(STATS_UID)
            .set_name("Stats")
            .set_output(true)
            .set_data_type(DataType::FLOAT)
            .set_dim({spec.batch, spec.queryHeads, spec.seqQ, 1})
            .set_stride({spec.queryHeads * spec.seqQ, spec.seqQ, 1, 1});
        if(packed)
        {
            result.sdpa.stats->set_stride({spec.seqQ * spec.queryHeads, 1, spec.queryHeads, 1})
                .set_ragged_offset(result.sdpa.offsetsQ)
                .set_ragged_offset_multiplier(spec.queryHeads);
        }
    }
    return result;
}

} // namespace hip_kernel_provider::sdpa_determinism

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENABLE_SDPA
