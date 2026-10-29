// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime_api.h>

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/types/Half.hpp>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/engine_details_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/EngineConfigWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/ingestor/GenericPlan.hpp>
#include <hipdnn_test_sdk/utilities/MockEngineConfig.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "core/Container.hpp"
#include "core/Context.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslNormDeviceTestSupport.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdTestGraphs.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaObjectTestSupport.hpp"

/**
 * @file FlydslSdpaBwdDeviceHarness.hpp
 * @brief The SDPA backward device harness: one graph described by a BwdCase, executed through
 *        the engine manager and compared with a double-precision backward pass.
 *
 * Each run builds the graph, asks the manager whether the engine under test claims it, builds
 * that engine's execution plan (the manager takes the engine by id, so no other engine can serve
 * the graph) and executes it on deterministic inputs. dQ, dK, dV and, on a bias graph that asks
 * for it, dBias are compared with a backward pass over the same inputs: the forward output and
 * the log-sum-exp the kernels consume are computed by the reference from those inputs, with the
 * same dropout mask, and rounded to the graph's dtype. A row no key reaches has O = 0 and
 * LSE = -inf (the forward engines' convention), and its dQ row, and its share of dK, dV and
 * dBias, is zero.
 *
 * A case describes the shape, the mask (a causal diagonal at either corner and a window), the
 * bias (and the values that mask whole rows), the head dims of Q/K and V, the scale (a value, a
 * pass-by-value tensor read at execute, or none) and the layout every tensor is read and
 * written through (head-major, sequence-major, packed QKV views). The reference works on dense
 * logical (B, H, S, D) tensors; every device tensor is scattered into, and gathered from, its
 * own strides.
 *
 * An ObjectPin selects one staged object of one slot of the split engine through the engine
 * configuration, the way hipdnn_bench --sweep does: the candidates the engine enumerates for
 * the graph carry each kernel's knob settings, and the settings of the pinned object's slot go
 * into the configuration the plan is built from. The run then checks that the plan built that
 * object.
 *
 * Tolerances are relative to the largest reference element of each gradient: a kernel that is
 * wrong, rather than rounded, misses by a good fraction of it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device
{

using hipdnn_flatbuffers_sdk::data_objects::DataType;

namespace uid = hip_kernel_provider::kernel_ingestor_engine::testing::sdpa_bwd_uid;

inline constexpr std::string_view MERGED_ENGINE = "hipkernel:flydsl_sdpa_bwd_merged";
inline constexpr std::string_view SPLIT_ENGINE = "hipkernel:flydsl_sdpa_bwd_split";

/// The ops whose objects an engine launches; the split engine launches one per slot.
inline const std::vector<std::string_view> MERGED_OPS{"sdpa_bwd_merged"};
inline const std::vector<std::string_view> SPLIT_OPS{
    "sdpa_bwd_prep", "sdpa_bwd_dkdv", "sdpa_bwd_dq"};

// ---------------------------------------------------------------------------
// Cases
// ---------------------------------------------------------------------------

/// The causal diagonal of a mask.
enum class BwdMask
{
    NONE,
    TOP_LEFT,
    BOTTOM_RIGHT
};

/// Where the softmax scale comes from.
enum class BwdScale
{
    /// attn_scale_value of 1 / sqrt(Dqk).
    VALUE,
    /// A pass-by-value tensor with no baked value, supplied at execute.
    RUNTIME_TENSOR,
    /// The graph states none: the scale is 1.
    NONE
};

/// How the tensors are laid out in memory.
enum class BwdLayout
{
    /// (B, H, S, D) with rows a whole vector apart.
    HEAD_MAJOR,
    /// (B, S, H, D): every data tensor sequence-major, the log-sum-exp (B, S, H).
    SEQ_MAJOR,
    /// Q, K, dQ, dK head-major; V, O, dO, dV sequence-major; the log-sum-exp (B, S, H).
    SEQ_MAJOR_VALUE,
    /// Q, K, V views into one (B, S, 3, H, D) buffer and dQ, dK, dV into another.
    PACKED_QKV
};

/// The values of a bias.
enum class BwdBiasFill
{
    VALUES,
    /// Every key of every third row at the lowest finite value of the bias's dtype.
    LOWEST_ROWS,
    /// Every key of every third row at -inf: a row no key reaches.
    NEG_INF_ROWS,
    /// The first `biasLeadingKeys` keys of every row at -inf.
    NEG_INF_LEADING_KEYS
};

/// What a graph computes: its shape and the features it turns on.
struct BwdCase
{
    DataType dtype = DataType::HALF;
    int64_t batch = 2;
    int64_t heads = 4;
    int64_t kvHeads = 4;
    int64_t seqQ = 96;
    int64_t seqKv = 96;
    int64_t headDim = 64;
    /// V's head dim; 0 is Q/K's.
    int64_t headDimV = 0;
    BwdMask mask = BwdMask::NONE;
    /// cuDNN-convention bounds: key j is seen from query row r when
    /// r + off - left < j <= r + off + right, off being Skv - Sq at a bottom-right diagonal;
    /// left L keeps the L keys ending at the diagonal, the diagonal included.
    /// -1 is unbounded, except that a causal mask without a right bound has right 0.
    int64_t windowLeft = -1;
    int64_t windowRight = -1;
    bool bias = false;
    /// With a bias: whether the graph asks for its gradient, and which axes the bias (and so
    /// its gradient) broadcasts over.
    bool dbias = true;
    bool biasOverBatch = false;
    bool biasOverHead = false;
    BwdBiasFill biasFill = BwdBiasFill::VALUES;
    int64_t biasLeadingKeys = 40;
    /// Per-batch valid lengths of a padded varlen graph; empty for a dense one.
    std::vector<int32_t> lengthsQ;
    std::vector<int32_t> lengthsKv;
    float dropout = 0.0F;
    int64_t seed = 0x1234567;
    int64_t offset = 77;
    BwdScale scaleForm = BwdScale::VALUE;
    /// The value of a RUNTIME_TENSOR scale.
    float runtimeScale = 0.07F;
    BwdLayout layout = BwdLayout::HEAD_MAJOR;

    bool padded() const
    {
        return !lengthsQ.empty();
    }

    int64_t valueDim() const
    {
        return headDimV == 0 ? headDim : headDimV;
    }

    int32_t validQ(int64_t batchIndex) const
    {
        return padded() ? lengthsQ[static_cast<size_t>(batchIndex)] : static_cast<int32_t>(seqQ);
    }

    int32_t validKv(int64_t batchIndex) const
    {
        return padded() ? lengthsKv[static_cast<size_t>(batchIndex)] : static_cast<int32_t>(seqKv);
    }

    int64_t biasRowStride() const
    {
        return ceilToVectorOf(dtype, seqKv);
    }

    int64_t biasBatch() const
    {
        return biasOverBatch ? 1 : batch;
    }

    int64_t biasHeads() const
    {
        return biasOverHead ? 1 : heads;
    }

    size_t biasElements() const
    {
        return static_cast<size_t>(biasBatch() * biasHeads() * seqQ * seqKv);
    }

    /// The scale the kernels apply, as the float the graph or the execute call carries.
    double scale() const
    {
        switch(scaleForm)
        {
        case BwdScale::RUNTIME_TENSOR:
            return static_cast<double>(runtimeScale);
        case BwdScale::NONE:
            return 1.0;
        default:
            return static_cast<double>(
                static_cast<float>(1.0 / std::sqrt(static_cast<double>(headDim))));
        }
    }
};

/// One tensor's place in device memory: its logical (B, H, S, D) extents, the element strides
/// of those axes, and where it starts in its buffer group.
struct TensorView
{
    std::array<int64_t, 4> dims{};
    std::array<int64_t, 4> strides{};
    int64_t offset = 0;
    size_t group = 0;

    /// Elements from the group's start to one past the last element this view addresses.
    size_t extent() const
    {
        int64_t last = offset;
        for(size_t axis = 0; axis < 4; ++axis)
        {
            last += (dims[axis] - 1) * strides[axis];
        }
        return static_cast<size_t>(last + 1);
    }
};

inline constexpr size_t GROUP_COUNT = 9;

/// Where every tensor of a case lives. Tensors of one group share a buffer.
struct BwdViews
{
    TensorView q;
    TensorView k;
    TensorView v;
    TensorView o;
    TensorView dO;
    TensorView stats;
    TensorView dq;
    TensorView dk;
    TensorView dv;

    std::array<TensorView*, GROUP_COUNT> all()
    {
        return {&q, &k, &v, &o, &dO, &stats, &dq, &dk, &dv};
    }

    std::array<const TensorView*, GROUP_COUNT> all() const
    {
        return {&q, &k, &v, &o, &dO, &stats, &dq, &dk, &dv};
    }

    /// Elements of each group's buffer.
    std::array<size_t, GROUP_COUNT> groupElements() const
    {
        std::array<size_t, GROUP_COUNT> elements{};
        for(const auto* view : all())
        {
            elements[view->group] = std::max(elements[view->group], view->extent());
        }
        return elements;
    }
};

inline BwdViews viewsOf(const BwdCase& c)
{
    const auto pitchQk = ceilToVectorOf(c.dtype, c.headDim);
    const auto pitchV = ceilToVectorOf(c.dtype, c.valueDim());
    const auto headMajor = [&](int64_t heads, int64_t seq, int64_t width, int64_t pitch) {
        TensorView view;
        view.dims = {c.batch, heads, seq, width};
        view.strides = {heads * seq * pitch, seq * pitch, pitch, 1};
        return view;
    };
    const auto seqMajor = [&](int64_t heads, int64_t seq, int64_t width, int64_t pitch) {
        TensorView view;
        view.dims = {c.batch, heads, seq, width};
        view.strides = {seq * heads * pitch, pitch, heads * pitch, 1};
        return view;
    };

    BwdViews views;
    const bool valueSeqMajor
        = c.layout == BwdLayout::SEQ_MAJOR || c.layout == BwdLayout::SEQ_MAJOR_VALUE;
    const bool qkSeqMajor = c.layout == BwdLayout::SEQ_MAJOR;
    views.q = qkSeqMajor ? seqMajor(c.heads, c.seqQ, c.headDim, pitchQk)
                         : headMajor(c.heads, c.seqQ, c.headDim, pitchQk);
    views.k = qkSeqMajor ? seqMajor(c.kvHeads, c.seqKv, c.headDim, pitchQk)
                         : headMajor(c.kvHeads, c.seqKv, c.headDim, pitchQk);
    views.v = valueSeqMajor ? seqMajor(c.kvHeads, c.seqKv, c.valueDim(), pitchV)
                            : headMajor(c.kvHeads, c.seqKv, c.valueDim(), pitchV);
    views.o = valueSeqMajor ? seqMajor(c.heads, c.seqQ, c.valueDim(), pitchV)
                            : headMajor(c.heads, c.seqQ, c.valueDim(), pitchV);
    views.dO = views.o;
    views.dq = views.q;
    views.dk = views.k;
    views.dv = views.v;

    views.stats.dims = {c.batch, c.heads, c.seqQ, 1};
    views.stats.strides = valueSeqMajor ? std::array<int64_t, 4>{c.seqQ * c.heads, 1, c.heads, 1}
                                        : std::array<int64_t, 4>{c.heads * c.seqQ, c.seqQ, 1, 1};

    if(c.layout == BwdLayout::PACKED_QKV)
    {
        if(c.heads != c.kvHeads || c.seqQ != c.seqKv || c.headDim != c.valueDim())
        {
            throw std::invalid_argument(
                "a packed QKV case needs equal head counts, lengths and head dims");
        }
        const auto slab = c.heads * pitchQk;
        for(auto* view : {&views.q, &views.k, &views.v})
        {
            view->strides = {c.seqQ * 3 * slab, pitchQk, 3 * slab, 1};
        }
        views.k.offset = slab;
        views.v.offset = 2 * slab;
        views.dq = views.q;
        views.dk = views.k;
        views.dv = views.v;
    }

    // Every tensor has a buffer of its own except the packed ones, which share per direction.
    const auto groups = std::array<TensorView*, GROUP_COUNT>{&views.q,
                                                             &views.k,
                                                             &views.v,
                                                             &views.o,
                                                             &views.dO,
                                                             &views.stats,
                                                             &views.dq,
                                                             &views.dk,
                                                             &views.dv};
    for(size_t index = 0; index < groups.size(); ++index)
    {
        groups[index]->group = index;
    }
    if(c.layout == BwdLayout::PACKED_QKV)
    {
        views.k.group = views.q.group;
        views.v.group = views.q.group;
        views.dk.group = views.dq.group;
        views.dv.group = views.dq.group;
    }
    return views;
}

/// The dense bias's place in its own buffer: (Bb, Hb, Sq, Skv), rows a whole vector apart.
inline TensorView biasViewOf(const BwdCase& c)
{
    const auto pitch = c.biasRowStride();
    TensorView view;
    view.dims = {c.biasBatch(), c.biasHeads(), c.seqQ, c.seqKv};
    view.strides = {c.biasHeads() * c.seqQ * pitch, c.seqQ * pitch, pitch, 1};
    return view;
}

inline SdpaBwdSpec specOf(const BwdCase& c)
{
    SdpaBwdSpec spec;
    spec.dtype = c.dtype;
    spec.batch = c.batch;
    spec.heads = c.heads;
    spec.kvHeads = c.kvHeads;
    spec.seqQ = c.seqQ;
    spec.seqKv = c.seqKv;
    spec.headDimQk = c.headDim;
    spec.headDimV = c.valueDim();
    const auto views = viewsOf(c);
    spec = spec.withEdit([=](SdpaBwdGraph& g) {
        g.tensor("q").strides.assign(views.q.strides.begin(), views.q.strides.end());
        g.tensor("k").strides.assign(views.k.strides.begin(), views.k.strides.end());
        g.tensor("v").strides.assign(views.v.strides.begin(), views.v.strides.end());
        g.tensor("o").strides.assign(views.o.strides.begin(), views.o.strides.end());
        g.tensor("do").strides.assign(views.dO.strides.begin(), views.dO.strides.end());
        g.tensor("stats").strides.assign(views.stats.strides.begin(), views.stats.strides.end());
        g.tensor("dq").strides.assign(views.dq.strides.begin(), views.dq.strides.end());
        g.tensor("dk").strides.assign(views.dk.strides.begin(), views.dk.strides.end());
        g.tensor("dv").strides.assign(views.dv.strides.begin(), views.dv.strides.end());
    });
    if(c.scaleForm == BwdScale::VALUE)
    {
        const auto scale = static_cast<float>(c.scale());
        spec = spec.withEdit([=](SdpaBwdGraph& g) { g.attributes.attnScaleValue = scale; });
    }
    else if(c.scaleForm == BwdScale::RUNTIME_TENSOR)
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { addScaleTensor(g, std::nullopt); });
    }
    if(c.mask == BwdMask::TOP_LEFT)
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMask = true; });
    }
    else if(c.mask == BwdMask::BOTTOM_RIGHT)
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { g.attributes.causalMaskBottomRight = true; });
    }
    if(c.windowLeft >= 0 || c.windowRight >= 0)
    {
        const auto left = c.windowLeft;
        const auto right = c.windowRight;
        spec = spec.withEdit([=](SdpaBwdGraph& g) {
            if(left >= 0)
            {
                g.attributes.leftBound = left;
            }
            if(right >= 0)
            {
                g.attributes.rightBound = right;
            }
        });
    }
    if(c.bias)
    {
        const auto overBatch = c.biasOverBatch;
        const auto overHead = c.biasOverHead;
        const auto withDbias = c.dbias;
        spec = spec.withEdit(
            [=](SdpaBwdGraph& g) { addBroadcastBias(g, overBatch, overHead, withDbias); });
    }
    if(c.padded())
    {
        spec = spec.withEdit([](SdpaBwdGraph& g) { addSeqLens(g); });
    }
    if(c.dropout > 0.0F)
    {
        const auto probability = c.dropout;
        spec = spec.withEdit(
            [=](SdpaBwdGraph& g) { addDropout(g, probability, DataType::INT64, false); });
    }
    return spec;
}

// ---------------------------------------------------------------------------
// Host reference
// ---------------------------------------------------------------------------

/// Deterministic k/32 values for |k| <= 32: exact in every dtype under test, so the reference
/// reads exactly what the device does.
inline std::vector<double> fillValues(size_t count, uint32_t seed)
{
    std::vector<double> values(count);
    uint32_t state = seed;
    for(auto& value : values)
    {
        state = state * 1664525U + 1013904223U;
        value = static_cast<double>(static_cast<int32_t>((state >> 16) % 65U) - 32) / 32.0;
    }
    return values;
}

/// The Philox-4x32-10 block the kernels draw a dropout mask from (Triton's round).
inline std::array<uint32_t, 4> philox4x32(uint64_t seed, uint64_t offset)
{
    constexpr uint32_t MUL_A = 0xD2511F53U;
    constexpr uint32_t MUL_B = 0xCD9E8D57U;
    constexpr uint32_t KEY_A = 0x9E3779B9U;
    constexpr uint32_t KEY_B = 0xBB67AE85U;
    auto c0 = static_cast<uint32_t>(offset);
    auto c1 = static_cast<uint32_t>(offset >> 32U);
    uint32_t c2 = 0;
    uint32_t c3 = 0;
    auto k0 = static_cast<uint32_t>(seed);
    auto k1 = static_cast<uint32_t>(seed >> 32U);
    for(int round = 0; round < 10; ++round)
    {
        const uint32_t previous0 = c0;
        const uint32_t previous2 = c2;
        const uint64_t productB = static_cast<uint64_t>(MUL_B) * previous2;
        const uint64_t productA = static_cast<uint64_t>(MUL_A) * previous0;
        c0 = static_cast<uint32_t>(productB >> 32U) ^ c1 ^ k0;
        c2 = static_cast<uint32_t>(productA >> 32U) ^ c3 ^ k1;
        c1 = static_cast<uint32_t>(productB);
        c3 = static_cast<uint32_t>(productA);
        k0 += KEY_A;
        k1 += KEY_B;
    }
    return {c0, c1, c2, c3};
}

/// Whether the kernels keep score (batch, head, row, column): one Philox block per four
/// columns of a row, a plane per (batch, query head), `ceil(Skv / 4)` blocks per row.
inline bool keeps(const BwdCase& c, int64_t batchIndex, int64_t head, int64_t row, int64_t column)
{
    if(c.dropout <= 0.0F)
    {
        return true;
    }
    const int64_t rowBlocks = (c.seqKv + 3) / 4;
    const int64_t plane = batchIndex * c.heads + head;
    const auto offset
        = static_cast<uint64_t>(c.offset + (plane * c.seqQ + row) * rowBlocks + column / 4);
    const auto block = philox4x32(static_cast<uint64_t>(c.seed), offset);
    const auto draw = static_cast<int32_t>(block[static_cast<size_t>(column % 4)]);
    const double threshold = (static_cast<double>(c.dropout) - 0.5) * 4294967295.0;
    const auto clamped = static_cast<int32_t>(std::clamp(threshold, -2147483648.0, 2147483647.0));
    return draw > clamped;
}

inline double dropoutScale(const BwdCase& c)
{
    return c.dropout > 0.0F ? static_cast<double>(
                                  static_cast<float>(1.0 / (1.0 - static_cast<double>(c.dropout))))
                            : 1.0;
}

/// Every operand the reference and the device share, as the device will read it, in dense
/// logical (B, H, S, D) order.
struct Inputs
{
    std::vector<double> q;
    std::vector<double> k;
    std::vector<double> v;
    std::vector<double> dO;
    std::vector<double> bias;
};

inline size_t
    at(int64_t b, int64_t h, int64_t s, int64_t e, int64_t heads, int64_t seq, int64_t width)
{
    return static_cast<size_t>(((b * heads + h) * seq + s) * width + e);
}

/// Index of a score's bias in the dense (Bb, Hb, Sq, Skv) bias, whose broadcast axes have
/// extent 1.
inline size_t biasAt(const BwdCase& c, int64_t b, int64_t h, int64_t s, int64_t e)
{
    return at(
        c.biasOverBatch ? 0 : b, c.biasOverHead ? 0 : h, s, e, c.biasHeads(), c.seqQ, c.seqKv);
}

/// Whether query row @p row may see key @p column.
inline bool visible(const BwdCase& c, int64_t batchIndex, int64_t row, int64_t column)
{
    if(column >= c.validKv(batchIndex))
    {
        return false;
    }
    const int64_t left = c.windowLeft;
    int64_t right = c.windowRight;
    if(c.mask != BwdMask::NONE && right < 0)
    {
        right = 0;
    }
    const int64_t diagonal = row + (c.mask == BwdMask::BOTTOM_RIGHT ? c.seqKv - c.seqQ : 0);
    if(left >= 0 && column <= diagonal - left)
    {
        return false;
    }
    return right < 0 || column <= diagonal + right;
}

/// Score of one (row, key) pair: the scaled dot product plus the bias; -inf where the bias is.
inline double scoreOf(const BwdCase& c,
                      const Inputs& in,
                      int64_t batchIndex,
                      int64_t head,
                      int64_t row,
                      int64_t column)
{
    const int64_t kvHead = head / (c.heads / c.kvHeads);
    double dot = 0.0;
    for(int64_t e = 0; e < c.headDim; ++e)
    {
        dot += in.q[at(batchIndex, head, row, e, c.heads, c.seqQ, c.headDim)]
               * in.k[at(batchIndex, kvHead, column, e, c.kvHeads, c.seqKv, c.headDim)];
    }
    double score = c.scale() * dot;
    if(c.bias)
    {
        score += in.bias[biasAt(c, batchIndex, head, row, column)];
    }
    return score;
}

/// What the forward pass hands the backward: its output and the log-sum-exp of each row.
struct Forward
{
    std::vector<double> o;
    std::vector<float> lse;
};

inline Forward forwardOf(const BwdCase& c, const Inputs& in)
{
    const int64_t dv = c.valueDim();
    Forward out;
    out.o.assign(static_cast<size_t>(c.batch * c.heads * c.seqQ * dv), 0.0);
    out.lse.assign(static_cast<size_t>(c.batch * c.heads * c.seqQ),
                   -std::numeric_limits<float>::infinity());
    const double dropScale = dropoutScale(c);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        for(int64_t h = 0; h < c.heads; ++h)
        {
            const int64_t kvHead = h / (c.heads / c.kvHeads);
            for(int64_t row = 0; row < c.validQ(b); ++row)
            {
                std::vector<double> scores(static_cast<size_t>(c.seqKv), 0.0);
                double maximum = -std::numeric_limits<double>::infinity();
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(visible(c, b, row, j))
                    {
                        scores[static_cast<size_t>(j)] = scoreOf(c, in, b, h, row, j);
                        maximum = std::max(maximum, scores[static_cast<size_t>(j)]);
                    }
                }
                if(maximum == -std::numeric_limits<double>::infinity())
                {
                    // No key reaches the row: O = 0 and LSE = -inf.
                    continue;
                }
                double sum = 0.0;
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(visible(c, b, row, j))
                    {
                        sum += std::exp(scores[static_cast<size_t>(j)] - maximum);
                    }
                }
                const double lse = maximum + std::log(sum);
                out.lse[static_cast<size_t>((b * c.heads + h) * c.seqQ + row)]
                    = static_cast<float>(lse);
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(!visible(c, b, row, j) || !keeps(c, b, h, row, j))
                    {
                        continue;
                    }
                    const double weight
                        = std::exp(scores[static_cast<size_t>(j)] - lse) * dropScale;
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        out.o[at(b, h, row, e, c.heads, c.seqQ, dv)]
                            += weight * in.v[at(b, kvHead, j, e, c.kvHeads, c.seqKv, dv)];
                    }
                }
            }
        }
    }
    return out;
}

/// The gradients a backward pass produces, each in dense logical (B, H, S, D) order.
struct Gradients
{
    std::vector<double> dq;
    std::vector<double> dk;
    std::vector<double> dv;
    std::vector<double> dbias;
};

/// The backward pass in double precision, from the (rounded) output and statistics the
/// device is given. dBias is dS, the gradient of the score before its scale.
inline Gradients backwardOf(const BwdCase& c,
                            const Inputs& in,
                            const std::vector<double>& o,
                            const std::vector<float>& lse)
{
    const int64_t dv = c.valueDim();
    Gradients grads;
    grads.dq.assign(in.q.size(), 0.0);
    grads.dk.assign(in.k.size(), 0.0);
    grads.dv.assign(in.v.size(), 0.0);
    grads.dbias.assign(in.bias.size(), 0.0);
    const double dropScale = dropoutScale(c);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        for(int64_t h = 0; h < c.heads; ++h)
        {
            const int64_t kvHead = h / (c.heads / c.kvHeads);
            for(int64_t row = 0; row < c.validQ(b); ++row)
            {
                const auto rowLse = static_cast<double>(
                    lse[static_cast<size_t>((b * c.heads + h) * c.seqQ + row)]);
                if(rowLse == -std::numeric_limits<double>::infinity())
                {
                    // A row no key reaches contributes nothing.
                    continue;
                }
                double delta = 0.0;
                for(int64_t e = 0; e < dv; ++e)
                {
                    const auto index = at(b, h, row, e, c.heads, c.seqQ, dv);
                    delta += in.dO[index] * o[index];
                }
                for(int64_t j = 0; j < c.seqKv; ++j)
                {
                    if(!visible(c, b, row, j))
                    {
                        continue;
                    }
                    const double score = scoreOf(c, in, b, h, row, j);
                    if(score == -std::numeric_limits<double>::infinity())
                    {
                        continue;
                    }
                    const double p = std::exp(score - rowLse);
                    double dp = 0.0;
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        dp += in.dO[at(b, h, row, e, c.heads, c.seqQ, dv)]
                              * in.v[at(b, kvHead, j, e, c.kvHeads, c.seqKv, dv)];
                    }
                    const bool kept = keeps(c, b, h, row, j);
                    const double ds = p * ((kept ? dp * dropScale : 0.0) - delta);
                    const double pDropped = kept ? p * dropScale : 0.0;
                    for(int64_t e = 0; e < c.headDim; ++e)
                    {
                        const auto qIndex = at(b, h, row, e, c.heads, c.seqQ, c.headDim);
                        const auto kIndex = at(b, kvHead, j, e, c.kvHeads, c.seqKv, c.headDim);
                        grads.dq[qIndex] += c.scale() * ds * in.k[kIndex];
                        grads.dk[kIndex] += c.scale() * ds * in.q[qIndex];
                    }
                    for(int64_t e = 0; e < dv; ++e)
                    {
                        grads.dv[at(b, kvHead, j, e, c.kvHeads, c.seqKv, dv)]
                            += pDropped * in.dO[at(b, h, row, e, c.heads, c.seqQ, dv)];
                    }
                    if(c.bias)
                    {
                        // A broadcast axis sums the scores it repeats over.
                        grads.dbias[biasAt(c, b, h, row, j)] += ds;
                    }
                }
            }
        }
    }
    return grads;
}

/// The bias's values: k/32 noise, then what @p c's fill overwrites with a mask value. The
/// masked values are written in the bias's dtype, so @p lowest is that dtype's lowest finite
/// value.
inline std::vector<double> biasValues(const BwdCase& c, double lowest)
{
    auto values = fillValues(c.biasElements(), 5U);
    if(c.biasFill == BwdBiasFill::VALUES)
    {
        return values;
    }
    const double infinity = std::numeric_limits<double>::infinity();
    for(int64_t b = 0; b < c.biasBatch(); ++b)
    {
        for(int64_t h = 0; h < c.biasHeads(); ++h)
        {
            for(int64_t row = 0; row < c.seqQ; ++row)
            {
                for(int64_t column = 0; column < c.seqKv; ++column)
                {
                    auto& value = values[at(b, h, row, column, c.biasHeads(), c.seqQ, c.seqKv)];
                    switch(c.biasFill)
                    {
                    case BwdBiasFill::LOWEST_ROWS:
                        value = row % 3 == 0 ? lowest : value;
                        break;
                    case BwdBiasFill::NEG_INF_ROWS:
                        value = row % 3 == 0 ? -infinity : value;
                        break;
                    default:
                        value = column < c.biasLeadingKeys ? -infinity : value;
                        break;
                    }
                }
            }
        }
    }
    return values;
}

// ---------------------------------------------------------------------------
// Device data
// ---------------------------------------------------------------------------

/// Device memory freed on every exit, including a failed ASSERT part-way through a case.
class DeviceBuffer
{
public:
    explicit DeviceBuffer(size_t bytes)
        : _bytes(bytes)
    {
        EXPECT_EQ(hipSuccess, hipMalloc(&_ptr, bytes));
    }

    ~DeviceBuffer()
    {
        static_cast<void>(hipFree(_ptr));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    DeviceBuffer& operator=(DeviceBuffer&&) = delete;

    void* get() const
    {
        return _ptr;
    }

    template <typename T>
    void upload(const std::vector<T>& values)
    {
        ASSERT_EQ(values.size() * sizeof(T), _bytes);
        ASSERT_EQ(hipSuccess, hipMemcpy(_ptr, values.data(), _bytes, hipMemcpyHostToDevice));
    }

    /// All-ones bytes: NaN in f16, bf16 and f32, so an element no kernel wrote fails the
    /// comparison instead of reading as a zero.
    void poison()
    {
        ASSERT_EQ(hipSuccess, hipMemset(_ptr, 0xFF, _bytes));
    }

    template <typename T>
    std::vector<T> download() const
    {
        std::vector<T> values(_bytes / sizeof(T));
        EXPECT_EQ(hipSuccess, hipMemcpy(values.data(), _ptr, _bytes, hipMemcpyDeviceToHost));
        return values;
    }

private:
    void* _ptr = nullptr;
    size_t _bytes;
};

template <typename T>
std::vector<T> roundedTo(const std::vector<double>& values)
{
    std::vector<T> rounded(values.size());
    std::transform(values.begin(), values.end(), rounded.begin(), [](double value) {
        return T(static_cast<float>(value));
    });
    return rounded;
}

template <typename T>
std::vector<double> widened(const std::vector<T>& values)
{
    std::vector<double> wide(values.size());
    std::transform(values.begin(), values.end(), wide.begin(), [](const T& value) {
        return static_cast<double>(static_cast<float>(value));
    });
    return wide;
}

/// Writes the dense logical tensor @p dense, rounded to @p T, into @p buffer through @p view.
/// The elements the view does not address keep their value.
template <typename T>
void scatter(std::vector<T>& buffer, const std::vector<double>& dense, const TensorView& view)
{
    size_t next = 0;
    for(int64_t b = 0; b < view.dims[0]; ++b)
    {
        for(int64_t h = 0; h < view.dims[1]; ++h)
        {
            for(int64_t s = 0; s < view.dims[2]; ++s)
            {
                for(int64_t e = 0; e < view.dims[3]; ++e)
                {
                    const auto index = view.offset + b * view.strides[0] + h * view.strides[1]
                                       + s * view.strides[2] + e * view.strides[3];
                    buffer[static_cast<size_t>(index)] = T(static_cast<float>(dense[next++]));
                }
            }
        }
    }
}

/// The inverse of scatter: the dense logical tensor @p view addresses in @p buffer, widened.
template <typename T>
std::vector<double> gather(const std::vector<T>& buffer, const TensorView& view)
{
    std::vector<double> dense;
    dense.reserve(static_cast<size_t>(view.dims[0] * view.dims[1] * view.dims[2] * view.dims[3]));
    for(int64_t b = 0; b < view.dims[0]; ++b)
    {
        for(int64_t h = 0; h < view.dims[1]; ++h)
        {
            for(int64_t s = 0; s < view.dims[2]; ++s)
            {
                for(int64_t e = 0; e < view.dims[3]; ++e)
                {
                    const auto index = view.offset + b * view.strides[0] + h * view.strides[1]
                                       + s * view.strides[2] + e * view.strides[3];
                    dense.push_back(static_cast<double>(
                        static_cast<float>(buffer[static_cast<size_t>(index)])));
                }
            }
        }
    }
    return dense;
}

/// Relative tolerance of a gradient in @p dtype, against the largest reference element.
inline double toleranceOf(DataType dtype)
{
    switch(dtype)
    {
    case DataType::FLOAT:
        return 4e-3;
    case DataType::HALF:
        return 2e-2;
    default:
        return 4e-2;
    }
}

/// Compares @p got with @p expected over the elements @p inScope names; an unwritten
/// (NaN) element is a miss.
inline void expectGradient(const char* name,
                           const std::vector<double>& got,
                           const std::vector<double>& expected,
                           const std::vector<bool>& inScope,
                           double relativeTolerance)
{
    ASSERT_EQ(got.size(), expected.size());
    double largest = 0.0;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        largest = inScope[i] ? std::max(largest, std::abs(expected[i])) : largest;
    }
    ASSERT_GT(largest, 0.0) << name << ": the reference is all zero, the case checks nothing";
    const double tolerance = relativeTolerance * largest;
    double worst = 0.0;
    size_t worstIndex = 0;
    for(size_t i = 0; i < expected.size(); ++i)
    {
        if(!inScope[i])
        {
            continue;
        }
        const double error = std::isfinite(got[i]) ? std::abs(got[i] - expected[i])
                                                   : std::numeric_limits<double>::infinity();
        if(error > worst)
        {
            worst = error;
            worstIndex = i;
        }
    }
    EXPECT_LE(worst, tolerance) << name << ": worst error at element " << worstIndex << " (got "
                                << got[worstIndex] << ", expected " << expected[worstIndex]
                                << "), against a bound of " << tolerance << " ("
                                << relativeTolerance << " of the largest element " << largest
                                << ")";
}

/// Which elements of a (B, H, S, D) tensor lie inside each batch's valid length.
inline std::vector<bool> scopeOf(const BwdCase& c, bool queryAxis, int64_t heads, int64_t width)
{
    const int64_t seq = queryAxis ? c.seqQ : c.seqKv;
    std::vector<bool> scope(static_cast<size_t>(c.batch * heads * seq * width), false);
    for(int64_t b = 0; b < c.batch; ++b)
    {
        const int64_t valid = queryAxis ? c.validQ(b) : c.validKv(b);
        for(int64_t h = 0; h < heads; ++h)
        {
            for(int64_t s = 0; s < valid; ++s)
            {
                for(int64_t e = 0; e < width; ++e)
                {
                    scope[at(b, h, s, e, heads, seq, width)] = true;
                }
            }
        }
    }
    return scope;
}

// ---------------------------------------------------------------------------
// Engine runs
// ---------------------------------------------------------------------------

inline hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper
    wrap(const flatbuffers::FlatBufferBuilder& builder)
{
    return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(builder.GetBufferPointer(),
                                                                      builder.GetSize());
}

inline int64_t idOf(std::string_view engine)
{
    return hipdnn_data_sdk::utilities::engineNameToId(engine);
}

inline void stubEngineId(hipdnn_test_sdk::utilities::MockEngineConfig& config, int64_t engineId)
{
    EXPECT_CALL(config, isValid()).WillRepeatedly(::testing::Return(false));
    EXPECT_CALL(config, engineId()).WillRepeatedly(::testing::Return(engineId));
}

/// An engine configuration that sets integer knobs.
class IntKnobConfig
{
public:
    using Settings = std::vector<std::pair<std::string, int64_t>>;

    IntKnobConfig(int64_t engineId, const Settings& settings)
        : _wrapper(build(_builder, engineId, settings))
    {
    }

    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& get() const
    {
        return _wrapper;
    }

private:
    static hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper
        build(flatbuffers::FlatBufferBuilder& builder, int64_t engineId, const Settings& settings)
    {
        using namespace hipdnn_flatbuffers_sdk::data_objects;
        std::vector<flatbuffers::Offset<KnobSetting>> knobs;
        knobs.reserve(settings.size());
        for(const auto& [name, value] : settings)
        {
            knobs.push_back(CreateKnobSettingDirect(builder,
                                                    name.c_str(),
                                                    KnobValue::IntValue,
                                                    CreateIntValue(builder, value).Union()));
        }
        builder.Finish(CreateEngineConfig(builder, engineId, builder.CreateVector(knobs)));
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper(
            builder.GetBufferPointer(), builder.GetSize());
    }

    flatbuffers::FlatBufferBuilder _builder;
    hipdnn_flatbuffers_sdk::flatbuffer_utilities::EngineConfigWrapper _wrapper;
};

/// One staged object of one slot of the split engine, and what selects it.
struct ObjectPin
{
    /// The slot the object belongs to: prep, dkdv or dq.
    std::string slot;
    std::string kernelName;
    /// The kernel's descriptor id, as a candidate's composite id spells it.
    std::string kernelId;
    /// `<slot>.<knob>` settings that narrow the slot to the object and its look-alikes: its
    /// integer-valued enrolled knobs.
    IntKnobConfig::Settings narrowing;
    /// The digest of the dBias reduce companion the object launches; empty for none.
    std::string reduceDigest;
};

/// A slot's value of a candidate composite id (`prep=<id>|dkdv=<id>|dq=<id>`).
inline std::string slotIdOf(const std::string& compositeId, const std::string& slot)
{
    const auto prefix = slot + "=";
    size_t begin = 0;
    while(begin <= compositeId.size())
    {
        const auto end = std::min(compositeId.find('|', begin), compositeId.size());
        const auto part = compositeId.substr(begin, end - begin);
        if(part.rfind(prefix, 0) == 0)
        {
            return part.substr(prefix.size());
        }
        begin = end + 1;
    }
    return {};
}

/// The knob settings of @p pin's slot that identify its object among the candidates the engine
/// enumerates for @p graph, the way a sweep takes them; empty (and a failure recorded) when
/// the engine enumerates no candidate that launches the object.
template <typename Manager>
IntKnobConfig::Settings
    pinnedKnobs(Manager& manager,
                Handle& handle,
                const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                int64_t engineId,
                const ObjectPin& pin)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    constexpr uint64_t PAGE = 10000;
    const IntKnobConfig narrowed(engineId, pin.narrowing);
    uint64_t total = 1;
    for(uint64_t offset = 0; offset < total; offset += PAGE)
    {
        hipdnnPluginConstData_t details{};
        manager.enumerateCandidates(handle, graph, narrowed.get(), offset, PAGE, details);
        const auto* page = GetEngineDetails(details.ptr)->candidate_page();
        total = page->total_count();
        if(page->candidates() == nullptr)
        {
            break;
        }
        for(const auto* candidate : *page->candidates())
        {
            if(slotIdOf(candidate->id()->str(), pin.slot) != pin.kernelId)
            {
                continue;
            }
            IntKnobConfig::Settings settings;
            for(const auto* setting : *candidate->knob_settings())
            {
                const auto name = setting->knob_id()->str();
                if(name.rfind(pin.slot + ".", 0) == 0)
                {
                    settings.emplace_back(name, setting->value_as_IntValue()->value());
                }
            }
            return settings;
        }
    }
    ADD_FAILURE() << "the engine enumerates no candidate that launches " << pin.kernelName << " ("
                  << pin.kernelId << ") for the graph; " << total
                  << " candidates match the narrowing of its integer knobs";
    return {};
}

/// Checks that the plan built for @p pin's engine launches the pinned object in its slot.
inline void expectPlanLaunches(const Context& context, const ObjectPin& pin)
{
    const auto* program = dynamic_cast<const hipdnn_plugin_sdk::ingestor::SlotProgramPlan<Handle>*>(
        &context.plan());
    ASSERT_NE(program, nullptr) << "the plan is not a slot program";
    const auto stage
        = std::find_if(program->stages().begin(),
                       program->stages().end(),
                       [&](const auto& candidate) { return candidate.slot == pin.slot; });
    ASSERT_NE(stage, program->stages().end()) << "the plan has no " << pin.slot << " stage";
    const auto& launched = stage->plan->kernel();
    EXPECT_EQ(launched.name, pin.kernelName)
        << "the plan launches another " << pin.slot << " object than the pinned one";
    if(!pin.reduceDigest.empty())
    {
        const auto companion = launched.sources.find("dbias_reduce");
        ASSERT_NE(companion, launched.sources.end())
            << launched.name << " launches no dBias reduce";
        EXPECT_EQ(companion->second.sha256, pin.reduceDigest);
    }
}

/// Executes @p c on @p engine and checks every gradient against the reference. With @p pin,
/// the split engine's plan is built with the object's knob settings and must launch it.
template <typename T>
void expectEngineMatchesReference(std::string_view engine,
                                  const BwdCase& c,
                                  const ObjectPin* pin = nullptr)
{
    const auto builder = buildSdpaBwdGraph(specOf(c));
    const auto graph = wrap(builder);

    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = idOf(engine);

    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    ASSERT_NE(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << engine << " does not claim the graph";

    hipdnn_test_sdk::utilities::MockEngineConfig mockConfig;
    std::unique_ptr<IntKnobConfig> pinnedConfig;
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig* config = &mockConfig;
    if(pin == nullptr)
    {
        stubEngineId(mockConfig, engineId);
    }
    else
    {
        const auto knobs = pinnedKnobs(manager, handle, graph, engineId, *pin);
        ASSERT_FALSE(knobs.empty());
        pinnedConfig = std::make_unique<IntKnobConfig>(engineId, knobs);
        config = &pinnedConfig->get();
    }
    Context context;
    ASSERT_NO_THROW(manager.initializeExecutionContext(handle, graph, *config, context));
    ASSERT_TRUE(context.hasValidPlan());
    if(pin != nullptr)
    {
        expectPlanLaunches(context, *pin);
        if(::testing::Test::HasFatalFailure())
        {
            return;
        }
    }

    // Inputs, and the forward results the backward consumes, rounded to the graph's dtype.
    const int64_t dqk = c.headDim;
    const int64_t dv = c.valueDim();
    Inputs in;
    const auto qCount = static_cast<size_t>(c.batch * c.heads * c.seqQ * dqk);
    const auto kCount = static_cast<size_t>(c.batch * c.kvHeads * c.seqKv * dqk);
    const auto vCount = static_cast<size_t>(c.batch * c.kvHeads * c.seqKv * dv);
    const auto oCount = static_cast<size_t>(c.batch * c.heads * c.seqQ * dv);
    in.q = widened(roundedTo<T>(fillValues(qCount, 1U)));
    in.k = widened(roundedTo<T>(fillValues(kCount, 2U)));
    in.v = widened(roundedTo<T>(fillValues(vCount, 3U)));
    in.dO = widened(roundedTo<T>(fillValues(oCount, 4U)));
    if(c.bias)
    {
        const double lowest = c.dtype == DataType::HALF ? -65504.0
                              : c.dtype == DataType::BFLOAT16
                                  ? -3.3895313892515355e38
                                  : static_cast<double>(std::numeric_limits<float>::lowest());
        in.bias = widened(roundedTo<T>(biasValues(c, lowest)));
    }
    // A padded graph's rows past its lengths carry no data: the gradient's input is zero
    // there, as a packed batch's padding is.
    if(c.padded())
    {
        const auto scope = scopeOf(c, true, c.heads, dv);
        for(size_t i = 0; i < in.dO.size(); ++i)
        {
            in.dO[i] = scope[i] ? in.dO[i] : 0.0;
        }
    }
    const auto forward = forwardOf(c, in);
    const auto oRounded = widened(roundedTo<T>(forward.o));
    const auto expected = backwardOf(c, in, oRounded, forward.lse);

    // Each group's buffer holds the tensors that share it; the elements between a view's
    // rows start as zero.
    const auto views = viewsOf(c);
    const auto elements = views.groupElements();
    std::array<std::vector<T>, GROUP_COUNT> hostData;
    for(size_t group = 0; group < GROUP_COUNT; ++group)
    {
        hostData[group].assign(elements[group], T(0.0F));
    }
    scatter(hostData[views.q.group], in.q, views.q);
    scatter(hostData[views.k.group], in.k, views.k);
    scatter(hostData[views.v.group], in.v, views.v);
    scatter(hostData[views.o.group], oRounded, views.o);
    scatter(hostData[views.dO.group], in.dO, views.dO);
    std::vector<float> statsHost(elements[views.stats.group], 0.0F);
    {
        // The log-sum-exp through its own strides.
        const auto& view = views.stats;
        size_t next = 0;
        for(int64_t b = 0; b < view.dims[0]; ++b)
        {
            for(int64_t h = 0; h < view.dims[1]; ++h)
            {
                for(int64_t s = 0; s < view.dims[2]; ++s)
                {
                    statsHost[static_cast<size_t>(b * view.strides[0] + h * view.strides[1]
                                                  + s * view.strides[2])]
                        = forward.lse[next++];
                }
            }
        }
    }

    std::vector<std::unique_ptr<DeviceBuffer>> device(GROUP_COUNT);
    for(size_t group = 0; group < GROUP_COUNT; ++group)
    {
        const auto bytes
            = elements[group] * (group == views.stats.group ? sizeof(float) : sizeof(T));
        if(bytes > 0)
        {
            device[group] = std::make_unique<DeviceBuffer>(bytes);
        }
    }
    device[views.q.group]->upload(hostData[views.q.group]);
    if(views.k.group != views.q.group)
    {
        device[views.k.group]->upload(hostData[views.k.group]);
    }
    if(views.v.group != views.q.group)
    {
        device[views.v.group]->upload(hostData[views.v.group]);
    }
    device[views.o.group]->upload(hostData[views.o.group]);
    device[views.dO.group]->upload(hostData[views.dO.group]);
    device[views.stats.group]->upload(statsHost);
    for(const auto* view : {&views.dq, &views.dk, &views.dv})
    {
        device[view->group]->poison();
    }
    const auto pointerOf = [&](const TensorView& view) {
        return static_cast<char*>(device[view.group]->get())
               + static_cast<size_t>(view.offset) * sizeof(T);
    };
    std::vector<hipdnnPluginDeviceBuffer_t> buffers{{uid::Q, pointerOf(views.q)},
                                                    {uid::K, pointerOf(views.k)},
                                                    {uid::V, pointerOf(views.v)},
                                                    {uid::O, pointerOf(views.o)},
                                                    {uid::DO, pointerOf(views.dO)},
                                                    {uid::STATS, device[views.stats.group]->get()},
                                                    {uid::DQ, pointerOf(views.dq)},
                                                    {uid::DK, pointerOf(views.dk)},
                                                    {uid::DV, pointerOf(views.dv)}};

    const auto biasView = biasViewOf(c);
    std::vector<T> biasHost;
    std::unique_ptr<DeviceBuffer> bias;
    std::unique_ptr<DeviceBuffer> dbias;
    if(c.bias)
    {
        biasHost.assign(biasView.extent(), T(0.0F));
        scatter(biasHost, in.bias, biasView);
        bias = std::make_unique<DeviceBuffer>(biasHost.size() * sizeof(T));
        bias->upload(biasHost);
        buffers.push_back({uid::BIAS, bias->get()});
        if(c.dbias)
        {
            dbias = std::make_unique<DeviceBuffer>(biasHost.size() * sizeof(T));
            dbias->poison();
            buffers.push_back({uid::DBIAS, dbias->get()});
        }
    }
    DeviceBuffer lengthsQ(c.lengthsQ.size() * sizeof(int32_t));
    DeviceBuffer lengthsKv(c.lengthsKv.size() * sizeof(int32_t));
    if(c.padded())
    {
        lengthsQ.upload(c.lengthsQ);
        lengthsKv.upload(c.lengthsKv);
        buffers.push_back({uid::SEQ_LEN_Q, lengthsQ.get()});
        buffers.push_back({uid::SEQ_LEN_KV, lengthsKv.get()});
    }
    DeviceBuffer seed(sizeof(int64_t));
    DeviceBuffer offset(sizeof(int64_t));
    if(c.dropout > 0.0F)
    {
        seed.upload(std::vector<int64_t>{c.seed});
        offset.upload(std::vector<int64_t>{c.offset});
        buffers.push_back({uid::SEED, seed.get()});
        buffers.push_back({uid::OFFSET, offset.get()});
    }
    // A pass-by-value scale is read from host memory at execute.
    float runtimeScale = c.runtimeScale;
    if(c.scaleForm == BwdScale::RUNTIME_TENSOR)
    {
        buffers.push_back({uid::SCALE, &runtimeScale});
    }

    // Poisoned, so a kernel that reads workspace it did not write fails the case.
    const auto workspaceBytes = context.plan().getWorkspaceSize(handle);
    const GuardedWorkspace workspace(workspaceBytes);
    if(workspaceBytes > 0)
    {
        ASSERT_EQ(hipSuccess, hipMemset(workspace.get(), 0xFF, workspaceBytes));
    }
    ASSERT_NO_THROW(context.plan().execute(
        handle, buffers.data(), static_cast<uint32_t>(buffers.size()), workspace.get()));
    ASSERT_EQ(hipSuccess, hipDeviceSynchronize());
    EXPECT_EQ(workspace.overwrittenGuardBytes(), 0U)
        << "the launch wrote past the " << workspaceBytes << "-byte workspace it reported";

    const double tolerance = toleranceOf(c.dtype);
    const auto dqHost = device[views.dq.group]->template download<T>();
    expectGradient(
        "dQ", gather(dqHost, views.dq), expected.dq, scopeOf(c, true, c.heads, dqk), tolerance);
    const auto dkHost = device[views.dk.group]->template download<T>();
    expectGradient(
        "dK", gather(dkHost, views.dk), expected.dk, scopeOf(c, false, c.kvHeads, dqk), tolerance);
    const auto dvHost = device[views.dv.group]->template download<T>();
    expectGradient(
        "dV", gather(dvHost, views.dv), expected.dv, scopeOf(c, false, c.kvHeads, dv), tolerance);
    if(c.bias && c.dbias)
    {
        expectGradient("dBias",
                       gather(dbias->template download<T>(), biasView),
                       expected.dbias,
                       std::vector<bool>(expected.dbias.size(), true),
                       tolerance);
    }
}

inline void expectCase(std::string_view engine, const BwdCase& c, const ObjectPin* pin = nullptr)
{
    switch(c.dtype)
    {
    case DataType::HALF:
        expectEngineMatchesReference<hipdnn_data_sdk::types::half>(engine, c, pin);
        break;
    case DataType::BFLOAT16:
        expectEngineMatchesReference<hipdnn_data_sdk::types::bfloat16>(engine, c, pin);
        break;
    default:
        expectEngineMatchesReference<float>(engine, c, pin);
        break;
    }
}

/// The graph is not among the engine's applicable graphs, and selecting the engine by id
/// for it is refused.
inline void expectDeclined(std::string_view engine, const SdpaBwdSpec& spec)
{
    const auto builder = buildSdpaBwdGraph(spec);
    const auto graph = wrap(builder);
    core::Container container;
    auto& manager = container.getEngineManager();
    Handle handle;
    const auto engineId = idOf(engine);

    const auto applicable = manager.getApplicableEngineIds(handle, graph);
    EXPECT_EQ(std::find(applicable.begin(), applicable.end(), engineId), applicable.end())
        << engine << " claims a graph it must decline";

    hipdnn_test_sdk::utilities::MockEngineConfig config;
    stubEngineId(config, engineId);
    Context context;
    EXPECT_THROW(manager.initializeExecutionContext(handle, graph, config, context),
                 hipdnn_plugin_sdk::HipdnnPluginException);
}

/// Skips the case unless this install ships every op of an engine for the current device,
/// whichever architecture's content serves it.
class FlydslSdpaBwdDevice : public ::testing::Test
{
protected:
    explicit FlydslSdpaBwdDevice(const std::vector<std::string_view>& ops)
        : _ops(ops)
    {
    }

    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        for(const auto op : _ops)
        {
            if(!flydslShipsForThisDevice(op))
            {
                GTEST_SKIP() << "this install ships no " << op << " objects for "
                             << packedArchDeviceProperties().gcnArchName;
            }
        }
    }

private:
    const std::vector<std::string_view>& _ops;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing::bwd_device

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
