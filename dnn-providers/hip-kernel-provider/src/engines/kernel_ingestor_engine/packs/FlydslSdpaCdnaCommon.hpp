// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Redundant with the CMake gate on purpose; see FlydslSdpaCdnaNative.cpp.
#if defined(HIPDNN_ENABLE_KERNEL_INGESTOR) && defined(HIPDNN_ENGINE_FLYDSL)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeRegistry.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/IKernelCompiler.hpp"
#include "compilation/KernelCompileOptions.hpp"
#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/hip_mlops_engine/HipMlopsKernelCompiler.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaShared.hpp"

/**
 * @file FlydslSdpaCdnaCommon.hpp
 * @brief What the three CDNA SDPA engines share: the graph matcher and its bindings, the
 *        kernel-scoped checks, the argument lists, and the dispatch, each for one family.
 *
 * FlyDSL's CDNA attention forwards, vendored under
 * `flydsl/kernels_src/kernels/attention_cdna/`, are three kernel families and so three
 * engines: the generic MFMA kernel's prefill objects (`hipkernel:flydsl_sdpa_cdna`,
 * FlydslSdpaCdnaNative.cpp), gfx950's dual-wave software-pipelined kernel
 * (`hipkernel:flydsl_sdpa_cdna_dualwave`, FlydslSdpaCdnaDualwaveNative.cpp), and the
 * generic kernel's split-KV decode objects, which pack a GQA group into each row tile and
 * merge the splits in a second kernel (`hipkernel:flydsl_sdpa_cdna_decode`,
 * FlydslSdpaCdnaDecodeNative.cpp). Which serves a graph several can is the heuristic's to
 * decide, not a rule here. All three serve the same graphs, so one graph matcher is
 * registered under each engine's symbol; each object's metadata carries only its family's
 * knobs, and `familyKnob` supplies the rest.
 *
 * Each object bakes its dtype, head_dim and whether a causal mask applies. Batch, both
 * sequence lengths, both head counts, every operand's (batch, sequence, head) strides, the
 * softmax scale, the mask's diagonal offset and whether to write the LSE are runtime
 * arguments (flydsl/COVERAGE.md section 7).
 *
 * Kernel arguments. FlyDSL lowers each tensor parameter to two kernel arguments: its
 * 8-byte global pointer, then a 40-byte by-value layout -- the four i32 extents of the
 * rank-4 tensor followed by the i64 strides of its three non-unit-stride dimensions, in
 * ascending order (FlyDSL's dynamic-layout contract, `_LayoutPlan` in
 * flydsl/compiler/jit_argument.py). The semantic names are in each object's manifest.json.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_cdna
{

using namespace hipdnn_plugin_sdk::ingestor;
using namespace flydsl_sdpa;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

// Kernel metadata fields, declared by the KMD this pack's KDP points at.
inline constexpr std::string_view DTYPE_FIELD = "dtype";
inline constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
/// V's and O's head dim: head_dim's, or narrower on the MLA objects (vendored-kernel
/// modification 13).
inline constexpr std::string_view HEAD_DIM_V_FIELD = "head_dim_v";
/// 1 for an object whose head_dim is a bucket: it serves any Q/K head dim up to it that is
/// a multiple of QK_MASK_ALIGN, read at runtime (modification 13).
inline constexpr std::string_view QK_MASK_FIELD = "qk_mask";
/// A masked K or V load bounds 8-element halves (vendored-kernel modification 17).
inline constexpr int64_t QK_MASK_ALIGN = 8;
inline constexpr std::string_view CAUSAL_FIELD = "causal";
inline constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// 0 for an object that serves any key length; otherwise the multiple the key length must
/// be, because the object skips the KV tail mask.
inline constexpr std::string_view KV_ALIGN_FIELD = "kv_align";
/// 1 for an object that serves any diagonal offset; 0 for one built without the handling
/// an offset other than 0 needs, which only an offset-0 graph may run.
inline constexpr std::string_view ANY_DIAGONAL_FIELD = "any_diagonal";
/// 1 for an object built to add a bias to the scores, 0 for one built without.
inline constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
/// When an object is preferred over its class's default: while the graph's
/// `prefer_feature` -- "rows" (batch x query heads x query length) or "seq_q" (the query
/// length) -- lies in [`prefer_min`, `prefer_max`), a `prefer_max` of 0 being unbounded.
/// "none" for an object with no preference, which ranks by its priority alone.
inline constexpr std::string_view PREFER_FEATURE_FIELD = "prefer_feature";
inline constexpr std::string_view PREFER_MIN_FIELD = "prefer_min";
inline constexpr std::string_view PREFER_MAX_FIELD = "prefer_max";
inline constexpr std::string_view PREFER_ROWS = "rows";
inline constexpr std::string_view PREFER_SEQ_Q = "seq_q";
/// Added to a preferred object's priority: more than any gap between priorities, so a
/// preferred object outranks every one that is not.
inline constexpr double PREFERRED_BONUS = 1000.0;

inline constexpr std::string_view Q_TOKEN = "flydsl_sdpa_cdna.q.uid";
inline constexpr std::string_view K_TOKEN = "flydsl_sdpa_cdna.k.uid";
inline constexpr std::string_view V_TOKEN = "flydsl_sdpa_cdna.v.uid";
inline constexpr std::string_view O_TOKEN = "flydsl_sdpa_cdna.o.uid";
/// 1 when the graph asks for causal masking, 0 for none.
inline constexpr std::string_view CAUSAL_TOKEN = "flydsl_sdpa_cdna.causal";
/// The causal diagonal: query row i sees keys j <= i + offset. Unused without a mask.
inline constexpr std::string_view DIAG_OFFSET_TOKEN = "flydsl_sdpa_cdna.diag_offset";
/// The sliding window's lower edge: query row i sees no key j < i + window_lo; NO_WINDOW
/// when the graph has none (or one no row of it reaches).
inline constexpr std::string_view WINDOW_TOKEN = "flydsl_sdpa_cdna.window_lo";
/// The uid of the softmax-statistics (LSE) output, or NO_STATS when the graph wants none.
inline constexpr std::string_view STATS_TOKEN = "flydsl_sdpa_cdna.stats.uid";
/// 1 when the LSE output is the packed [B, H, Sq] the dual-wave kernel writes (or there is
/// none); 0 when it has strides of its own, which only the generic kernel takes
/// (vendored-kernel modification 20).
inline constexpr std::string_view STATS_PACKED_TOKEN = "flydsl_sdpa_cdna.stats.packed";
/// 1 when V's strides differ from K's at an equal head dim. The generic kernel's
/// direct-to-LDS path reads V through K's descriptor, so only the objects built on its
/// masked path -- a Q/K mask or a narrower V -- read V through its own.
inline constexpr std::string_view V_OWN_STRIDES_TOKEN = "flydsl_sdpa_cdna.v.own_strides";
/// The uid of the additive bias (attn_mask), or NO_BIAS when the graph has none.
inline constexpr std::string_view BIAS_TOKEN = "flydsl_sdpa_cdna.bias.uid";
/// The attention sink's uid, or NO_SINK, and its element stride across query heads
/// (vendored-kernel modification 19).
inline constexpr std::string_view SINK_TOKEN = "flydsl_sdpa_cdna.sink.uid";
inline constexpr std::string_view SINK_STRIDE_TOKEN = "flydsl_sdpa_cdna.sink.stride";
/// 1 for an object built to add an attention sink's logit to its softmax (modification 19).
inline constexpr std::string_view HAS_SINK_FIELD = "has_sink";

/// The dual-wave kernel's workgroup is fixed (two waves of 64 per 32 query rows of its
/// 256-row tile); the generic kernel runs two threads per query row of its tile.
inline constexpr int64_t DUALWAVE_BLOCK_THREADS = 512;
inline constexpr int64_t GENERIC_THREADS_PER_ROW = 2;

/// 1 for a generic object built to split the keys across workgroups (vendored-kernel
/// modification 14); it carries the merge kernel that combines the splits, by name.
inline constexpr std::string_view SPLIT_KV_FIELD = "split_kv";
inline constexpr std::string_view MERGE_SYMBOL_FIELD = "merge_symbol";
/// 1 for a generic object whose rows are a GQA group's query heads by position
/// (modification 15): one workgroup per (batch, row tile, KV head).
inline constexpr std::string_view PACK_GQA_FIELD = "pack_gqa";
/// How many workgroups split O's columns (modification 16): the grid's z.
inline constexpr std::string_view DV_SPLIT_FIELD = "dv_split";
/// The split count: enough workgroups to fill the device -- SPLIT_WORKGROUPS_PER_UNIT per
/// reported compute unit -- but no split shorter than SPLIT_MIN_TILES key tiles of
/// SPLIT_TILE keys, and at most SPLIT_MAX. Deterministic in the problem and the device,
/// so workspaceBytes and prepare agree.
inline constexpr int64_t SPLIT_WORKGROUPS_PER_UNIT = 2;
inline constexpr int64_t SPLIT_TILE = 64;
inline constexpr int64_t SPLIT_MIN_TILES = 4;
inline constexpr int64_t SPLIT_MAX = 64;
inline constexpr int64_t SPLIT_FALLBACK_COMPUTE_UNITS = 256;
/// Threads per merge workgroup, one (batch, head, query) row each.
inline constexpr int64_t MERGE_THREADS = 64;

/// The generic kernel's `window_lo` for "no window": below any row's reach (the kernels'
/// _NO_WINDOW, vendored-kernel modification 11).
inline constexpr int64_t NO_WINDOW = -(int64_t{1} << 30);

/// Where the softmax scale comes from; see ScaleSource.
inline constexpr std::string_view SCALE_SOURCE_TOKEN = "flydsl_sdpa_cdna.scale_source";
/// The scale's f32 bit pattern when it is a value (BoundTokens carry int64_t), else the uid
/// of the pass-by-value scale tensor.
inline constexpr std::string_view SCALE_TOKEN = "flydsl_sdpa_cdna.scale";

/// Where the kernel's `sm_scale` argument comes from.
enum class ScaleSource : int64_t
{
    /// attn_scale_value, or 1.0 when the graph gives none -- cuDNN's default, and the one
    /// hipDNN's references apply.
    VALUE = 0,
    /// A pass-by-value scalar tensor, resolved at execute.
    TENSOR = 1,
};

/// One kernel family per engine (`_instances.SDPA_CDNA_FAMILY_CONSTANTS`): the generic
/// kernel's prefill objects, gfx950's dual-wave kernel, and the generic kernel's split-KV
/// decode objects. An object's metadata carries only its own family's knobs; the ones that
/// told the families apart are this family's constants.
enum class CdnaFamily
{
    GENERIC,
    DUALWAVE,
    DECODE,
};

/// A knob of @p kernel, or its family's constant for a knob the family does not carry.
inline std::optional<int64_t>
    familyKnob(const KernelDefinition& kernel, std::string_view field, CdnaFamily family);

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

/// The graph's shape, read from Q and K. Callers must have validated both operands.
struct SdpaProblem
{
    int64_t batch = 0;
    int64_t numHeads = 0;
    int64_t numKvHeads = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t headDim = 0;
    /// V's and O's head dim; Q/K's, or narrower (multi-head latent attention).
    int64_t headDimV = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

inline SdpaProblem problemFor(const data_objects::TensorAttributes& q,
                              const data_objects::TensorAttributes& k,
                              const data_objects::TensorAttributes* v)
{
    SdpaProblem problem;
    problem.batch = q.dims()->Get(BATCH_AXIS);
    problem.numHeads = q.dims()->Get(HEAD_AXIS);
    problem.seqLenQ = q.dims()->Get(SEQ_AXIS);
    problem.seqLenKv = k.dims()->Get(SEQ_AXIS);
    problem.headDim = q.dims()->Get(HEAD_DIM_AXIS);
    problem.numKvHeads = k.dims()->Get(HEAD_AXIS);
    problem.headDimV = problem.headDim;
    if(v != nullptr && v->dims() != nullptr && v->dims()->size() == SDPA_RANK)
    {
        problem.headDimV = v->dims()->Get(HEAD_DIM_AXIS);
    }
    problem.dataType = q.data_type();
    return problem;
}

/// One operand's (batch, sequence, head) strides, in elements; head_dim's is 1.
struct OperandStrides
{
    int64_t batch = 0;
    int64_t sequence = 0;
    int64_t head = 0;

    bool operator==(const OperandStrides& other) const
    {
        return batch == other.batch && sequence == other.sequence && head == other.head;
    }
};

/// The kernels move 16-byte vectors: eight bf16 / f16 elements.
inline constexpr int64_t VECTOR_ELEMENTS = 8;

/**
 * @brief The strides of @p tensor as the kernels take them, or nullopt for a tensor they
 *        cannot read.
 *
 * Logical shape [B, H, S, D] with head_dim contiguous; the batch, sequence and head strides
 * are kernel arguments, so BSHD, BHSD and packed-QKV views are all served. Each stride on an
 * axis longer than 1 must be positive and keep every row 16-byte aligned. Each (batch, head)
 * slice is one buffer view of (S - 1) * stride_s + D elements, which must stay within 32-bit
 * addressing; rows past the sequence -- a query tile's padding rows -- read as zero and
 * store nowhere only because they fall outside it, which needs a sequence stride of at
 * least D. So a single-row sequence is given stride D rather than its stated one, which
 * is never stepped, and overlapping rows decline. A batch or head axis of extent 1 is never
 * stepped and gets stride 0. A ragged operand would be read as one dense batch, so it
 * declines too.
 */
inline std::optional<OperandStrides> servableStrides(const data_objects::TensorAttributes& tensor,
                                                     int64_t batch,
                                                     int64_t heads,
                                                     int64_t sequence,
                                                     int64_t headDim)
{
    const auto* dims = tensor.dims();
    const auto* strides = tensor.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != sequence || dims->Get(HEAD_DIM_AXIS) != headDim
       || strides->Get(HEAD_DIM_AXIS) != 1)
    {
        return std::nullopt;
    }
    const auto stepped = [&](uint32_t axis, int64_t extent) -> std::optional<int64_t> {
        if(extent == 1)
        {
            return axis == SEQ_AXIS ? headDim : 0;
        }
        const int64_t stride = strides->Get(axis);
        if(stride <= 0 || stride % VECTOR_ELEMENTS != 0 || stride > INT32_LIMIT)
        {
            return std::nullopt;
        }
        return stride;
    };
    const auto batchStride = stepped(BATCH_AXIS, batch);
    const auto sequenceStride = stepped(SEQ_AXIS, sequence);
    const auto headStride = stepped(HEAD_AXIS, heads);
    if(!batchStride.has_value() || !sequenceStride.has_value() || !headStride.has_value()
       || *sequenceStride < headDim)
    {
        return std::nullopt;
    }
    const auto spanRows = checkedProduct({sequence - 1, *sequenceStride});
    if(!spanRows.has_value() || (*spanRows + headDim) * BYTES_PER_ELEMENT > INT32_LIMIT)
    {
        return std::nullopt;
    }
    if(tensor.ragged_offset_tensor_uid().has_value() || tensor.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&tensor))
    {
        return std::nullopt;
    }
    return OperandStrides{*batchStride, *sequenceStride, *headStride};
}

/// The strides the kernels take: Q's, O's, K's and V's.
struct ViewStrides
{
    OperandStrides q;
    OperandStrides o;
    OperandStrides kv;
    OperandStrides v;
    /// The LSE output's (batch, head, query) strides; packed when there is none.
    OperandStrides lse;
};

/// The operands' strides, or nullopt when any is unservable. V's may differ from K's; which
/// objects read V through its own strides is kernel_match's question (V_OWN_STRIDES_TOKEN).
inline std::optional<ViewStrides> viewStridesFor(const data_objects::TensorAttributes& q,
                                                 const data_objects::TensorAttributes& k,
                                                 const data_objects::TensorAttributes& v,
                                                 const data_objects::TensorAttributes& o,
                                                 const SdpaProblem& problem)
{
    const auto qStrides
        = servableStrides(q, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim);
    const auto oStrides
        = servableStrides(o, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDimV);
    const auto kStrides
        = servableStrides(k, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDim);
    const auto vStrides
        = servableStrides(v, problem.batch, problem.numKvHeads, problem.seqLenKv, problem.headDimV);
    if(!qStrides.has_value() || !oStrides.has_value() || !kStrides.has_value()
       || !vStrides.has_value())
    {
        return std::nullopt;
    }
    return ViewStrides{*qStrides,
                       *oStrides,
                       *kStrides,
                       *vStrides,
                       OperandStrides{problem.numHeads * problem.seqLenQ, 1, problem.seqLenQ}};
}

/// The bias's (batch, head, query) strides as the kernels take them; 0 broadcasts an axis.
struct BiasStrides
{
    int64_t batch = 0;
    int64_t head = 0;
    int64_t query = 0;
};

/**
 * @brief The bias's strides, or nullopt for one the kernels cannot add.
 *
 * The objects that add a bias read it in the query's dtype, as PyTorch requires of a float
 * attn_mask, right-aligned to [B, H, Sq, Skv] with any of B, H and Sq broadcast (extent 1 or
 * absent); the key axis is read contiguously and cannot be. Rows move as 16-byte vectors, so
 * every stride is a multiple of 8 elements, and one (batch, head) slice is one buffer view
 * of (Sq - 1) * stride_q + Skv elements within 32-bit addressing. A row stride below Skv
 * would overlap rows, so it declines.
 */
inline std::optional<BiasStrides> servableBias(const data_objects::TensorAttributes& bias,
                                               const SdpaProblem& problem)
{
    const auto* dims = bias.dims();
    const auto* strides = bias.strides();
    if(dims == nullptr || strides == nullptr || dims->empty() || dims->size() > SDPA_RANK
       || strides->size() != dims->size() || bias.data_type() != problem.dataType || bias.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&bias)
       || bias.ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Right-aligned: the last axis is the keys, then the queries, heads and batch.
    const auto rank = static_cast<int64_t>(dims->size());
    const std::array<int64_t, 4> full{
        problem.batch, problem.numHeads, problem.seqLenQ, problem.seqLenKv};
    std::array<int64_t, 4> stride{0, 0, 0, 0};
    for(int64_t axis = 0; axis < SDPA_RANK; ++axis)
    {
        const int64_t declared = axis - (SDPA_RANK - rank);
        const int64_t extent
            = declared < 0 ? 1 : dims->Get(static_cast<flatbuffers::uoffset_t>(declared));
        if(extent == 1 && full[static_cast<size_t>(axis)] != 1 && axis != 3)
        {
            continue; // broadcast
        }
        if(extent != full[static_cast<size_t>(axis)])
        {
            return std::nullopt;
        }
        const int64_t value
            = extent == 1 ? 0 : strides->Get(static_cast<flatbuffers::uoffset_t>(declared));
        stride[static_cast<size_t>(axis)] = value;
    }
    if(problem.seqLenKv > 1 && stride[3] != 1)
    {
        return std::nullopt;
    }
    for(size_t axis = 0; axis < 3; ++axis)
    {
        if(stride[axis] < 0 || stride[axis] % VECTOR_ELEMENTS != 0 || stride[axis] > INT32_LIMIT)
        {
            return std::nullopt;
        }
    }
    if(stride[2] != 0 && stride[2] < problem.seqLenKv)
    {
        return std::nullopt;
    }
    // The kernels read the bias as runs of 4 keys, and the slice's view ends at the last
    // row's last key: a run straddling that end may read as zero, so a key length that is
    // not a multiple of 4 declines.
    if(problem.seqLenKv % 4 != 0)
    {
        return std::nullopt;
    }
    const auto span = checkedProduct({problem.seqLenQ - 1, stride[2]});
    if(!span.has_value() || (*span + problem.seqLenKv) * BYTES_PER_ELEMENT > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return BiasStrides{stride[0], stride[1], stride[2]};
}

/// The (batch, head, query) strides of @p stats, an LSE output the kernels write: f32
/// [B, H, Sq, 1], each axis of more than one element at a positive stride, every element
/// within 32-bit offsets. nullopt for one they cannot. An axis of one element gets stride 0.
inline std::optional<OperandStrides> servableStatsStrides(
    const data_objects::TensorAttributes& stats, int64_t batch, int64_t heads, int64_t sequence)
{
    const auto* dims = stats.dims();
    const auto* strides = stats.strides();
    if(dims == nullptr || strides == nullptr || dims->size() != SDPA_RANK
       || strides->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    if(dims->Get(BATCH_AXIS) != batch || dims->Get(HEAD_AXIS) != heads
       || dims->Get(SEQ_AXIS) != sequence || dims->Get(HEAD_DIM_AXIS) != 1)
    {
        return std::nullopt;
    }
    if(stats.data_type() != data_objects::DataType::FLOAT || stats.virtual_()
       || hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(&stats)
       || stats.ragged_offset_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // A stride on an axis of extent 1 is never used to address anything.
    const auto axisStride = [strides](uint32_t axis, int64_t extent) -> std::optional<int64_t> {
        if(extent == 1)
        {
            return 0;
        }
        const int64_t stride = strides->Get(axis);
        if(stride <= 0 || stride > INT32_LIMIT)
        {
            return std::nullopt;
        }
        return stride;
    };
    const auto batchStride = axisStride(BATCH_AXIS, batch);
    const auto headStride = axisStride(HEAD_AXIS, heads);
    const auto seqStride = axisStride(SEQ_AXIS, sequence);
    if(!batchStride.has_value() || !headStride.has_value() || !seqStride.has_value())
    {
        return std::nullopt;
    }
    // The merge kernel addresses the whole tensor with one 32-bit byte offset (element
    // offset times 4).
    int64_t last = 0;
    for(const auto& [extent, stride] : {std::pair{batch, *batchStride},
                                        std::pair{heads, *headStride},
                                        std::pair{sequence, *seqStride}})
    {
        const auto reach = checkedProduct({extent - 1, stride});
        if(!reach.has_value() || *reach >= INT32_LIMIT)
        {
            return std::nullopt;
        }
        last += *reach;
    }
    if(last >= INT32_LIMIT / static_cast<int64_t>(sizeof(float)))
    {
        return std::nullopt;
    }
    return OperandStrides{*batchStride, *seqStride, *headStride};
}

/// Whether @p lse is the packed [B, H, Sq] the dual-wave kernel writes.
inline bool isPackedStats(const OperandStrides& lse, int64_t batch, int64_t heads, int64_t sequence)
{
    return (sequence == 1 || lse.sequence == 1) && (heads == 1 || lse.head == sequence)
           && (batch == 1 || lse.batch == heads * sequence);
}

/// The mask the kernels apply: none, or j <= i + offset for query row i and key j, and
/// with a sliding window also j >= i + windowLo.
struct CdnaMask
{
    bool causal = false;
    int64_t offset = 0;
    int64_t windowLo = NO_WINDOW;
};

/**
 * @brief The graph's mask in the kernels' terms, or nullopt for one they cannot apply.
 *
 * In the kernels' terms (resolveMaskBounds; left is the graph's left_bound - 1) the
 * reference hides key j from query row i when
 * j > i + align + right, with align 0 top-left and Skv - Sq bottom-right; so causal and a
 * band to the right of the diagonal are both one offset, align + right. It hides j <
 * i + align - left too, a sliding window: the window's lower edge, align - left, is the
 * generic kernel's `window_lo` (vendored-kernel modification 11), and a window no row
 * reaches is none. A left bound with no right one declines: only the causal objects mask
 * at all. So does an offset past Skv - Sq: the causal objects mask the last KV tile's
 * padding with the same compare, which holds only while every row's diagonal stays
 * inside the keys -- top-left with more queries than keys, or a band.
 */
inline std::optional<CdnaMask>
    maskFor(const data_objects::SdpaAttributes& attributes, int64_t seqLenQ, int64_t seqLenKv)
{
    // The deprecated booleans resolved as the reference does (resolveMaskBounds).
    const auto bounds = resolveMaskBounds(attributes);
    if(!bounds.has_value())
    {
        return std::nullopt;
    }
    const int64_t left = bounds->left;
    const int64_t right = bounds->right;
    const bool alignBottomRight = bounds->alignBottomRight;
    if(right == UNBOUNDED)
    {
        if(left != UNBOUNDED)
        {
            return std::nullopt;
        }
        return CdnaMask{};
    }
    const int64_t align = alignBottomRight ? seqLenKv - seqLenQ : 0;
    // Compared before adding, so a bound near INT64_MAX cannot overflow.
    if(right > seqLenKv - seqLenQ - align)
    {
        return std::nullopt;
    }
    const int64_t offset = align + right;
    CdnaMask mask{true, offset};
    // The last query row, the one reaching furthest right, hides keys j < (Sq - 1) +
    // window_lo; at or below -(Sq - 1) the window hides nothing (left compared, not
    // subtracted, for the same reason).
    if(left != UNBOUNDED && left < align + seqLenQ - 1)
    {
        mask.windowLo = align - left;
    }
    return mask;
}

/**
 * @brief Graph-scoped applicability: is this a single SDPA-forward node one of this pack's
 *        pre-built CDNA kernels computes exactly?
 *
 * Refuses rather than approximates: every feature the kernels do not implement is
 * declined, so a graph asking for one falls through to an engine that can honour it.
 */
inline std::optional<BoundTokens> flydslSdpaCdnaGraphMatches(const MatchContext& context)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return std::nullopt;
    }
    const auto& attributes = *attributesPtr;
    if(!computesInFloat(context))
    {
        return std::nullopt;
    }

    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    const auto* v = findTensor(context, attributes.v_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || k == nullptr || v == nullptr || o == nullptr || q->dims() == nullptr
       || k->dims() == nullptr || q->dims()->size() != SDPA_RANK || k->dims()->size() != SDPA_RANK)
    {
        return std::nullopt;
    }
    // Rows move as 16-byte vectors from each operand's base.
    const int64_t vectorBytes = VECTOR_ELEMENTS * BYTES_PER_ELEMENT;
    // O is written by many workgroups at once, so no two of its elements may share an
    // address.
    if(!declaresAlignment(*q, vectorBytes) || !declaresAlignment(*k, vectorBytes)
       || !declaresAlignment(*v, vectorBytes) || !declaresAlignment(*o, vectorBytes)
       || !elementsDoNotOverlap(*o))
    {
        return std::nullopt;
    }

    const auto problem = problemFor(*q, *k, v);
    if(problem.batch <= 0 || problem.numHeads <= 0 || problem.numKvHeads <= 0
       || problem.seqLenQ <= 0 || problem.seqLenKv <= 0 || problem.headDim <= 0
       || problem.headDimV <= 0 || problem.headDimV > problem.headDim)
    {
        return std::nullopt;
    }
    // One dtype throughout, and one the kernels are built for.
    if(!kernelDataTypeName(problem.dataType).has_value() || k->data_type() != problem.dataType
       || v->data_type() != problem.dataType || o->data_type() != problem.dataType)
    {
        return std::nullopt;
    }
    // V shaped as K but for a head dim that may be narrower, O as Q with V's head dim, GQA by
    // integer division.
    if(problem.numHeads % problem.numKvHeads != 0)
    {
        return std::nullopt;
    }
    const auto views = viewStridesFor(*q, *k, *v, *o, problem);
    if(!views.has_value())
    {
        return std::nullopt;
    }
    const bool vOwnStrides = problem.headDimV == problem.headDim && !(views->kv == views->v);
    if(problem.seqLenQ > INT32_LIMIT || problem.seqLenKv > INT32_LIMIT
       || problem.batch > INT32_LIMIT)
    {
        return std::nullopt;
    }

    const auto mask = maskFor(attributes, problem.seqLenQ, problem.seqLenKv);
    if(!mask.has_value())
    {
        return std::nullopt;
    }

    // Every optional feature the kernels as built do not implement, declined explicitly.
    if(attributes.seq_len_q_tensor_uid().has_value()
       || attributes.seq_len_kv_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    if(attributes.seed_tensor_uid().has_value() || attributes.offset_tensor_uid().has_value()
       || attributes.dropout_mask_tensor_uid().has_value()
       || attributes.dropout_scale_tensor_uid().has_value()
       || attributes.dropout_probability().has_value())
    {
        return std::nullopt;
    }
    if(attributes.page_table_k_tensor_uid().has_value()
       || attributes.page_table_v_tensor_uid().has_value()
       || attributes.max_seq_len_kv().has_value() || attributes.block_mask_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    if(attributes.descale_q_tensor_uid().has_value()
       || attributes.descale_k_tensor_uid().has_value()
       || attributes.descale_v_tensor_uid().has_value()
       || attributes.descale_s_tensor_uid().has_value()
       || attributes.scale_s_tensor_uid().has_value() || attributes.scale_o_tensor_uid().has_value()
       || attributes.amax_s_tensor_uid().has_value() || attributes.amax_o_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    // Softmax statistics: the LSE output is served; the split max / sum-exp outputs and the
    // RNG dump are not. Asking for stats without naming a tensor to write them to is a graph
    // the kernels cannot satisfy.
    if(attributes.max_tensor_uid().has_value() || attributes.sum_exp_tensor_uid().has_value()
       || attributes.rng_dump_tensor_uid().has_value())
    {
        return std::nullopt;
    }
    int64_t biasUid = NO_BIAS;
    if(attributes.attn_mask_tensor_uid().has_value())
    {
        const auto* bias = findTensor(context, attributes.attn_mask_tensor_uid().value());
        if(bias == nullptr || !declaresAlignment(*bias, vectorBytes)
           || !servableBias(*bias, problem).has_value())
        {
            return std::nullopt;
        }
        biasUid = attributes.attn_mask_tensor_uid().value();
    }
    int64_t statsUid = NO_STATS;
    bool statsPacked = true;
    if(attributes.stats_tensor_uid().has_value())
    {
        // An LSE tensor with generate_stats explicitly false would be written anyway: the
        // two disagree, so the graph declines rather than one being ignored.
        const auto* stats = findTensor(context, attributes.stats_tensor_uid().value());
        const auto lseStrides
            = stats == nullptr
                  ? std::nullopt
                  : servableStatsStrides(*stats, problem.batch, problem.numHeads, problem.seqLenQ);
        if(!lseStrides.has_value() || !declaresAlignment(*stats, sizeof(float))
           || !elementsDoNotOverlap(*stats)
           || (attributes.generate_stats().has_value() && !attributes.generate_stats().value()))
        {
            return std::nullopt;
        }
        statsUid = attributes.stats_tensor_uid().value();
        statsPacked = isPackedStats(*lseStrides, problem.batch, problem.numHeads, problem.seqLenQ);
    }
    else if(attributes.generate_stats().has_value() && attributes.generate_stats().value())
    {
        return std::nullopt;
    }
    if(attributes.alibi_mask() || attributes.padding_mask())
    {
        return std::nullopt;
    }
    const auto mmaCoreMode = attributes.mma_core_mode();
    if(mmaCoreMode != data_objects::DataType::UNSET && mmaCoreMode != data_objects::DataType::HALF
       && mmaCoreMode != data_objects::DataType::BFLOAT16)
    {
        return std::nullopt;
    }
    if(attributes.implementation() != data_objects::AttentionImplementation::AUTO)
    {
        return std::nullopt;
    }

    // The softmax scale: a value, a pass-by-value tensor, or the default.
    ScaleSource scaleSource = ScaleSource::VALUE;
    int64_t scaleToken = 0;
    if(attributes.scale_tensor_uid().has_value())
    {
        // Only a host scalar can become a kernel argument; a device-resident scale would
        // need a device read the kernels do not do. A value set as well would be ignored,
        // so the two together decline.
        if(attributes.attn_scale_value().has_value()
           || !isServableScaleTensor(context, attributes.scale_tensor_uid().value()))
        {
            return std::nullopt;
        }
        scaleSource = ScaleSource::TENSOR;
        scaleToken = attributes.scale_tensor_uid().value();
    }
    else
    {
        // An absent attn_scale_value is 1.0, cuDNN's default: its SDPA node multiplies by
        // attn_scale only when one is set. hipDNN's references read it the same way.
        const float scale = attributes.attn_scale_value().value_or(1.0F);
        if(!isServableScale(static_cast<double>(scale)))
        {
            return std::nullopt;
        }
        int32_t scaleBits = 0;
        static_assert(sizeof(scaleBits) == sizeof(scale), "float must be 32-bit to round-trip");
        std::memcpy(&scaleBits, &scale, sizeof(scale));
        scaleToken = static_cast<int64_t>(scaleBits);
    }

    // An attention sink: one logit per query head, in the query's dtype, as a rank-1
    // [Hq] or rank-4 [1, Hq, 1, 1] device tensor (vendored-kernel modification 19).
    int64_t sinkUid = NO_SINK;
    int64_t sinkStride = 0;
    if(attributes.sink_token_tensor_uid().has_value())
    {
        const auto stride = servableSinkStride(
            context, *attributes.sink_token_tensor_uid(), problem.numHeads, problem.dataType);
        if(!stride.has_value())
        {
            return std::nullopt;
        }
        sinkUid = *attributes.sink_token_tensor_uid();
        sinkStride = *stride;
    }

    BoundTokens bound;
    bound[std::string(Q_TOKEN)] = attributes.q_tensor_uid();
    bound[std::string(K_TOKEN)] = attributes.k_tensor_uid();
    bound[std::string(V_TOKEN)] = attributes.v_tensor_uid();
    bound[std::string(O_TOKEN)] = attributes.o_tensor_uid();
    bound[std::string(CAUSAL_TOKEN)] = mask->causal ? 1 : 0;
    bound[std::string(DIAG_OFFSET_TOKEN)] = mask->offset;
    bound[std::string(WINDOW_TOKEN)] = mask->windowLo;
    bound[std::string(STATS_TOKEN)] = statsUid;
    bound[std::string(STATS_PACKED_TOKEN)] = statsPacked ? 1 : 0;
    bound[std::string(V_OWN_STRIDES_TOKEN)] = vOwnStrides ? 1 : 0;
    bound[std::string(BIAS_TOKEN)] = biasUid;
    bound[std::string(SINK_TOKEN)] = sinkUid;
    bound[std::string(SINK_STRIDE_TOKEN)] = sinkStride;
    bound[std::string(SCALE_SOURCE_TOKEN)] = static_cast<int64_t>(scaleSource);
    bound[std::string(SCALE_TOKEN)] = scaleToken;
    return bound;
}

/// The bindings a match established, re-read for dispatch.
struct FlydslSdpaCdnaBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t diagOffset = 0;
    int64_t windowLo = NO_WINDOW;
    int64_t stats = NO_STATS;
    int64_t bias = NO_BIAS;
    int64_t sink = NO_SINK;
    int64_t sinkStride = 0;
    ScaleSource scaleSource = ScaleSource::VALUE;
    int64_t scale = 0;
};

inline FlydslSdpaCdnaBinding flydslSdpaCdnaBinding(const BoundTokens& bound)
{
    const auto read = [&bound](std::string_view token) {
        const auto value = tryGetBoundInt(bound, token);
        if(!value.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                "flydsl_sdpa_cdna dispatch is missing bound token '" + std::string(token)
                    + "', or it does not hold an integer");
        }
        return *value;
    };

    FlydslSdpaCdnaBinding binding;
    binding.q = read(Q_TOKEN);
    binding.k = read(K_TOKEN);
    binding.v = read(V_TOKEN);
    binding.o = read(O_TOKEN);
    binding.causal = read(CAUSAL_TOKEN);
    binding.diagOffset = read(DIAG_OFFSET_TOKEN);
    binding.windowLo = read(WINDOW_TOKEN);
    binding.stats = read(STATS_TOKEN);
    binding.bias = read(BIAS_TOKEN);
    binding.sink = read(SINK_TOKEN);
    binding.sinkStride = read(SINK_STRIDE_TOKEN);
    binding.scaleSource = static_cast<ScaleSource>(read(SCALE_SOURCE_TOKEN));
    binding.scale = read(SCALE_TOKEN);
    return binding;
}

inline std::optional<int64_t>
    familyKnob(const KernelDefinition& kernel, std::string_view field, CdnaFamily family)
{
    // Each family's constants, in the KMD's own field names.
    struct Constant
    {
        std::string_view field;
        int64_t value;
    };
    static constexpr std::array<Constant, 3> GENERIC_CONSTANTS{
        {{"any_diagonal", 1}, {"split_kv", 0}, {"pack_gqa", 0}}};
    static constexpr std::array<Constant, 6> DUALWAVE_CONSTANTS{{{"kv_align", 0},
                                                                 {"qk_mask", 0},
                                                                 {"has_sink", 0},
                                                                 {"dv_split", 1},
                                                                 {"split_kv", 0},
                                                                 {"pack_gqa", 0}}};
    static constexpr std::array<Constant, 4> DECODE_CONSTANTS{
        {{"any_diagonal", 1}, {"kv_align", 0}, {"split_kv", 1}, {"pack_gqa", 1}}};
    const auto lookup = [field](const auto& constants) -> std::optional<int64_t> {
        for(const auto& constant : constants)
        {
            if(constant.field == field)
            {
                return constant.value;
            }
        }
        return std::nullopt;
    };
    std::optional<int64_t> constant;
    switch(family)
    {
    case CdnaFamily::GENERIC:
        constant = lookup(GENERIC_CONSTANTS);
        break;
    case CdnaFamily::DUALWAVE:
        constant = lookup(DUALWAVE_CONSTANTS);
        break;
    case CdnaFamily::DECODE:
        constant = lookup(DECODE_CONSTANTS);
        break;
    default:
        break;
    }
    return constant.has_value() ? constant : integerMetadata(kernel, field);
}

/// A candidate's launch shape, or nullopt when its metadata describes none this pack
/// knows how to launch.
struct CdnaTile
{
    bool dualwave = false;
    int64_t blockM = 0;
    bool splitKv = false;
    bool packGqa = false;
    int64_t dvSplit = 1;
};

inline std::optional<CdnaTile> candidateTile(const KernelDefinition& kernel, CdnaFamily family)
{
    const auto blockM = integerMetadata(kernel, BLOCK_M_FIELD);
    if(!blockM.has_value() || *blockM <= 0)
    {
        return std::nullopt;
    }
    if(family == CdnaFamily::DUALWAVE)
    {
        return CdnaTile{true, *blockM};
    }
    const auto splitKv = familyKnob(kernel, SPLIT_KV_FIELD, family);
    const auto packGqa = familyKnob(kernel, PACK_GQA_FIELD, family);
    const auto dvSplit = familyKnob(kernel, DV_SPLIT_FIELD, family);
    if(!splitKv.has_value() || !packGqa.has_value() || !dvSplit.has_value() || *dvSplit < 1)
    {
        return std::nullopt;
    }
    return CdnaTile{false, *blockM, *splitKv == 1, *packGqa == 1, *dvSplit};
}

/// The launch grid, or nullopt past int32: the dual-wave kernel one workgroup per (query
/// head, query block, batch); the generic one per (batch, query tile, query head),
/// flattened onto x.
inline std::optional<std::array<int64_t, 3>> candidateGrid(const SdpaProblem& problem,
                                                           const CdnaTile& tile)
{
    const auto queryBlocks = (problem.seqLenQ + tile.blockM - 1) / tile.blockM;
    if(tile.dualwave)
    {
        if(queryBlocks > INT32_LIMIT || problem.numHeads > INT32_LIMIT)
        {
            return std::nullopt;
        }
        return std::array<int64_t, 3>{problem.numHeads, queryBlocks, problem.batch};
    }
    // A packed object: (batch, row tile, KV head), the group's rows packed.
    const auto rows = tile.packGqa
                          ? checkedProduct({problem.numHeads / problem.numKvHeads, problem.seqLenQ})
                          : std::optional<int64_t>(problem.seqLenQ);
    if(!rows.has_value())
    {
        return std::nullopt;
    }
    const auto gridX
        = tile.packGqa
              ? checkedProduct(
                    {problem.batch, (*rows + tile.blockM - 1) / tile.blockM, problem.numKvHeads})
              : checkedProduct({problem.batch, queryBlocks, problem.numHeads});
    if(!gridX.has_value() || *gridX > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return std::array<int64_t, 3>{*gridX, 1, tile.dvSplit};
}

/// How many ways a split object divides the keys; 1 for any other object.
inline int64_t candidateSplits(const SdpaProblem& problem, const CdnaTile& tile, int computeUnits)
{
    const auto grid = candidateGrid(problem, tile);
    if(!tile.splitKv || !grid.has_value())
    {
        return 1;
    }
    const int64_t units = (computeUnits > 0 ? computeUnits : SPLIT_FALLBACK_COMPUTE_UNITS)
                          * SPLIT_WORKGROUPS_PER_UNIT;
    const int64_t byDevice = std::max<int64_t>(1, units / std::max<int64_t>(1, (*grid)[0]));
    const int64_t tiles = (problem.seqLenKv + SPLIT_TILE - 1) / SPLIT_TILE;
    const int64_t byWork = std::max<int64_t>(1, tiles / SPLIT_MIN_TILES);
    return std::min({byDevice, byWork, SPLIT_MAX});
}

/// The split workspace: a normalized f32 partial O per (batch, query head, query
/// position, split), then each partial's LSE; nothing when one split writes O directly.
inline std::optional<size_t> splitWorkspaceBytes(const SdpaProblem& problem, int64_t splits)
{
    if(splits <= 1)
    {
        return size_t{0};
    }
    const auto elements = checkedProduct(
        {problem.batch, problem.numHeads, problem.seqLenQ, splits, problem.headDimV + 1});
    if(!elements.has_value() || *elements * 4 > INT32_LIMIT)
    {
        return std::nullopt;
    }
    return static_cast<size_t>(*elements) * sizeof(float);
}

/// The bytes from @p tensor's first element to one past its last, through @p strides, or
/// nullopt past int64: what one 32-bit buffer view or offset over the whole tensor must
/// reach.
inline std::optional<int64_t> wholeTensorBytes(const OperandStrides& strides,
                                               int64_t batch,
                                               int64_t heads,
                                               int64_t sequence,
                                               int64_t width,
                                               int64_t elementBytes)
{
    int64_t last = width;
    for(const auto& [extent, stride] : {std::pair{batch, strides.batch},
                                        std::pair{heads, strides.head},
                                        std::pair{sequence, strides.sequence}})
    {
        const auto reach = checkedProduct({extent - 1, stride});
        if(!reach.has_value() || *reach > INT32_LIMIT)
        {
            return std::nullopt;
        }
        last += *reach;
    }
    return checkedProduct({last, elementBytes});
}

/// Whether a candidate's addressing reaches every element of Q and O within 32 bits: a
/// row-tiled object's padding rows up to its last tile's end, the merge kernel's store of O
/// at one offset over the whole tensor, and a packed object's view of a KV head's group of
/// Q (and O) heads as one buffer.
inline bool reachesWithin32Bits(const MatchContext& context,
                                const data_objects::SdpaAttributes& attributes,
                                const SdpaProblem& problem,
                                const CdnaTile& tile)
{
    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* o = findTensor(context, attributes.o_tensor_uid());
    if(q == nullptr || o == nullptr)
    {
        return false;
    }
    const auto qStrides
        = servableStrides(*q, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDim);
    const auto oStrides
        = servableStrides(*o, problem.batch, problem.numHeads, problem.seqLenQ, problem.headDimV);
    if(!qStrides.has_value() || !oStrides.has_value())
    {
        return false;
    }
    const auto fits
        = [](std::optional<int64_t> bytes) { return bytes.has_value() && *bytes <= INT32_LIMIT; };
    if(!tile.packGqa)
    {
        // A query tile's padding rows, up to the last tile's end, are addressed at row *
        // sequence stride like the real ones, with 32-bit byte offsets: past 32 bits a padding
        // row's offset wraps back into the slice and its store lands on a real row. So the
        // reach of every row the tiles cover must fit, not only the sequence's.
        const auto paddedRows
            = checkedProduct({(problem.seqLenQ + tile.blockM - 1) / tile.blockM, tile.blockM});
        if(!paddedRows.has_value())
        {
            return false;
        }
        const auto rowsReach = [&](const OperandStrides& strides, int64_t width) {
            const auto span = checkedProduct({*paddedRows - 1, strides.sequence});
            return span.has_value() && *span <= INT32_LIMIT - width
                   && fits(checkedProduct({*span + width, BYTES_PER_ELEMENT}));
        };
        if(!rowsReach(*qStrides, problem.headDim) || !rowsReach(*oStrides, problem.headDimV))
        {
            return false;
        }
    }
    if(tile.splitKv
       && !fits(wholeTensorBytes(*oStrides,
                                 problem.batch,
                                 problem.numHeads,
                                 problem.seqLenQ,
                                 problem.headDimV,
                                 BYTES_PER_ELEMENT)))
    {
        return false;
    }
    if(tile.packGqa)
    {
        // One KV head's group: its query heads and every query row.
        const int64_t group = problem.numHeads / problem.numKvHeads;
        if(!fits(wholeTensorBytes(
               *qStrides, 1, group, problem.seqLenQ, problem.headDim, BYTES_PER_ELEMENT))
           || !fits(wholeTensorBytes(
               *oStrides, 1, group, problem.seqLenQ, problem.headDimV, BYTES_PER_ELEMENT)))
        {
            return false;
        }
    }
    return true;
}

/**
 * @brief Kernel-scoped applicability: do THIS candidate's baked fields fit the graph?
 *
 * dtype, head_dim and the causal flag are baked, so each must equal the graph's; the key length must be a multiple of the candidate's `kv_align`, since an
 * object built without the KV tail mask reads past the end of a ragged last tile; and the
 * candidate must declare a launch shape this pack knows.
 */
inline bool familyKernelMatches(const MatchContext& context,
                                const BoundTokens& bound,
                                const KernelDefinition& kernel,
                                CdnaFamily family)
{
    const auto* attributes = sdpaNode(context);
    if(attributes == nullptr)
    {
        return false;
    }
    const auto* q = findTensor(context, attributes->q_tensor_uid());
    const auto* k = findTensor(context, attributes->k_tensor_uid());
    if(q == nullptr || k == nullptr)
    {
        return false;
    }
    const auto problem = problemFor(*q, *k, findTensor(context, attributes->v_tensor_uid()));
    const auto tile = candidateTile(kernel, family);
    if(!tile.has_value() || !candidateGrid(problem, *tile).has_value())
    {
        return false;
    }
    // A split object carries its merge kernel by name, and its partials must fit the
    // workspace's 32-bit addressing.
    if(tile->splitKv
       && (stringMetadata(kernel, MERGE_SYMBOL_FIELD).value_or("").empty()
           || !splitWorkspaceBytes(
                   problem,
                   candidateSplits(problem, *tile, context.deviceProperties.multiProcessorCount))
                   .has_value()))
    {
        return false;
    }
    if(!reachesWithin32Bits(context, *attributes, problem, *tile))
    {
        return false;
    }

    const auto hasBias = integerMetadata(kernel, HAS_BIAS_FIELD);
    const auto boundBias = tryGetBoundInt(bound, BIAS_TOKEN);
    if(!hasBias.has_value() || !boundBias.has_value()
       || *hasBias != (*boundBias != NO_BIAS ? 1 : 0))
    {
        return false;
    }
    const auto anyDiagonal = familyKnob(kernel, ANY_DIAGONAL_FIELD, family);
    const auto diagOffset = tryGetBoundInt(bound, DIAG_OFFSET_TOKEN);
    if(!anyDiagonal.has_value() || !diagOffset.has_value()
       || (*anyDiagonal == 0 && *diagOffset != 0))
    {
        return false;
    }
    // A sink object for a graph with an attention sink, a plain one for a graph without
    // (vendored-kernel modification 19; the dual-wave kernel builds none).
    const auto sinkUid = tryGetBoundInt(bound, SINK_TOKEN);
    const auto hasSink = familyKnob(kernel, HAS_SINK_FIELD, family);
    if(!sinkUid.has_value() || !hasSink.has_value() || *hasSink != (*sinkUid != NO_SINK ? 1 : 0))
    {
        return false;
    }
    // Only the generic kernel takes a window (vendored-kernel modification 11), or an LSE
    // with strides of its own (modification 20).
    const auto windowLo = tryGetBoundInt(bound, WINDOW_TOKEN);
    if(!windowLo.has_value() || (*windowLo != NO_WINDOW && tile->dualwave))
    {
        return false;
    }
    const auto statsPacked = tryGetBoundInt(bound, STATS_PACKED_TOKEN);
    if(!statsPacked.has_value() || (*statsPacked == 0 && tile->dualwave))
    {
        return false;
    }
    const auto kvAlign = familyKnob(kernel, KV_ALIGN_FIELD, family);
    if(!kvAlign.has_value() || *kvAlign < 0 || (*kvAlign > 0 && problem.seqLenKv % *kvAlign != 0))
    {
        return false;
    }

    // An object built for one head dim serves it exactly; a qk_mask one serves any Q/K head
    // dim up to its bucket that its masks cover. V's must be the object's own -- or, on a
    // qk_mask object whose V is the bucket's width, Q/K's, which it then follows
    // (vendored-kernel modification 17).
    const auto headDim = integerMetadata(kernel, HEAD_DIM_FIELD);
    const auto headDimV = integerMetadata(kernel, HEAD_DIM_V_FIELD);
    const auto qkMask = familyKnob(kernel, QK_MASK_FIELD, family);
    if(!headDim.has_value() || !headDimV.has_value() || !qkMask.has_value())
    {
        return false;
    }
    // V with strides of its own at Q/K's head dim: only an object on the masked path reads
    // V through them.
    const auto vOwnStrides = tryGetBoundInt(bound, V_OWN_STRIDES_TOKEN);
    if(!vOwnStrides.has_value() || (*vOwnStrides == 1 && *qkMask == 0 && *headDimV == *headDim))
    {
        return false;
    }
    const bool vFollowsQk = *qkMask == 1 && *headDimV == *headDim;
    if(vFollowsQk ? problem.headDimV != problem.headDim : *headDimV != problem.headDimV)
    {
        return false;
    }
    // The merge kernel and the split workspace know only the baked V width: a split object
    // whose V follows a narrower runtime width would overrun both.
    if(vFollowsQk && tile->splitKv && problem.headDimV != *headDimV)
    {
        return false;
    }
    const bool servesHead
        = *qkMask == 1 ? problem.headDim <= *headDim && problem.headDim % QK_MASK_ALIGN == 0
                       : *qkMask == 0 && *headDim == problem.headDim;

    const auto dataTypeName = kernelDataTypeName(problem.dataType);
    const auto causal = tryGetBoundInt(bound, CAUSAL_TOKEN);
    return servesHead && dataTypeName.has_value()
           && stringMetadata(kernel, DTYPE_FIELD) == *dataTypeName && causal.has_value()
           && integerMetadata(kernel, CAUSAL_FIELD) == *causal;
}

/// Whether the graph lies in the range @p kernel declares it is preferred for.
inline bool isPreferredFor(const SdpaProblem& problem, const KernelDefinition& kernel)
{
    const auto feature = stringMetadata(kernel, PREFER_FEATURE_FIELD);
    const auto low = integerMetadata(kernel, PREFER_MIN_FIELD);
    const auto high = integerMetadata(kernel, PREFER_MAX_FIELD);
    if(!feature.has_value() || !low.has_value() || !high.has_value())
    {
        return false;
    }
    std::optional<int64_t> value;
    if(*feature == PREFER_ROWS)
    {
        value = checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
    }
    else if(*feature == PREFER_SEQ_Q)
    {
        value = problem.seqLenQ;
    }
    if(!value.has_value())
    {
        // "none", or rows past int64: no preference.
        return false;
    }
    return *value >= *low && (*high == 0 || *value < *high);
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// FlyDSL's dynamic layout for a rank-4 tensor: the four extents, then the strides of the
/// three dimensions whose stride is not 1, in ascending order. Passed by value after each
/// tensor's pointer.
struct FlydslLayout4
{
    std::array<int32_t, 4> extents;
    std::array<int64_t, 3> strides;
};
static_assert(sizeof(FlydslLayout4) == 40, "FlyDSL's rank-4 dynamic layout is 40 bytes");

/// The layout of a contiguous BSHD tensor viewed as [B, S, H, D], as upstream's launchers
/// pass it: strides (S*H*D, H*D, D) for the batch, sequence and head dimensions. The
/// kernels read only each tensor's pointer -- the strides are their own arguments -- so this
/// fills the layout slots in a well-formed way whatever the operands' strides.
inline FlydslLayout4 bshdLayout(int64_t batch, int64_t sequence, int64_t heads, int64_t headDim)
{
    FlydslLayout4 layout{};
    layout.extents[0] = static_cast<int32_t>(batch);
    layout.extents[1] = static_cast<int32_t>(sequence);
    layout.extents[2] = static_cast<int32_t>(heads);
    layout.extents[3] = static_cast<int32_t>(headDim);
    layout.strides[0] = sequence * heads * headDim;
    layout.strides[1] = heads * headDim;
    layout.strides[2] = headDim;
    return layout;
}

/**
 * @brief The argument lists this pack launches with, mirroring launch() one for one.
 *
 * Dual-wave (flash_attn_gfx950.py): Q, K, V, O, LSE, DebugCounts, CuSeqQ, CuSeqKv,
 * BlockTable, Bias, AlibiSlopes, Sink (each a pointer and its layout); then seq_len,
 * seq_len_kv, stride_q_n, stride_kv_n, head_dim_runtime, block_table_stride,
 * bias_stride0, alibi_stride_b (i32); sm_scale (f32); num_heads_q, num_heads_kv, lse_on,
 * diag_offset, then the strides stride_q_b, stride_q_h, stride_o_b, stride_o_n,
 * stride_o_h, stride_kv_b, stride_kv_h, stride_bias_b, stride_bias_h (i32); its bias
 * row stride is bias_stride0. Generic (flash_attn_generic.py): Q, K, V, O,
 * LSE; seq_len, seq_len_kv (i32); CuSeqQ, CuSeqKv, BlockTable; block_table_stride (i32);
 * sm_scale (f32); num_heads_q, num_heads_kv, stride_q_n, stride_kv_n, lse_on, diag_offset
 * and the same seven strides (i32), then the window, the bias and its strides, V's strides,
 * the Q/K head dim, the split workspace and count, the sink and its stride and switch,
 * and the LSE's batch, head and token strides (i32). A 4-byte scalar slot is the same to the
 * signature check whichever type it holds, so each list is written as runs of tensors
 * and scalars.
 */
struct ArgumentRun
{
    int count;
    bool tensors;
};

inline std::vector<KernelArgument> signatureOf(std::initializer_list<ArgumentRun> runs)
{
    const KernelArgument pointer{"global_buffer", static_cast<uint32_t>(sizeof(void*)), 0, ""};
    const KernelArgument layout{"by_value", static_cast<uint32_t>(sizeof(FlydslLayout4)), 0, ""};
    const KernelArgument i32{"by_value", static_cast<uint32_t>(sizeof(int32_t)), 0, ""};
    std::vector<KernelArgument> signature;
    for(const auto& run : runs)
    {
        for(int i = 0; i < run.count; ++i)
        {
            if(run.tensors)
            {
                signature.push_back(pointer);
                signature.push_back(layout);
            }
            else
            {
                signature.push_back(i32);
            }
        }
    }
    return signature;
}

inline const std::vector<KernelArgument>& dualwaveSignature()
{
    static const std::vector<KernelArgument> s_signature = signatureOf({{12, true}, {22, false}});
    return s_signature;
}

inline const std::vector<KernelArgument>& genericSignature()
{
    static const std::vector<KernelArgument> s_signature = signatureOf({{5, true},
                                                                        {2, false},
                                                                        {3, true},
                                                                        {16, false},
                                                                        {1, true},
                                                                        {7, false},
                                                                        {2, true},
                                                                        {1, false},
                                                                        {1, true},
                                                                        {2, false},
                                                                        {3, false}});
    return s_signature;
}

/// The split objects' merge kernel: O, LSE, WsO, WsLse; then seq_len, num_heads_q, lse_on,
/// num_splits and O's batch, token and head strides (i32); the Sink, its head stride and
/// sink_on; and the LSE's batch, head and token strides (i32).
inline const std::vector<KernelArgument>& mergeSignature()
{
    static const std::vector<KernelArgument> s_signature
        = signatureOf({{4, true}, {7, false}, {1, true}, {2, false}, {3, false}});
    return s_signature;
}

inline const data_objects::TensorAttributes& requireTensor(const MatchContext& context, int64_t uid)
{
    return requireSdpaTensor(context, uid, "flydsl_sdpa_cdna");
}

/// The compiled kernel plus everything launch() needs.
class PreparedFlydslSdpaCdna : public PreparedDispatch
{
public:
    PreparedFlydslSdpaCdna(IngestorKernelCode code,
                           FlydslSdpaCdnaBinding binding,
                           SdpaProblem problem,
                           bool dualwave,
                           std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand,
                           ViewStrides strides,
                           BiasStrides biasStrides,
                           int64_t splits = 1,
                           std::optional<IngestorKernelCode> mergeCode = std::nullopt)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _dualwave(dualwave)
        , _scaleOperand(scaleOperand)
        , _strides(strides)
        , _biasStrides(biasStrides)
        , _splits(splits)
        , _mergeCode(std::move(mergeCode))
    {
    }

    /// How many ways the keys are split; the merge kernel runs when more than one.
    int64_t splits() const
    {
        return _splits;
    }

    compilation::IRunnableKernel& mergeForStream(hipStream_t stream) const
    {
        return _mergeCode->kernelForStream(stream);
    }

    const BiasStrides& biasStrides() const
    {
        return _biasStrides;
    }

    const ViewStrides& strides() const
    {
        return _strides;
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const FlydslSdpaCdnaBinding& binding() const
    {
        return _binding;
    }

    const SdpaProblem& problem() const
    {
        return _problem;
    }

    bool dualwave() const
    {
        return _dualwave;
    }

    /// Set when the scale is a pass-by-value tensor, read at each launch.
    const std::optional<hipdnn_plugin_sdk::ScalarOperand>& scaleOperand() const
    {
        return _scaleOperand;
    }

private:
    IngestorKernelCode _code;
    FlydslSdpaCdnaBinding _binding;
    SdpaProblem _problem;
    bool _dualwave = false;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> _scaleOperand;
    ViewStrides _strides;
    BiasStrides _biasStrides;
    int64_t _splits = 1;
    std::optional<IngestorKernelCode> _mergeCode;
};

/**
 * @brief The native dispatch behind this pack's UDD. Everything graph/kernel-derived
 *        resolves once at prepare(); launch() only resolves buffers and the scale.
 */
class FlydslSdpaCdnaDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    FlydslSdpaCdnaDispatchHandler(const compilation::IKernelCompiler& kernelCompiler,
                                  const compilation::KpackKernelLoader& kpackLoader,
                                  CdnaFamily family)
        : _kernelCompiler(kernelCompiler)
        , _kpackLoader(kpackLoader)
        , _family(family)
    {
    }

    /// Only a split object launched with more than one split needs scratch: the partials.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto tile = candidateTile(kernel, _family);
        if(!tile.has_value() || !tile->splitKv)
        {
            return 0;
        }
        const auto binding = flydslSdpaCdnaBinding(bound);
        const auto problem = problemFor(requireTensor(context, binding.q),
                                        requireTensor(context, binding.k),
                                        &requireTensor(context, binding.v));
        const auto bytes = splitWorkspaceBytes(
            problem, candidateSplits(problem, *tile, context.deviceProperties.multiProcessorCount));
        if(!bytes.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: the split workspace exceeds 32-bit addressing");
        }
        return *bytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = flydslSdpaCdnaBinding(bound);
        const auto tile = candidateTile(kernel, _family);
        if(!tile.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: kernel '" + toString(kernel.kernelId)
                    + "' declares no kernel family or block_m this pack can launch");
        }
        const auto problem = problemFor(requireTensor(context, binding.q),
                                        requireTensor(context, binding.k),
                                        &requireTensor(context, binding.v));
        const auto grid = candidateGrid(problem, *tile);
        if(!grid.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: launch grid exceeds int32 for kernel '"
                    + toString(kernel.kernelId) + "'");
        }

        const compilation::KernelCompileOptions options(context.deviceProperties.gcnArchName);
        auto code
            = buildIngestorKernelCode(_kernelCompiler,
                                      _kpackLoader,
                                      context,
                                      kernel,
                                      options,
                                      tile->dualwave ? dualwaveSignature() : genericSignature());
        const int64_t threads
            = tile->dualwave ? DUALWAVE_BLOCK_THREADS : tile->blockM * GENERIC_THREADS_PER_ROW;
        // A split object runs one grid row per split (vendored-kernel modification 14).
        const auto splits
            = candidateSplits(problem, *tile, context.deviceProperties.multiProcessorCount);
        code.setBlockSize(static_cast<unsigned int>(threads), 1, 1);
        code.setGridSize(static_cast<unsigned int>((*grid)[0]),
                         static_cast<unsigned int>((*grid)[1] * splits),
                         static_cast<unsigned int>((*grid)[2]));
        // The merge kernel is the same code object under another symbol.
        std::optional<IngestorKernelCode> mergeCode;
        if(splits > 1)
        {
            const auto mergeSymbol = stringMetadata(kernel, MERGE_SYMBOL_FIELD);
            const auto mergeGrid
                = checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
            if(!mergeSymbol.has_value() || mergeSymbol->empty() || !mergeGrid.has_value()
               || *mergeGrid > INT32_LIMIT)
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_cdna: split kernel '" + toString(kernel.kernelId)
                        + "' names no merge kernel, or its launch overflows");
            }
            KernelDefinition mergeKernel = kernel;
            mergeKernel.source.symbol = *mergeSymbol;
            mergeKernel.source.signature = mergeSignature();
            mergeCode.emplace(buildIngestorKernelCode(
                _kernelCompiler, _kpackLoader, context, mergeKernel, options, mergeSignature()));
            mergeCode->setBlockSize(static_cast<unsigned int>(MERGE_THREADS), 1, 1);
            mergeCode->setGridSize(static_cast<unsigned int>(*mergeGrid), 1, 1);
        }
        std::optional<hipdnn_plugin_sdk::ScalarOperand> scaleOperand;
        if(binding.scaleSource == ScaleSource::TENSOR)
        {
            scaleOperand = hipdnn_plugin_sdk::makeScalarOperand(
                context.graph.getTensorMap(), binding.scale, "scale");
        }
        auto strides = viewStridesFor(requireTensor(context, binding.q),
                                      requireTensor(context, binding.k),
                                      requireTensor(context, binding.v),
                                      requireTensor(context, binding.o),
                                      problem);
        if(!strides.has_value())
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "flydsl_sdpa_cdna: the bound operands' strides are not ones the kernels read");
        }
        if(binding.stats != NO_STATS)
        {
            const auto lse = servableStatsStrides(requireTensor(context, binding.stats),
                                                  problem.batch,
                                                  problem.numHeads,
                                                  problem.seqLenQ);
            if(!lse.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_cdna: the bound LSE output is not one the kernels write");
            }
            strides->lse = *lse;
        }
        BiasStrides biasStrides;
        if(binding.bias != NO_BIAS)
        {
            const auto servable = servableBias(requireTensor(context, binding.bias), problem);
            if(!servable.has_value())
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_cdna: the bound bias is not one the kernels can add");
            }
            biasStrides = *servable;
        }
        return std::make_unique<PreparedFlydslSdpaCdna>(std::move(code),
                                                        binding,
                                                        problem,
                                                        tile->dualwave,
                                                        scaleOperand,
                                                        *strides,
                                                        biasStrides,
                                                        splits,
                                                        std::move(mergeCode));
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedSdpa = dynamic_cast<const PreparedFlydslSdpaCdna&>(prepared);
        const auto& binding = preparedSdpa.binding();
        const auto& problem = preparedSdpa.problem();

        float scale = 0.0F;
        if(preparedSdpa.scaleOperand().has_value())
        {
            scale = static_cast<float>(
                hipdnn_plugin_sdk::toDouble(hipdnn_plugin_sdk::resolveScalarOperand(
                    *preparedSdpa.scaleOperand(), deviceBuffers, numDeviceBuffers)));
            // A pass-by-value scalar, read from host memory. A constant scale was checked at
            // match time; this one arrives now.
            if(!isServableScale(static_cast<double>(scale)))
            {
                throw hipdnn_plugin_sdk::HipdnnPluginException(
                    HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                    "flydsl_sdpa_cdna: the runtime softmax scale must be positive and at most "
                    "2^20");
            }
        }
        else
        {
            const auto scaleBits = static_cast<int32_t>(binding.scale);
            std::memcpy(&scale, &scaleBits, sizeof(scale));
        }

        void* q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers).ptr;
        void* k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers).ptr;
        void* v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers).ptr;
        void* o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers).ptr;
        // The objects carry the LSE store; with no stats tensor bound it gets an empty
        // buffer (lse_on 0) and every LSE store is dropped, so O stands in for the pointer.
        const bool lseOn = binding.stats != NO_STATS;
        void* lse = lseOn ? hipdnn_plugin_sdk::findDeviceBuffer(
                                binding.stats, deviceBuffers, numDeviceBuffers)
                                .ptr
                          : o;
        const int32_t lseFlag = lseOn ? 1 : 0;
        // The bias, when one is bound; otherwise O stands in, and the objects built
        // without a bias never read it.
        void* bias = binding.bias != NO_BIAS ? hipdnn_plugin_sdk::findDeviceBuffer(
                                                   binding.bias, deviceBuffers, numDeviceBuffers)
                                                   .ptr
                                             : o;
        const auto& biasStrides = preparedSdpa.biasStrides();

        const auto qLayout
            = bshdLayout(problem.batch, problem.seqLenQ, problem.numHeads, problem.headDim);
        const auto kvLayout
            = bshdLayout(problem.batch, problem.seqLenKv, problem.numKvHeads, problem.headDim);
        const auto seqLenQ = static_cast<int32_t>(problem.seqLenQ);
        const auto seqLenKv = static_cast<int32_t>(problem.seqLenKv);
        const auto diagOffset = static_cast<int32_t>(binding.diagOffset);
        const auto numHeads = static_cast<int32_t>(problem.numHeads);
        const auto numKvHeads = static_cast<int32_t>(problem.numKvHeads);
        const auto& strides = preparedSdpa.strides();
        const auto i32 = [](int64_t value) { return static_cast<int32_t>(value); };
        const int32_t zero = 0;
        auto& kernel = preparedSdpa.kernelForStream(handle.getStream());
        // The attention sink, when one is bound; otherwise O stands in and sink_on is 0.
        const bool hasSink = binding.sink != NO_SINK;
        void* sink = hasSink ? hipdnn_plugin_sdk::findDeviceBuffer(
                                   binding.sink, deviceBuffers, numDeviceBuffers)
                                   .ptr
                             : o;
        const auto sinkStride = static_cast<int32_t>(binding.sinkStride);
        const int32_t sinkOn = hasSink ? 1 : 0;
        // The split workspace: the partial O, then the partial LSEs. Unread with one split,
        // when O stands in.
        const auto splits = static_cast<int32_t>(preparedSdpa.splits());
        void* wsO = splits > 1 ? workspace : o;
        void* wsLse = splits > 1
                          ? static_cast<void*>(
                                static_cast<char*>(workspace)
                                + static_cast<size_t>(problem.batch * problem.numHeads
                                                      * problem.seqLenQ * splits * problem.headDimV)
                                      * sizeof(float))
                          : o;

        // A tensor the launch does not use is passed as O, as upstream's launchers do; the
        // kernels as built never read it. Changing these argument lists means changing
        // dualwaveSignature() / genericSignature() with them.
        if(preparedSdpa.dualwave())
        {
            const auto headDim = static_cast<int32_t>(problem.headDim);
            kernel.launch(handle.getStream(),
                          q,
                          qLayout,
                          k,
                          kvLayout,
                          v,
                          kvLayout,
                          o,
                          qLayout,
                          lse,
                          qLayout, // read for its pointer only
                          o, // DebugCounts
                          qLayout,
                          o, // CuSeqQ
                          qLayout,
                          o, // CuSeqKv
                          qLayout,
                          o, // BlockTable
                          qLayout,
                          bias,
                          qLayout, // read for its pointer only
                          o, // AlibiSlopes
                          qLayout,
                          o, // Sink
                          qLayout,
                          seqLenQ,
                          seqLenKv,
                          i32(strides.q.sequence),
                          i32(strides.kv.sequence),
                          headDim,
                          zero,
                          i32(biasStrides.query),
                          zero,
                          scale,
                          numHeads,
                          numKvHeads,
                          lseFlag,
                          diagOffset,
                          i32(strides.q.batch),
                          i32(strides.q.head),
                          i32(strides.o.batch),
                          i32(strides.o.sequence),
                          i32(strides.o.head),
                          i32(strides.kv.batch),
                          i32(strides.kv.head),
                          i32(biasStrides.batch),
                          i32(biasStrides.head));
            return;
        }
        kernel.launch(handle.getStream(),
                      q,
                      qLayout,
                      k,
                      kvLayout,
                      v,
                      kvLayout,
                      o,
                      qLayout,
                      lse,
                      qLayout, // read for its pointer only
                      seqLenQ,
                      seqLenKv,
                      o, // CuSeqQ
                      qLayout,
                      o, // CuSeqKv
                      qLayout,
                      o, // BlockTable
                      qLayout,
                      zero,
                      scale,
                      numHeads,
                      numKvHeads,
                      i32(strides.q.sequence),
                      i32(strides.kv.sequence),
                      lseFlag,
                      diagOffset,
                      i32(strides.q.batch),
                      i32(strides.q.head),
                      i32(strides.o.batch),
                      i32(strides.o.sequence),
                      i32(strides.o.head),
                      i32(strides.kv.batch),
                      i32(strides.kv.head),
                      static_cast<int32_t>(binding.windowLo),
                      bias,
                      qLayout, // read for its pointer only
                      i32(biasStrides.query),
                      i32(biasStrides.batch),
                      i32(biasStrides.head),
                      i32(strides.v.sequence),
                      i32(strides.v.batch),
                      i32(strides.v.head),
                      static_cast<int32_t>(problem.headDim),
                      wsO,
                      qLayout, // read for its pointer only
                      wsLse,
                      qLayout,
                      splits,
                      sink,
                      qLayout, // read for its pointer only
                      sinkStride,
                      sinkOn,
                      i32(strides.lse.batch),
                      i32(strides.lse.head),
                      i32(strides.lse.sequence));
        if(splits > 1)
        {
            // Changing this argument list means changing mergeSignature() with it.
            preparedSdpa.mergeForStream(handle.getStream())
                .launch(handle.getStream(),
                        o,
                        qLayout,
                        lse,
                        qLayout,
                        wsO,
                        qLayout,
                        wsLse,
                        qLayout,
                        seqLenQ,
                        numHeads,
                        lseFlag,
                        splits,
                        i32(strides.o.batch),
                        i32(strides.o.sequence),
                        i32(strides.o.head),
                        sink,
                        qLayout,
                        sinkStride,
                        sinkOn,
                        i32(strides.lse.batch),
                        i32(strides.lse.head),
                        i32(strides.lse.sequence));
        }
    }

private:
    const compilation::IKernelCompiler& _kernelCompiler;
    const compilation::KpackKernelLoader& _kpackLoader;
    CdnaFamily _family;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_cdna

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
