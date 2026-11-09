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
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/RuntimePassByValue.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/FlydslSdpaBwdCommon.hpp"
#include "engines/kernel_ingestor_engine/packs/FlydslSdpaDropout.hpp"

/**
 * @file FlydslSdpaBwdNativeShared.hpp
 * @brief What every FlyDSL SDPA backward native handler computes the same way, whatever the
 *        launch it makes: the workspace layout, the window values, the dBias reduce launch and
 *        its arguments, the seed's workspace binding and upload, and the kernel argument
 *        machinery (signature, image).
 *
 * A handler owns its kernel's argument struct and fills it from these values; one launch per
 * handler, the by-symbol dBias reduce a second launch of the same code object. The workspace
 * layout is the one place every region's size and offset is decided, with checked arithmetic.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_bwd_native
{

namespace bwd = flydsl_sdpa_bwd;

using bwd::detail::INT32_LIMIT;

/// The largest y or z extent of a launch grid; the same limit as `GRID_YZ_LIMIT` in
/// FlydslSdpaAiwNative.cpp.
constexpr int64_t GRID_YZ_LIMIT = 65535;

/// The mask sentinels the kernel resolves per sequence (`fmha_common_gfx1201.py`,
/// WINDOW_TOPLEFT and WINDOW_BOTRIGHT).
constexpr int32_t WINDOW_TOPLEFT = -2147483647;
constexpr int32_t WINDOW_BOTRIGHT = -2147483646;

/// The workspace's seed region, `[seed u64 | offset u64]`, and the alignment of every region
/// after the first.
constexpr size_t SEED_REGION_BYTES = 16;
constexpr size_t SEED_WORD_BYTES = static_cast<size_t>(bwd::detail::INT64_BYTES);
constexpr size_t WORKSPACE_ALIGNMENT = 256;
constexpr int64_t WORKSPACE_ELEMENT_BYTES = 2; // bf16 and f16 only
/// Delta and LSE' are fp32 `[B, Hq, Sq]`, indexed (batch, head, in-sequence row) at pitch Sq
/// whatever the layout of the caller's stats.
constexpr int64_t ROW_STATISTIC_BYTES = bwd::detail::STATS_BYTES_PER_ELEMENT;

/// The reduce kernel's launch: 256 threads, each owning 8 consecutive columns of one row
/// (`fmha_bwd_dbias_reduce_gfx950.py`, BLOCK_THREADS and VEC).
constexpr unsigned int REDUCE_THREADS = 256;
constexpr int64_t REDUCE_VECTOR = 8;

// Kernel metadata fields, declared by the KMD each pack's KDP points at.
constexpr std::string_view DTYPE_FIELD = "dtype";
/// The head-dim tile the object is built at.
constexpr std::string_view HEAD_DIM_FIELD = "head_dim";
constexpr std::string_view CAUSAL_FIELD = "causal";
constexpr std::string_view HAS_BIAS_FIELD = "has_bias";
constexpr std::string_view DROPOUT_FIELD = "has_dropout";
/// 0: serves head dims equal to the tile; 1: serves the dims between the rung below and it.
constexpr std::string_view PADDED_FIELD = "padded";
/// -1 leaves the dB store to the policy (it exists); 0 builds a dQ role without it.
constexpr std::string_view STORE_DB_FIELD = "store_db";
/// 1 packs the GQA group into the dK/dV role's q-tile row axis; -1 leaves it unset.
constexpr std::string_view GQA_PACK_FIELD = "gqa_pack";
/// The query tile of the dK/dV role; a packed row carries its resolved value, -1 leaves it
/// unset.
constexpr std::string_view BLOCK_Q_FIELD = "block_q";
/// The launch geometry of the build: threads per block, and the key and query tile heights.
constexpr std::string_view BLOCK_THREADS_FIELD = "block_threads";
constexpr std::string_view BLOCK_KV_FIELD = "block_kv";
constexpr std::string_view BLOCK_M_FIELD = "block_m";
/// The symbol of the row's dBias reduce kernel, which its `dbias_reduce` launch names too;
/// empty on a row without one.
constexpr std::string_view REDUCE_SYMBOL_FIELD = "reduce_symbol";

/// The launch names every row's `sources` carry besides the kernel's own.
constexpr std::string_view REDUCE_LAUNCH = "dbias_reduce";
constexpr std::string_view TAILZERO_LAUNCH = "tailzero";

/// The graph classes an object is built for: `causal` serves every masked graph, `bias`
/// every bias graph (masked graphs with a bias are declined by the matcher), `plain` the rest.
constexpr int64_t CLASS_PLAIN = 0;
constexpr int64_t CLASS_CAUSAL = 1;
constexpr int64_t CLASS_BIAS = 2;

/// The head-dim tiles `_sdpa_bwd_spec.LADDER` builds objects at.
constexpr std::array<int64_t, 10> LADDER{32, 64, 96, 128, 160, 192, 224, 256, 384, 512};

/// `rt_flags`: bit 0 serves a head dim below the tile, bit 1 turns dropout on.
constexpr int32_t RT_FLAG_PADDED = 1;
constexpr int32_t RT_FLAG_DROPOUT = 2;

/// The packed GQA query tile that `packedSpansFit` assumes: the largest block_q of the dK/dV
/// role among the shipped gqa_pack == 1 objects.
constexpr int64_t PACK_MAX_BLOCK_Q = 64;

/// The tensor uids a matched graph binds.
struct Uids
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t dO = 0;
    int64_t dQ = 0;
    int64_t dK = 0;
    int64_t dV = 0;
    int64_t stats = 0;
};

/// A BAD_PARAM exception whose message reads "<scope>: <message>"; @p scope names the engine.
[[noreturn]] inline void failBadParam(std::string_view scope, const std::string& message)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   std::string(scope) + ": " + message);
}

[[noreturn]] inline void failBadParam(const std::string& message)
{
    failBadParam("flydsl_sdpa_bwd", message);
}

/// "<engine> declined the graph [<rule or cause>]: <detail>", at INFO: hipDNN reports a
/// declined graph to its caller only as "No engine configurations available".
inline void logDecline(std::string_view engine, std::string_view cause, const std::string& detail)
{
    HIPDNN_PLUGIN_LOG_INFO(engine << " declined the graph [" << cause << "]: " << detail);
}

inline void requireHip(hipError_t status, const char* what)
{
    if(status != hipSuccess)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            std::string("flydsl_sdpa_bwd: ") + what + " failed: " + hipGetErrorString(status));
    }
}

inline int32_t toInt32(int64_t value, const char* what)
{
    if(value < std::numeric_limits<int32_t>::min() || value > std::numeric_limits<int32_t>::max())
    {
        failBadParam(std::string(what) + " " + std::to_string(value) + " does not fit in int32");
    }
    return static_cast<int32_t>(value);
}

// ---------------------------------------------------------------------------
// Launch values
// ---------------------------------------------------------------------------

/// The output (batch, head) planes of the dBias reduce launch, one grid row each: the axes
/// the dBias keeps. 0 when the graph's dBias is not broadcast; int64 max on overflow.
inline int64_t reducePlanes(const bwd::SdpaBwdProblem& problem)
{
    if(problem.dbiasUid == bwd::NO_UID
       || !(problem.biasBroadcastBatch || problem.biasBroadcastHead))
    {
        return 0;
    }
    const auto planes
        = bwd::detail::checkedProduct({problem.biasBroadcastBatch ? int64_t{1} : problem.batch,
                                       problem.biasBroadcastHead ? int64_t{1} : problem.numHeads});
    return planes.value_or(std::numeric_limits<int64_t>::max());
}

/// The kernel's window bounds: `column j is kept for row i iff i - left <= j <= i + right`.
struct Window
{
    int32_t left = 0;
    int32_t right = 0;
};

/// The bounds a graph asks for in the kernel's convention. A graph without a mask passes 0, 0
/// (the object has no window). Plain causal, no left bound and right 0, passes the
/// sentinel of its alignment for both bounds, which the kernel resolves against each
/// sequence's own lengths. Every other mask passes literals: cuDNN's right bound plus the
/// diagonal offset, clamped to the key length; cuDNN's left bound less one and less the
/// offset, clamped to the query length plus the offset's magnitude. A side the graph leaves
/// unbounded is its clamp, and both clamps keep every value int32-safe.
inline Window windowFor(const bwd::SdpaBwdProblem& problem)
{
    if(!problem.masked)
    {
        return {};
    }
    if(problem.windowLeft == bwd::NO_WINDOW && problem.windowRight == 0)
    {
        const int32_t sentinel = problem.alignBottomRight ? WINDOW_BOTRIGHT : WINDOW_TOPLEFT;
        return {sentinel, sentinel};
    }
    const int64_t offset = problem.diagonalOffset;
    const int64_t leftClamp = problem.seqLenQ + (offset < 0 ? -offset : offset);
    const int64_t rightClamp = problem.seqLenKv;

    int64_t left = leftClamp;
    if(problem.windowLeft != bwd::NO_WINDOW && problem.windowLeft <= leftClamp + 1 + offset)
    {
        left = problem.windowLeft - 1 - offset;
    }
    int64_t right = rightClamp;
    if(problem.windowRight != bwd::NO_WINDOW && problem.windowRight <= rightClamp - offset)
    {
        right = problem.windowRight + offset;
    }
    return {toInt32(left, "window_left"), toInt32(right, "window_right")};
}

/// The strides a kernel is given for a tensor of @p rows sequence rows. The matcher records 0
/// on a unit axis. A unit sequence axis keeps a row pitch instead: the kernels drop padding
/// rows through the buffer bound rows * stride_seq, which a zero stride collapses onto row 0.
/// The pitch is the head dim rounded to @p vector elements (8 for 16-bit, 4 for fp32), which
/// keeps the vector alignment of every other stride. The matcher declines a zero sequence stride on a multi-row axis, so the
/// substitution keys on the extent.
inline bwd::Strides rowPitchStrides(bwd::Strides strides,
                                    int64_t rows,
                                    int64_t headDim,
                                    int64_t vector = bwd::detail::VECTOR_ELEMENTS)
{
    if(rows == 1)
    {
        strides.seq = bwd::detail::ceilToVector(headDim, vector);
    }
    return strides;
}

/// The stats' strides for a kernel that bounds its stats reads by rows * stride_seq (the
/// merged kernels' LSE span). The matcher records 0 on a unit axis, which collapses that span
/// to zero bytes and reads every LSE as 0; a unit sequence axis keeps the contiguous unit
/// pitch of dense stats instead. The split prep kernel reads row 0 at offset 0 and bounds
/// nothing by the stride, so it takes the stats as recorded.
inline bwd::Strides rowStatsStrides(bwd::Strides strides, int64_t rows)
{
    if(rows == 1 && strides.seq == 0)
    {
        strides.seq = 1;
    }
    return strides;
}

/// The `varlen_bits` of one side: the individual lengths of the `seq_len` tensor with the
/// token positions the side's element offsets give (`VARLEN_SEQUSED_PACKED_SIDE`, 0x15), or
/// the lengths alone with implied positions (`VARLEN_SEQUSED_CACHE_SIDE`, 0x04). Bits 0..7
/// are the Q side, bits 8..15 the K side; the lengths go in `seqinfo_q0` / `seqinfo_k0` and
/// the offsets of a ragged side in `seqinfo_q1` / `seqinfo_k1`. A graph without padding_mask
/// passes 0. The stats' layout is not a bit: the kernels take its strides.
constexpr int32_t VARLEN_SIDE_ARRAY = 0x15;
constexpr int32_t VARLEN_SIDE_LENGTHS = 0x04;

inline int32_t varlenBitsOf(const bwd::SdpaBwdProblem& problem)
{
    if(!problem.paddingMask)
    {
        return 0;
    }
    const int32_t qSide = problem.ragged.q.ragged() ? VARLEN_SIDE_ARRAY : VARLEN_SIDE_LENGTHS;
    const int32_t kSide = problem.ragged.k.ragged() ? VARLEN_SIDE_ARRAY : VARLEN_SIDE_LENGTHS;
    return qSide | (kSide << 8);
}

/// One tensor's offset arguments: the INT32 `[B + 1]` offset array and its multiplier, the
/// base of batch b being offset[b] * multiplier elements. A dense tensor passes a null
/// pointer and multiplier 0, and is addressed at b * its batch stride.
struct OffsetArgs
{
    void* offset = nullptr;
    int64_t multiplier = 0;
};

/// The offset arguments of @p ragged, resolving its offset tensor with @p buffer.
template <typename Buffer>
OffsetArgs offsetArgsOf(const bwd::RaggedOffset& ragged, const Buffer& buffer)
{
    if(!ragged.ragged())
    {
        return {};
    }
    return {buffer(ragged.uid), ragged.multiplier};
}

/// The arguments of the caller's stats a kernel reads: its offset and multiplier (null and 0
/// when the stats are dense) and its element strides. The address of (batch b, head h, row
/// r) is offset[b] * multiplier + h * head + r * seq when ragged, b * batch + h * head + r *
/// seq when dense.
struct StatsArgs
{
    OffsetArgs offset;
    bwd::Strides strides;
};

template <typename Buffer>
StatsArgs statsArgsOf(const bwd::SdpaBwdProblem& problem, const Buffer& buffer)
{
    return {offsetArgsOf(problem.ragged.stats, buffer), problem.stats};
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// A byte range of the workspace.
struct Region
{
    size_t offset = 0;
    size_t bytes = 0;

    size_t end() const
    {
        return offset + bytes;
    }
};

/// Which optional shared regions a handler's launches need.
struct WorkspaceRequest
{
    /// Delta and LSE' (fp32 `[B, Hq, Sq]` each), written by a prep launch and read by the
    /// dK/dV and dQ launches.
    bool rowStatistics = false;
    /// Bytes of one element of the broadcast dBias region: its dtype's, 2 for f16 and bf16.
    int64_t dbiasElementBytes = WORKSPACE_ELEMENT_BYTES;
};

/// Where the workspace's regions are: the one place its size and its offsets are decided,
/// read by workspaceBytes() and by every launch.
///
/// Shared regions come first, each after the one before on a 256-byte boundary: `[seed u64 |
/// offset u64]` at offset 0 when the seed and offset are not 64-bit device tensors, then
/// Delta, then LSE' when requested. They end at `sharedBytes`. The scratch region, the
/// full-shape dBias buffer when dBias is broadcast over batch or head, starts at
/// `scratchOffset`, `sharedBytes` rounded up to 256. The dBias region is never zeroed: the dQ
/// role writes every in-range element of it exactly once. Every offset and extent is a checked
/// sum, so every region ends within `totalBytes`.
struct WorkspaceLayout
{
    bool uploadsSeed = false;
    bool reduces = false;
    Region seed;
    Region delta;
    Region lse;
    size_t sharedBytes = 0;
    size_t scratchOffset = 0;
    /// Meaningful when `reduces`; its offset is `scratchOffset`.
    Region dbias;
    /// (B, Hq, Sq, Sk) strides of the dBias region, in elements.
    bwd::Strides dbiasStrides;
    size_t totalBytes = 0;
};

inline size_t alignedUp(size_t value)
{
    return (value + WORKSPACE_ALIGNMENT - 1) / WORKSPACE_ALIGNMENT * WORKSPACE_ALIGNMENT;
}

inline WorkspaceLayout workspaceLayout(const bwd::SdpaBwdProblem& problem,
                                       const WorkspaceRequest& request = {})
{
    WorkspaceLayout layout;
    // A device tensor (INT64 only) is read in place; a pass-by-value one is written here.
    layout.uploadsSeed = problem.seedForm == bwd::SeedForm::PASS_BY_VALUE;
    if(layout.uploadsSeed)
    {
        layout.seed = {0, SEED_REGION_BYTES};
        layout.sharedBytes = layout.seed.end();
    }

    if(request.rowStatistics)
    {
        const auto rows
            = bwd::detail::checkedProduct({problem.batch, problem.numHeads, problem.seqLenQ});
        const auto bytes = rows ? bwd::detail::checkedProduct({*rows, ROW_STATISTIC_BYTES})
                                : std::optional<int64_t>{};
        if(!bytes)
        {
            failBadParam("the Delta and LSE' workspace overflows int64");
        }
        const auto place = [&](Region& region) {
            region = {layout.sharedBytes == 0 ? 0 : alignedUp(layout.sharedBytes),
                      static_cast<size_t>(*bytes)};
            layout.sharedBytes = region.end();
        };
        place(layout.delta);
        place(layout.lse);
    }
    layout.scratchOffset = alignedUp(layout.sharedBytes);
    layout.totalBytes = layout.sharedBytes;

    layout.reduces = problem.dbiasUid != bwd::NO_UID
                     && (problem.biasBroadcastBatch || problem.biasBroadcastHead);
    if(!layout.reduces)
    {
        return layout;
    }
    const auto plane = bwd::detail::checkedProduct({problem.seqLenQ, problem.seqLenKv});
    const auto head = plane ? bwd::detail::checkedProduct({problem.numHeads, *plane})
                            : std::optional<int64_t>{};
    const auto elements
        = head ? bwd::detail::checkedProduct({problem.batch, *head}) : std::optional<int64_t>{};
    const auto bytes = elements
                           ? bwd::detail::checkedProduct({*elements, request.dbiasElementBytes})
                           : std::optional<int64_t>{};
    if(!bytes)
    {
        failBadParam("the broadcast dBias workspace overflows int64");
    }
    layout.dbiasStrides = {*head, *plane, problem.seqLenKv};
    layout.dbias = {layout.scratchOffset, static_cast<size_t>(*bytes)};
    layout.totalBytes = layout.dbias.end();
    return layout;
}

/// The address @p offset bytes into @p workspace.
inline void* inWorkspace(void* workspace, size_t offset)
{
    return static_cast<void*>(static_cast<char*>(workspace) + offset);
}

/// Declines a launch handed no workspace when the layout needs one.
inline void requireWorkspace(const WorkspaceLayout& layout, const void* workspace)
{
    if(workspace == nullptr && layout.totalBytes > 0)
    {
        failBadParam("the launch needs its " + std::to_string(layout.totalBytes)
                     + " byte workspace");
    }
}

// ---------------------------------------------------------------------------
// Scalar operands and the seed
// ---------------------------------------------------------------------------

/// The graph's scalar tensors a launch resolves at execute time.
struct ScalarOperands
{
    /// Set for a tensor-form scale; the value form is problem.scaleValue.
    std::optional<hipdnn_plugin_sdk::ScalarOperand> scale;
    /// Set for a pass-by-value seed and offset.
    std::optional<hipdnn_plugin_sdk::ScalarOperand> seed;
    std::optional<hipdnn_plugin_sdk::ScalarOperand> offset;
};

template <typename TensorMap>
ScalarOperands scalarOperandsOf(const TensorMap& tensors, const bwd::SdpaBwdProblem& problem)
{
    ScalarOperands operands;
    if(problem.scaleSource == bwd::ScaleSource::TENSOR)
    {
        operands.scale = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.scaleUid, "scale");
    }
    if(problem.seedForm == bwd::SeedForm::PASS_BY_VALUE)
    {
        operands.seed = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.seedUid, "seed");
        operands.offset
            = hipdnn_plugin_sdk::makeScalarOperand(tensors, problem.offsetUid, "offset");
    }
    return operands;
}

/**
 * @brief Fills what every engine's plan takes from the graph the same way: the tensor uids, the
 *        window, the dropout threshold and scale, the runtime flags, the varlen bits and the
 *        scalar operands.
 *
 * @p Plan carries the fields of those names and its matched `problem`, which is set before the
 * call. @p tile is the kernel's head-dim tile: a head dim below it sets RT_FLAG_PADDED.
 */
template <typename Plan>
void bindGraphOperands(Plan& plan,
                       const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       int64_t tile)
{
    const auto& problem = plan.problem;
    const auto& attributes
        = context.graph.getNodeWrapper(0)
              .attributesAs<hipdnn_flatbuffers_sdk::data_objects::SdpaBackwardAttributes>();
    plan.uids = Uids{attributes.q_tensor_uid(),
                     attributes.k_tensor_uid(),
                     attributes.v_tensor_uid(),
                     attributes.o_tensor_uid(),
                     attributes.do_tensor_uid(),
                     attributes.dq_tensor_uid(),
                     attributes.dk_tensor_uid(),
                     attributes.dv_tensor_uid(),
                     attributes.stats_tensor_uid()};

    plan.window = windowFor(problem);
    if(problem.dropoutP > 0.0F)
    {
        plan.dropoutThreshold = flydsl_sdpa::dropoutThreshold(problem.dropoutP);
        plan.dropoutScale = flydsl_sdpa::dropoutScale(problem.dropoutP);
    }
    const bool paddedHead = problem.headDimQk != tile || problem.headDimV != tile;
    plan.runtimeFlags
        = (paddedHead ? RT_FLAG_PADDED : 0) | (problem.dropoutP > 0.0F ? RT_FLAG_DROPOUT : 0);
    plan.varlenBits = varlenBitsOf(problem);

    plan.scalars = scalarOperandsOf(context.graph.getTensorMap(), problem);
}

inline float resolveScale(const bwd::SdpaBwdProblem& problem,
                          const ScalarOperands& operands,
                          const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                          uint32_t numDeviceBuffers)
{
    if(!operands.scale.has_value())
    {
        return problem.scaleValue;
    }
    const double value = hipdnn_plugin_sdk::toDouble(
        hipdnn_plugin_sdk::resolveScalarOperand(*operands.scale, deviceBuffers, numDeviceBuffers));
    // A pass-by-value scalar read from host memory: a constant scale was checked at match
    // time, this one arrives now.
    if(!std::isfinite(value) || std::fabs(value) > bwd::detail::MAX_SOFTMAX_SCALE)
    {
        failBadParam("the runtime softmax scale must be finite and at most 2^20 in magnitude");
    }
    return static_cast<float>(value);
}

/// How a pass-by-value seed and offset reach the workspace: the four 32-bit words of
/// `[seed u64 | offset u64]` written as values (either width, widened to 64 bits).
struct SeedUpload
{
    std::array<uint32_t, 4> words{};
};

/// The Philox seed and offset pointers a kernel reads, and the upload that fills the
/// workspace words when they live there.
struct SeedBinding
{
    void* seed = nullptr;
    void* offset = nullptr;
    std::optional<SeedUpload> upload;
};

/// A pass-by-value integer as the low and high words of its 64-bit extension. A 32-bit value
/// is sign-extended, as the forward SDPA engine does: a negative int32 seed has an all-ones
/// high word.
inline std::array<uint32_t, 2> wordsOf(const hipdnn_plugin_sdk::ScalarValue& value,
                                       const char* what)
{
    if(const auto* narrow = std::get_if<int32_t>(&value))
    {
        const auto bits = static_cast<uint64_t>(static_cast<int64_t>(*narrow));
        return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
    }
    if(const auto* wide = std::get_if<int64_t>(&value))
    {
        const auto bits = static_cast<uint64_t>(*wide);
        return {static_cast<uint32_t>(bits), static_cast<uint32_t>(bits >> 32)};
    }
    failBadParam(std::string("the ") + what + " is not an integer");
}

/// The seed and offset a launch passes. A graph without dropout passes @p unused for both: a
/// tensor the graph does not use is passed as O, as the harness does. A device pair is read in
/// place; a pass-by-value pair is in the workspace's seed region, with the upload that writes
/// it.
inline SeedBinding bindSeed(const bwd::SdpaBwdProblem& problem,
                            const WorkspaceLayout& layout,
                            const ScalarOperands& operands,
                            const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                            uint32_t numDeviceBuffers,
                            void* workspace,
                            void* unused)
{
    const auto buffer = [&](int64_t uid) {
        return hipdnn_plugin_sdk::findDeviceBuffer(uid, deviceBuffers, numDeviceBuffers).ptr;
    };
    SeedBinding binding{unused, unused, std::nullopt};
    if(problem.seedForm == bwd::SeedForm::NONE)
    {
        return binding;
    }
    if(!layout.uploadsSeed)
    {
        binding.seed = buffer(problem.seedUid);
        binding.offset = buffer(problem.offsetUid);
        return binding;
    }
    binding.seed = inWorkspace(workspace, layout.seed.offset);
    binding.offset = inWorkspace(workspace, layout.seed.offset + SEED_WORD_BYTES);
    const auto seed = wordsOf(
        hipdnn_plugin_sdk::resolveScalarOperand(*operands.seed, deviceBuffers, numDeviceBuffers),
        "seed");
    const auto offset = wordsOf(
        hipdnn_plugin_sdk::resolveScalarOperand(*operands.offset, deviceBuffers, numDeviceBuffers),
        "offset");
    SeedUpload upload;
    upload.words = {seed[0], seed[1], offset[0], offset[1]};
    binding.upload = upload;
    return binding;
}

/// Writes the seed and offset words of a pass-by-value form into the seed region at the front
/// of @p workspace, ahead of the launch on @p stream: four word writes.
inline void uploadSeed(const SeedUpload& upload, void* workspace, hipStream_t stream)
{
    auto* base = static_cast<char*>(workspace);
    for(size_t word = 0; word < upload.words.size(); ++word)
    {
        requireHip(hipMemsetD32Async(static_cast<hipDeviceptr_t>(base + word * sizeof(uint32_t)),
                                     static_cast<int>(upload.words[word]),
                                     1,
                                     stream),
                   "hipMemsetD32Async of the seed words");
    }
}

// ---------------------------------------------------------------------------
// The dBias reduce launch
// ---------------------------------------------------------------------------

/// The reduce kernel's grid.
struct ReduceGrid
{
    unsigned int x = 0;
    unsigned int y = 0;
};

/// The grid of the reduce over the batch and head axes @p problem's dBias is broadcast over,
/// for a reduce kernel whose threads each own @p vector consecutive columns.
inline ReduceGrid reduceGridOf(const bwd::SdpaBwdProblem& problem, int64_t vector = REDUCE_VECTOR)
{
    const int64_t vectors = (problem.seqLenKv + vector - 1) / vector;
    const auto items = bwd::detail::checkedProduct({problem.seqLenQ, vectors});
    const auto planes
        = bwd::detail::checkedProduct({problem.biasBroadcastBatch ? int64_t{1} : problem.batch,
                                       problem.biasBroadcastHead ? int64_t{1} : problem.numHeads});
    if(!items || !planes)
    {
        failBadParam("the dBias reduce launch overflows int64");
    }
    const int64_t gridX = (*items + REDUCE_THREADS - 1) / REDUCE_THREADS;
    if(gridX > INT32_LIMIT || *planes > GRID_YZ_LIMIT)
    {
        failBadParam("the dBias reduce launch grid exceeds its limits");
    }
    return {static_cast<unsigned int>(gridX), static_cast<unsigned int>(*planes)};
}

/// The dBias reduce kernel's arguments (`_PARAMS["dbias_reduce"]`): the full-shape workspace
/// and the user's dBias, the problem's extents, which axes are summed, and the strides of
/// each tensor: 14 arguments, 88 bytes.
struct ReduceKernargs
{
    void* workspace = nullptr;
    void* dBias = nullptr;
    int32_t numBatch = 0;
    int32_t numHead = 0;
    int32_t seqLenQ = 0;
    int32_t seqLenK = 0;
    int32_t reduceBatch = 0;
    int32_t reduceHead = 0;
    bwd::Strides strideWorkspace;
    bwd::Strides strideOut;

    auto tie() const
    {
        return std::tie(workspace,
                        dBias,
                        numBatch,
                        numHead,
                        seqLenQ,
                        seqLenK,
                        reduceBatch,
                        reduceHead,
                        strideWorkspace.batch,
                        strideWorkspace.head,
                        strideWorkspace.seq,
                        strideOut.batch,
                        strideOut.head,
                        strideOut.seq);
    }
};

/// The reduce's arguments over the layout's dBias region at @p workspaceDbias and the user's
/// dBias at @p userDbias.
inline ReduceKernargs reduceKernargsOf(const bwd::SdpaBwdProblem& problem,
                                       const WorkspaceLayout& layout,
                                       void* workspaceDbias,
                                       void* userDbias)
{
    ReduceKernargs reduce;
    reduce.workspace = workspaceDbias;
    reduce.dBias = userDbias;
    reduce.numBatch = toInt32(problem.batch, "batch");
    reduce.numHead = toInt32(problem.numHeads, "num_head_q");
    reduce.seqLenQ = toInt32(problem.seqLenQ, "seqlen_q");
    reduce.seqLenK = toInt32(problem.seqLenKv, "seqlen_k");
    reduce.reduceBatch = problem.biasBroadcastBatch ? 1 : 0;
    reduce.reduceHead = problem.biasBroadcastHead ? 1 : 0;
    reduce.strideWorkspace = layout.dbiasStrides;
    reduce.strideOut = problem.dbiasStrides;
    return reduce;
}

// ---------------------------------------------------------------------------
// Graph predicates
// ---------------------------------------------------------------------------

/// Whether any node of the graph is an SDPA-backward node. Only such a graph hears why it was
/// declined: graph_match sees every graph the catalog misses on, and a line for each
/// convolution or normalization graph would bury the attention ones.
inline bool hasSdpaBackwardNode(const hipdnn_plugin_sdk::ingestor::MatchContext& context)
{
    for(const auto& node : context.graph.nodeWrappers())
    {
        if(node->attributesType()
           == hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaBackwardAttributes)
        {
            return true;
        }
    }
    return false;
}

/// The bias's LDS staging rule (the kernel's bias chunk contract), at the bias's own element
/// width: every non-unit stride a multiple of one 16-byte vector of bias elements (the matcher
/// records 0 on a broadcast or unit axis), a 16-byte aligned base, and a last row whose final
/// vector lies inside the allocation: a row stride of at least the key length rounded up to a
/// vector, or, for a single or row-broadcast row (recorded stride 0), which has no padding
/// behind it, a key length that is itself a whole number of vectors. A 16-bit bias that fails
/// it is served only by a dq row that loads the bias directly.
inline bool stagesBias(const bwd::SdpaBwdProblem& problem)
{
    constexpr int64_t ALIGNMENT_BYTES = 16;
    const int64_t vector = ALIGNMENT_BYTES / static_cast<int64_t>(problem.biasElementBytes);
    const bool lastRowFits
        = problem.biasStrides.seq == 0
              ? problem.seqLenKv % vector == 0
              : problem.biasStrides.seq >= bwd::detail::ceilToVector(problem.seqLenKv, vector);
    return problem.biasStrides.batch % vector == 0 && problem.biasStrides.head % vector == 0
           && problem.biasStrides.seq % vector == 0 && lastRowFits
           && problem.biasAlignmentBytes % ALIGNMENT_BYTES == 0;
}

/// Whether every span a gqa_pack dK/dV object binds, from head 0 of the group over its G
/// heads, fits the 32-bit buffer descriptor: Q, dO and O, the statistics (through their actual head and token strides) and, when present,
/// the bias; and whether the kernel's pack_split multiply-high is exact for every packed row
/// it can address: (Sq * G + 2 * block_q) * Sq < 2^31, the rows Sq * G plus the pad to an
/// even tile count, at the largest query tile @p maxBlockQ of the packed objects. A graph
/// that is not grouped has no packed span. Packing is dense only: the packed row index and
/// the Philox counter assume every sequence is max_seqlen_q long, so padded varlen is never
/// packed.
inline bool packedSpansFit(const bwd::SdpaBwdProblem& problem, int64_t maxBlockQ)
{
    if(problem.numHeads <= problem.numKvHeads || problem.paddingMask)
    {
        return false;
    }
    const int64_t group = problem.numHeads / problem.numKvHeads;
    const auto packedRows = bwd::detail::checkedProduct({problem.seqLenQ, group});
    if(!packedRows || *packedRows > INT32_LIMIT)
    {
        return false;
    }
    const auto splitBound
        = bwd::detail::checkedProduct({*packedRows + 2 * maxBlockQ, problem.seqLenQ});
    if(!splitBound || *splitBound >= (int64_t{1} << 31))
    {
        return false;
    }
    const int64_t elementBytes = problem.elementBytes;
    const int64_t vector = bwd::detail::REQUIRED_ALIGNMENT_BYTES / elementBytes;
    const auto spanFits
        = [&](int64_t rows, int64_t seqStride, int64_t width, int64_t headStride, int64_t bytes) {
              const auto slab = bwd::detail::slabElements(rows, seqStride, width);
              const auto heads = bwd::detail::checkedProduct({group - 1, headStride});
              if(!slab || !heads || *heads > INT32_LIMIT)
              {
                  return false;
              }
              const auto bytesOfSpan = bwd::detail::checkedProduct({*slab + *heads, bytes});
              return bytesOfSpan.has_value() && *bytesOfSpan <= INT32_LIMIT;
          };
    const int64_t rowsQ = problem.seqLenQ - 1;
    if(!spanFits(rowsQ,
                 problem.q.seq,
                 bwd::detail::ceilToVector(problem.headDimQk, vector),
                 problem.q.head,
                 elementBytes)
       || !spanFits(rowsQ,
                    problem.dO.seq,
                    bwd::detail::ceilToVector(problem.headDimV, vector),
                    problem.dO.head,
                    elementBytes)
       || !spanFits(rowsQ,
                    problem.o.seq,
                    bwd::detail::ceilToVector(problem.headDimV, vector),
                    problem.o.head,
                    elementBytes))
    {
        return false;
    }
    // The statistics, through their actual strides: the group's heads at problem.stats.head
    // apart, then the Sq rows of the last head at problem.stats.seq (both 0 on a unit axis).
    const auto statsHeads = bwd::detail::checkedProduct({group - 1, problem.stats.head});
    const auto statsRows = bwd::detail::checkedProduct({problem.seqLenQ, problem.stats.seq});
    int64_t statsElements = 0;
    if(!statsHeads || !statsRows || __builtin_add_overflow(*statsHeads, *statsRows, &statsElements))
    {
        return false;
    }
    const auto statsBytes = bwd::detail::checkedProduct({statsElements, ROW_STATISTIC_BYTES});
    if(!statsBytes || *statsBytes > INT32_LIMIT)
    {
        return false;
    }
    if(problem.biasUid == bwd::NO_UID)
    {
        return true;
    }
    return spanFits(rowsQ,
                    problem.biasStrides.seq,
                    problem.seqLenKv,
                    problem.biasStrides.head,
                    problem.biasElementBytes);
}

// ---------------------------------------------------------------------------
// Kernel argument lists
// ---------------------------------------------------------------------------

/// The natural offset of the next argument of type @p T after @p offset.
template <typename T>
size_t alignedOffset(size_t offset)
{
    return (offset + alignof(T) - 1) / alignof(T) * alignof(T);
}

template <typename T>
hipdnn_plugin_sdk::ingestor::KernelArgument describeArgument(size_t& offset)
{
    const size_t at = alignedOffset<T>(offset);
    offset = at + sizeof(T);
    return hipdnn_plugin_sdk::ingestor::KernelArgument{std::is_pointer_v<T> ? "global_buffer"
                                                                            : "by_value",
                                                       static_cast<uint32_t>(sizeof(T)),
                                                       static_cast<uint32_t>(at),
                                                       ""};
}

template <typename Tuple, size_t... Index>
std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>
    signatureFromTuple(std::index_sequence<Index...> /*indices*/)
{
    size_t offset = 0;
    // A braced list is evaluated left to right, so each offset follows the one before.
    return std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>{describeArgument<
        std::remove_cv_t<std::remove_reference_t<std::tuple_element_t<Index, Tuple>>>>(offset)...};
}

/// The signature of a kernel argument struct, from the types of its tie().
template <typename Kernargs>
std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument> signatureOf()
{
    using Tuple = decltype(std::declval<const Kernargs&>().tie());
    return signatureFromTuple<Tuple>(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
}

inline const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& reduceSignature()
{
    static const auto s_signature = signatureOf<ReduceKernargs>();
    return s_signature;
}

// ---------------------------------------------------------------------------
// The tail-zero launch
// ---------------------------------------------------------------------------

/// The tail-zero kernel's threads per block. One thread owns one vector (16 bytes) of one
/// tail row of one (batch, head) plane of dQ, dK or dV
/// (`fmha_bwd_tailzero_gfx950.py`, `fmha_bwd_tailzero_gfx11.py`).
constexpr unsigned int TAILZERO_THREADS = 256;

/// `mask` bits: the outputs the launch zeroes, those the graph holds dense.
constexpr int32_t TAILZERO_DQ = 1;
constexpr int32_t TAILZERO_DK = 2;
constexpr int32_t TAILZERO_DV = 4;

/// The outputs whose rows past `seq_len` the tail-zero launch zeroes: with padding_mask, each
/// output that is dense. A ragged output's storage past its length belongs to the gap or the
/// next sequence and is never written. 0 means no launch.
inline int32_t tailzeroMaskOf(const bwd::SdpaBwdProblem& problem)
{
    if(!problem.paddingMask)
    {
        return 0;
    }
    return (problem.ragged.dQ.ragged() ? 0 : TAILZERO_DQ)
           | (problem.ragged.dK.ragged() ? 0 : TAILZERO_DK)
           | (problem.ragged.dV.ragged() ? 0 : TAILZERO_DV);
}

/// The tail-zero kernel's arguments (`PARAMS` in `flydsl/generators/gen_sdpa_bwd_tailzero.py`):
/// 22 arguments. With @p HV, the V-head variant's (`PARAMS_HV`): 23, `Hv` (`num_head_v`) right
/// after `Hkv` (`num_head_k`), for a graph whose V heads differ from K's.
template <bool HV>
struct TailzeroKernargsT
{
    void* dQ = nullptr;
    void* dK = nullptr;
    void* dV = nullptr;
    void* seqLenQ = nullptr;
    void* seqLenKv = nullptr;
    int32_t numBatch = 0;
    int32_t numHeadQ = 0;
    int32_t numHeadK = 0;
    /// Read only when @p HV.
    int32_t numHeadV = 0;
    int32_t maxSeqlenQ = 0;
    int32_t maxSeqlenK = 0;
    int32_t hdimQk = 0;
    int32_t hdimV = 0;
    bwd::Strides strideDQ;
    bwd::Strides strideDK;
    bwd::Strides strideDV;
    int32_t mask = 0;

    auto tie() const
    {
        const auto leading = std::tie(dQ, dK, dV, seqLenQ, seqLenKv, numBatch, numHeadQ, numHeadK);
        const auto trailing = std::tie(maxSeqlenQ,
                                       maxSeqlenK,
                                       hdimQk,
                                       hdimV,
                                       strideDQ.batch,
                                       strideDQ.head,
                                       strideDQ.seq,
                                       strideDK.batch,
                                       strideDK.head,
                                       strideDK.seq,
                                       strideDV.batch,
                                       strideDV.head,
                                       strideDV.seq,
                                       mask);
        if constexpr(HV)
        {
            return std::tuple_cat(leading, std::tie(numHeadV), trailing);
        }
        else
        {
            return std::tuple_cat(leading, trailing);
        }
    }
};

using TailzeroKernargs = TailzeroKernargsT<false>;
using TailzeroHvKernargs = TailzeroKernargsT<true>;

inline const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& tailzeroSignature()
{
    static const auto s_signature = signatureOf<TailzeroKernargs>();
    return s_signature;
}

inline const std::vector<hipdnn_plugin_sdk::ingestor::KernelArgument>& tailzeroHvSignature()
{
    static const auto s_signature = signatureOf<TailzeroHvKernargs>();
    return s_signature;
}

/// The tail-zero launch's grid: x covers the largest tail-row extent in vectors, y the
/// planes (batch x (Hq + Hk + Hv)), z is 1. nullopt past a grid limit.
struct TailzeroGrid
{
    unsigned int x = 0;
    unsigned int y = 0;
};

inline std::optional<TailzeroGrid> tailzeroGridOf(const bwd::SdpaBwdProblem& problem)
{
    const int64_t vector
        = bwd::detail::REQUIRED_ALIGNMENT_BYTES / static_cast<int64_t>(problem.elementBytes);
    const int64_t vectorsQk = (problem.headDimQk + vector - 1) / vector;
    const int64_t vectorsV = (problem.headDimV + vector - 1) / vector;
    const auto dq = bwd::detail::checkedProduct({problem.seqLenQ, vectorsQk});
    const auto dk = bwd::detail::checkedProduct({problem.seqLenKv, vectorsQk});
    const auto dv = bwd::detail::checkedProduct({problem.seqLenKv, vectorsV});
    const int64_t heads = problem.numKvHeads + problem.numVHeads;
    const auto planes = bwd::detail::checkedProduct({problem.batch, problem.numHeads + heads});
    if(!dq || !dk || !dv || !planes)
    {
        return std::nullopt;
    }
    const int64_t items = std::max({*dq, *dk, *dv});
    const int64_t gridX = (items + TAILZERO_THREADS - 1) / TAILZERO_THREADS;
    if(gridX > INT32_LIMIT || *planes > GRID_YZ_LIMIT)
    {
        return std::nullopt;
    }
    return TailzeroGrid{static_cast<unsigned int>(gridX), static_cast<unsigned int>(*planes)};
}

/// The tail-zero arguments over the outputs at @p dQ, @p dK and @p dV and the lengths at
/// @p seqLenQ and @p seqLenKv. The strides are the graph's own, so a view shared between
/// outputs is addressed as the graph declares it. @p HV selects the V-head variant's list.
template <bool HV = false>
inline TailzeroKernargsT<HV> tailzeroKernargsOf(
    const bwd::SdpaBwdProblem& problem, void* dQ, void* dK, void* dV, void* seqLenQ, void* seqLenKv)
{
    TailzeroKernargsT<HV> tail;
    tail.dQ = dQ;
    tail.dK = dK;
    tail.dV = dV;
    tail.seqLenQ = seqLenQ;
    tail.seqLenKv = seqLenKv;
    tail.numBatch = toInt32(problem.batch, "batch");
    tail.numHeadQ = toInt32(problem.numHeads, "num_head_q");
    tail.numHeadK = toInt32(problem.numKvHeads, "num_head_k");
    if constexpr(HV)
    {
        tail.numHeadV = toInt32(problem.numVHeads, "num_head_v");
    }
    tail.maxSeqlenQ = toInt32(problem.seqLenQ, "seqlen_q");
    tail.maxSeqlenK = toInt32(problem.seqLenKv, "seqlen_k");
    tail.hdimQk = toInt32(problem.headDimQk, "hdim_qk");
    tail.hdimV = toInt32(problem.headDimV, "hdim_vo");
    tail.strideDQ = problem.dQ;
    tail.strideDK = problem.dK;
    tail.strideDV = problem.dV;
    tail.mask = tailzeroMaskOf(problem);
    return tail;
}

/// The bytes of a kernel argument struct, each argument at its natural offset.
template <typename Kernargs>
std::vector<unsigned char> kernargImage(const Kernargs& kernargs)
{
    std::vector<unsigned char> image;
    std::apply(
        [&image](const auto&... argument) {
            const auto append = [&image](const auto& value) {
                using T = std::remove_cv_t<std::remove_reference_t<decltype(value)>>;
                const size_t at = alignedOffset<T>(image.size());
                image.resize(at + sizeof(T));
                std::memcpy(image.data() + at, static_cast<const void*>(&value), sizeof(T));
            };
            (append(argument), ...);
        },
        kernargs.tie());
    return image;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::flydsl_sdpa_bwd_native

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR && HIPDNN_ENGINE_FLYDSL
