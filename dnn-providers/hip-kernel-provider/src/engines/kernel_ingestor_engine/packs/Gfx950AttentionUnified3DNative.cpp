// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/FlatbufferTypeHelpers.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/NativeHooks.hpp>
#include <hipdnn_plugin_sdk/ingestor/SymbolScope.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "compilation/KpackModuleCache.hpp"
#include "core/Handle.hpp"
#include "engines/asm_sdpa_engine/plans/SdpaPlanUtils.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/IngestorPacks.hpp"
#include "engines/kernel_ingestor_engine/LaunchKernelDefinition.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified3DGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGraphMatch.hpp"

/**
 * @file Gfx950AttentionUnified3DNative.cpp
 * @brief The hipkernel:Gfx950AttentionUnified3D engine's native half: matching, scoring and
 *        dispatch.
 *
 * The kernels are `rocke/library/kernels/gfx950/attention_tiled_3d.py`'s split-KV pair,
 * packaged (kind: rocke -> kpack) for gfx950 only as one code object built by
 * `build_unified_attention_3d_tiled_module`: the segment kernel and the reduce kernel that
 * combines its partial results. Each descriptor carries them as two launches of a
 * multi-launch kernel (UKD `sources`), named `segment` and `reduce`; this handler declares
 * exactly those (launchNames), loads each through launchKernelDefinition and launches the
 * segment kernel, then always the reduce. Every shipped kernel is built for DECODE (one
 * query token per sequence, the decode grid); prefill is Unified2D's. Like Unified2D, a
 * graph's contiguous BSHD K/V is served as a paged cache with an identity block table,
 * staged in the workspace ahead of the segment scratch (see
 * Gfx950AttentionUnified3DGeometry.hpp).
 *
 * Three registry candidates share the pack: head_size 64 and 128, both data types, at 16,
 * 32 and 64 segments with a one-page KV tile. The matcher also accepts the bf16 head_size
 * 256 decode geometry (rocKE's d256_decode cohort), which no shipped catalog carries yet.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

using namespace hipdnn_plugin_sdk::ingestor;
namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

namespace
{

// The contract with the installed descriptor files, which restate these same strings.
constexpr std::string_view GRAPH_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_3d.graph_match";
constexpr std::string_view KERNEL_MATCHER_SYMBOL
    = "hipkernel.gfx950_attention_unified_3d.kernel_match";
constexpr std::string_view SCORE_SYMBOL = "hipkernel.gfx950_attention_unified_3d.score";
constexpr std::string_view DISPATCH_SYMBOL = "hipkernel.gfx950_attention_unified_3d.dispatch";

// The launch names every descriptor's `sources` carries (the module's primary kernel and its
// `reduce` companion), which the state manager requires to equal launchNames().
constexpr const char* SEGMENT_LAUNCH = "segment";
constexpr const char* REDUCE_LAUNCH = "reduce";

// KMD fields. Fields a shipped variant holds at one value across the whole catalog
// (waves_per_eu, the tile, kv_storage_dtype, the sink/alibi/qq-bias/softcap switches) are
// absent: a variant that moved one of them would have to add it here first, or the
// candidates collide on the catalog key and the loader drops one.
constexpr std::string_view DTYPE_FIELD = "dtype";
constexpr std::string_view HEAD_SIZE_FIELD = "head_size";
constexpr std::string_view NUM_QUERY_HEADS_FIELD = "num_query_heads";
constexpr std::string_view NUM_KV_HEADS_FIELD = "num_kv_heads";
constexpr std::string_view BLOCK_SIZE_FIELD = "block_size";
constexpr std::string_view MASK_TYPE_FIELD = "mask_type";
constexpr std::string_view SLIDING_WINDOW_FIELD = "sliding_window";
constexpr std::string_view NUM_SEGMENTS_FIELD = "num_segments";
constexpr std::string_view USE_DECODE_GRID_FIELD = "use_decode_grid";
/// 1 for an LSE-on kernel, whose reduce also writes the stats output.
constexpr std::string_view EMIT_LSE_FIELD = "emit_lse";

/// head_size 256 is rocKE's d256_decode cohort: bf16 only.
constexpr int64_t D256_HEAD_SIZE = 256;

constexpr std::string_view Q_TOKEN = "gfx950_attention_unified_3d.q.uid";
constexpr std::string_view K_TOKEN = "gfx950_attention_unified_3d.k.uid";
constexpr std::string_view V_TOKEN = "gfx950_attention_unified_3d.v.uid";
constexpr std::string_view O_TOKEN = "gfx950_attention_unified_3d.o.uid";
/// The mask kind graphMatches derived (a MaskType value), so kernelMatches does not re-derive it.
constexpr std::string_view MASK_TOKEN = "gfx950_attention_unified_3d.mask_type";
/// The sliding window the candidate must have been built with. Always 0: maskTypeFor
/// declines every left bound, so no graph that reaches the bind asks for a window.
constexpr std::string_view SLIDING_WINDOW_TOKEN = "gfx950_attention_unified_3d.sliding_window";
/// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
constexpr std::string_view SCALE_BITS_TOKEN = "gfx950_attention_unified_3d.scale_bits";
/// The stats output's uid, or NO_STATS for a graph that asks for none.
constexpr std::string_view STATS_TOKEN = "gfx950_attention_unified_3d.stats.uid";

// ---------------------------------------------------------------------------
// Matching helpers
// ---------------------------------------------------------------------------

using unified_attention::findTensor;
using unified_attention::hasBshdStrides;
using unified_attention::integerMetadata;
using unified_attention::isWellFormedOperand;
using unified_attention::MaskType;
using unified_attention::problemFor;
using unified_attention::sdpaNode;
using unified_attention::stringMetadata;
using unified_attention::supportedDataTypeName;
using unified_attention::UnifiedAttentionBinding;
using unified_attention::UnifiedAttentionProblem;

// ---------------------------------------------------------------------------
// Decline logging
//
// hipDNN reports a declined graph to its caller only as "No engine configurations available
// for the graph", so each decline logs its cause at INFO. graph_match runs on every catalog
// miss, so every piece of that work sits inside HIPDNN_PLUGIN_LOG_INFO's argument: the macro
// tests the level first, and with INFO off nothing below is called.
// ---------------------------------------------------------------------------

/// The head of this engine's decline lines (unified_attention::DeclinedBy).
struct Declined : unified_attention::DeclinedBy
{
    explicit Declined(std::string_view cause)
        : DeclinedBy{GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME, cause}
    {
    }
};

/// Why head_size 256 cannot be served for this graph, or nullopt when it can. rocKE's
/// `_d256_decode_cohort` is bf16 only. Shared by the graph matcher, which logs the reason,
/// and the kernel matcher, which does not.
std::optional<std::string_view> d256Decline(data_objects::DataType dataType)
{
    if(dataType != data_objects::DataType::BFLOAT16)
    {
        return "head size 256 is built for BFLOAT16 only (the d256_decode cohort)";
    }
    return std::nullopt;
}

/**
 * @brief The catalog's mask key for a graph's mask kind with one query row.
 *
 * With a single query token, full attention and bottom-right causal admit the same keys
 * (every key up to the last), and rocKE's `normalize_band` builds NO_MASK decode as the
 * bottom-right causal kernel. The catalog therefore carries no NO_MASK kernel; a NO_MASK
 * graph is served by the BOTTOM_RIGHT_CAUSAL one. Top-left causal is distinct (one row sees
 * only key 0).
 */
MaskType decodeMaskKey(MaskType mask)
{
    return mask == MaskType::NO_MASK ? MaskType::BOTTOM_RIGHT_CAUSAL : mask;
}

constexpr unified_attention::BoundTokenNames TOKEN_NAMES{Q_TOKEN,
                                                         K_TOKEN,
                                                         V_TOKEN,
                                                         O_TOKEN,
                                                         MASK_TOKEN,
                                                         SLIDING_WINDOW_TOKEN,
                                                         SCALE_BITS_TOKEN,
                                                         STATS_TOKEN,
                                                         "gfx950 attention_unified_3d"};

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * The shared unified-attention match (UnifiedAttentionGraphMatch.hpp) around this engine's
 * rules: GQA groups dividing BLOCK_M, head_size 64, 128 or bf16 256, single-token decode,
 * and NO_MASK served by the bottom-right causal kernel. Each decline of a graph with an
 * SDPA-forward node logs its cause at INFO (see Declined); any other graph declines silently.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<BoundTokens> gfx950AttentionUnified3DGraphMatches(const MatchContext& context)
{
    const auto operands
        = unified_attention::matchOperands(context, GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME);
    if(!operands.has_value())
    {
        return std::nullopt;
    }
    const auto& problem = operands->problem;

    // GQA: the kernel derives its group size by integer division, so a non-divisible pair
    // silently drops heads, and a group over 16 is a shape it is not built for.
    if(!gfx950Unified3DServesGqaGroup(problem.numQueryHeads, problem.numKvHeads))
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"gqa"} << "query heads " << problem.numQueryHeads
                                               << " must be a whole multiple of key/value heads "
                                               << problem.numKvHeads << " whose group size divides "
                                               << GFX950_UNIFIED_3D_BLOCK_M
                                               << " (the segment kernel's BLOCK_M)");
        return std::nullopt;
    }

    // head_size 64 or 128 (the narrow 16x16 geometry) or 256 (rocKE's bf16 D256 decode cohort).
    // The shipped catalog carries no 256 kernel yet.
    if(problem.headSize != 64 && problem.headSize != 128 && problem.headSize != D256_HEAD_SIZE)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"head_size"}
                               << "head size " << problem.headSize
                               << " is not supported; this engine matches 64, 128 and 256");
        return std::nullopt;
    }
    if(problem.headSize == D256_HEAD_SIZE)
    {
        if(const auto why = d256Decline(problem.dataType))
        {
            HIPDNN_PLUGIN_LOG_INFO(Declined{"d256"} << *why);
            return std::nullopt;
        }
    }

    // Decode only: every shipped kernel is built with the decode grid, which launches one
    // CTA per sequence and reads exactly one query row from it. A longer query would have
    // its other rows silently left unwritten. Prefill is Unified2D's.
    if(problem.seqLenQ != 1)
    {
        HIPDNN_PLUGIN_LOG_INFO(Declined{"seqlen_q"}
                               << "seqlen_q " << problem.seqLenQ
                               << " is not 1; this engine serves single-token decode only");
        return std::nullopt;
    }

    const auto matched = unified_attention::matchAttributes(
        context, *operands->attributes, problem, GFX950_ATTENTION_UNIFIED_3D_ENGINE_NAME);
    if(!matched.has_value())
    {
        return std::nullopt;
    }
    // All three mask kinds are served; NO_MASK by the bottom-right causal kernel (see
    // decodeMaskKey), which the bind records as the key kernelMatches compares.
    return unified_attention::bindGraph(*operands->attributes,
                                        decodeMaskKey(matched->mask),
                                        matched->scale,
                                        matched->stats,
                                        TOKEN_NAMES);
}

/// What a candidate bakes, read from its completed metadata and judged buildable.
struct AttentionUnified3DCandidate
{
    int64_t headSize = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t blockSize = 0;
    int64_t maskType = 0;
    int64_t slidingWindow = 0;
    int64_t numSegments = 0;
    /// The reduce also writes the stats output; its kernarg list ends in lse_ptr.
    bool emitLse = false;
};

/**
 * @brief The candidate's baked configuration, or nullopt when it has none this engine can
 *        launch.
 *
 * Validated against isSupportedGfx950Unified3DGeometry, so a record the Python builders
 * would refuse is declined like any other unbuildable one. Declined as well: a record that
 * is not a decode-grid kernel (this engine launches no other grid), one whose mask key is
 * not a causal kind (decode carries no NO_MASK kernel; see decodeMaskKey), and one whose
 * launches are not exactly the segment and reduce kernels -- without the reduce the segment
 * kernel's partial results are never combined and the output is left unwritten. The state
 * manager already drops such a kernel against launchNames(); this repeats it for a caller
 * that builds a definition by hand. An absent field is
 * declined rather than defaulted: a substituted geometry launches a binary with another
 * binary's grid.
 */
std::optional<AttentionUnified3DCandidate> candidateFor(const KernelDefinition& kernel)
{
    const auto headSize = integerMetadata(kernel, HEAD_SIZE_FIELD);
    const auto numQueryHeads = integerMetadata(kernel, NUM_QUERY_HEADS_FIELD);
    const auto numKvHeads = integerMetadata(kernel, NUM_KV_HEADS_FIELD);
    const auto blockSize = integerMetadata(kernel, BLOCK_SIZE_FIELD);
    const auto maskType = integerMetadata(kernel, MASK_TYPE_FIELD);
    const auto slidingWindow = integerMetadata(kernel, SLIDING_WINDOW_FIELD);
    const auto numSegments = integerMetadata(kernel, NUM_SEGMENTS_FIELD);
    const auto useDecodeGrid = integerMetadata(kernel, USE_DECODE_GRID_FIELD);
    const auto emitLse = integerMetadata(kernel, EMIT_LSE_FIELD);
    if(!headSize || !numQueryHeads || !numKvHeads || !blockSize || !maskType || !slidingWindow
       || !numSegments || !useDecodeGrid || !emitLse)
    {
        return std::nullopt;
    }
    if(*emitLse != 0 && *emitLse != 1)
    {
        return std::nullopt;
    }
    if(*useDecodeGrid != 1 || kernel.sources.size() != 2
       || kernel.sources.count(SEGMENT_LAUNCH) != 1 || kernel.sources.count(REDUCE_LAUNCH) != 1)
    {
        return std::nullopt;
    }
    if(*maskType != static_cast<int64_t>(MaskType::TOP_LEFT_CAUSAL)
       && *maskType != static_cast<int64_t>(MaskType::BOTTOM_RIGHT_CAUSAL))
    {
        return std::nullopt;
    }
    if(!isSupportedGfx950Unified3DGeometry(
           *headSize, *blockSize, *numQueryHeads, *numKvHeads, *numSegments))
    {
        return std::nullopt;
    }

    return AttentionUnified3DCandidate{*headSize,
                                       *numQueryHeads,
                                       *numKvHeads,
                                       *blockSize,
                                       *maskType,
                                       *slidingWindow,
                                       *numSegments,
                                       *emitLse != 0};
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * One binary serves any batch and key length for its head/dtype/mask configuration: both
 * reach the kernels at runtime and in the staged tables. What a candidate does bake, and
 * what this checks against the graph:
 *
 *   - dtype, head_size, num_query_heads, num_kv_heads, the mask key and sliding window;
 *   - block_size, which must divide seqlen_kv (the identity cache is whole blocks);
 *   - the decode grid, which serves one query token per sequence only.
 */
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel)
{
    const auto* attributesPtr = sdpaNode(context);
    if(attributesPtr == nullptr)
    {
        return false;
    }
    const auto& attributes = *attributesPtr;

    const auto* q = findTensor(context, attributes.q_tensor_uid());
    const auto* k = findTensor(context, attributes.k_tensor_uid());
    // problemFor reads dims without checking them, so a direct caller (not behind
    // graph_match) must not reach it with a malformed operand.
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        return false;
    }
    const auto problem = problemFor(*q, *k);

    // Before any comparison that could divide by it.
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        return false;
    }

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != dataTypeName)
    {
        return false;
    }

    if(candidate->headSize != problem.headSize || candidate->numQueryHeads != problem.numQueryHeads
       || candidate->numKvHeads != problem.numKvHeads)
    {
        return false;
    }

    // graph_match already declined these for the graph; the kernel matcher is total over any
    // (graph, candidate) pair it is handed.
    if(problem.seqLenQ != 1)
    {
        return false;
    }
    if(candidate->headSize == D256_HEAD_SIZE && d256Decline(problem.dataType))
    {
        return false;
    }

    const auto mask = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, MASK_TOKEN);
    if(!mask.has_value() || candidate->maskType != *mask)
    {
        return false;
    }

    // Sliding window: the KV-loop bound is baked at compile time.
    const auto slidingWindow
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, SLIDING_WINDOW_TOKEN);
    if(!slidingWindow.has_value() || candidate->slidingWindow != *slidingWindow)
    {
        return false;
    }

    // Only an LSE-on kernel serves a graph that asks for stats, and only such a graph: the
    // reduce's LSE store is baked in, with its trailing lse_ptr.
    const auto stats = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, STATS_TOKEN);
    if(!stats.has_value() || candidate->emitLse != (*stats != unified_attention::NO_STATS))
    {
        return false;
    }

    // The identity cache must be whole blocks of THIS candidate's size, and the scratch
    // small enough for the kernels' 32-bit indexing.
    return gfx950Unified3DWorkspaceLayout(problem.batch,
                                          problem.seqLenQ,
                                          problem.seqLenKv,
                                          candidate->blockSize,
                                          problem.numQueryHeads,
                                          problem.headSize,
                                          candidate->numSegments)
        .has_value();
}

/// Scores for the cold ranking. Every legal candidate scores in (0, 1]; a record
/// kernelMatches already declined gets UNSUPPORTED_SCORE.
constexpr double SEGMENT_SPAN = 8192.0; // 2^13, above every legal num_segments
constexpr double UNSUPPORTED_SCORE = 0.0;

/**
 * @brief Ranks candidates that survived kernelMatches. Higher wins.
 *
 * A stable fallback order for cold selection, not a performance model: fewer segments
 * first (less scratch, a smaller reduce). Where a cell ships more than one segment count
 * (the bf16 bottom-right causal D128 cells), cold selection picks 16; the wider variants
 * are reached through autotuning or the num_segments knob. Each distinct segment count
 * gets a distinct score, so the selector's descriptor-id tie-break never decides between
 * two geometries.
 */
double scoreKernel(const MatchContext& /*context*/,
                   const BoundTokens& /*bound*/,
                   const KernelDefinition& kernel)
{
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        return UNSUPPORTED_SCORE;
    }
    return 1.0 - static_cast<double>(candidate->numSegments - 1) / SEGMENT_SPAN;
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// The segment kernel's signature: 12 pointers (the three scratch outputs first), 4 f32
/// scalars, three i32 and the runtime shape kernargs (`_tiled_3d_signatures`: head counts and
/// window).
///
/// NAMES ARE LOAD-BEARING. requireSignatureMatch compares kind and size always, but names
/// only when BOTH sides carry one; kind and size alone cannot tell the twelve pointers
/// apart, so without names an operand permutation passes and the kernel reads the wrong
/// buffers. Keep these spellings identical to build_unified_attention_3d_tiled's `b.param`
/// calls. Offsets are printed in a mismatch diagnostic and never compared.
std::vector<KernelArgument> unified3DSegmentSignature()
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    return {KernelArgument{"global_buffer", PTR, 0, "segm_output_ptr"},
            KernelArgument{"global_buffer", PTR, 8, "segm_max_ptr"},
            KernelArgument{"global_buffer", PTR, 16, "segm_expsum_ptr"},
            KernelArgument{"global_buffer", PTR, 24, "query_ptr"},
            KernelArgument{"global_buffer", PTR, 32, "key_cache_ptr"},
            KernelArgument{"global_buffer", PTR, 40, "value_cache_ptr"},
            KernelArgument{"global_buffer", PTR, 48, "sink_ptr"},
            KernelArgument{"global_buffer", PTR, 56, "block_tables_ptr"},
            KernelArgument{"global_buffer", PTR, 64, "seq_lens_ptr"},
            KernelArgument{"global_buffer", PTR, 72, "alibi_slopes_ptr"},
            KernelArgument{"global_buffer", PTR, 80, "qq_bias_ptr"},
            KernelArgument{"global_buffer", PTR, 88, "query_start_len_ptr"},
            KernelArgument{"by_value", F32, 96, "scale"},
            KernelArgument{"by_value", F32, 100, "k_scale"},
            KernelArgument{"by_value", F32, 104, "v_scale"},
            KernelArgument{"by_value", F32, 108, "softcap"},
            KernelArgument{"by_value", I32, 112, "num_seqs"},
            KernelArgument{"by_value", I32, 116, "block_table_stride"},
            KernelArgument{"by_value", I32, 120, "qq_bias_stride0"},
            KernelArgument{"by_value", I32, 124, "num_query_heads"},
            KernelArgument{"by_value", I32, 128, "num_kv_heads"},
            KernelArgument{"by_value", I32, 132, "sliding_window"}};
}

/// The reduce kernel's signature: the output then the three scratch arrays it folds,
/// seq_lens and the runtime query head count (`_tiled_3d_signatures`). Names load-bearing as
/// above; keep them identical to build_unified_attention_reduce_tiled's `b.param` calls. An
/// LSE-on reduce (@p emitLse) appends lse_ptr.
std::vector<KernelArgument> unified3DReduceSignature(bool emitLse = false)
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));

    std::vector<KernelArgument> arguments{
        KernelArgument{"global_buffer", PTR, 0, "output_ptr"},
        KernelArgument{"global_buffer", PTR, 8, "segm_output_ptr"},
        KernelArgument{"global_buffer", PTR, 16, "segm_max_ptr"},
        KernelArgument{"global_buffer", PTR, 24, "segm_expsum_ptr"},
        KernelArgument{"global_buffer", PTR, 32, "seq_lens_ptr"},
        KernelArgument{"by_value", I32, 40, "num_query_heads"}};
    if(emitLse)
    {
        // The LSE-on reduce (UnifiedAttentionReduceTiledSpec.emit_lse) appends it after the
        // runtime head count, 8-byte aligned (read off the compiled LSE reduce).
        arguments.push_back(KernelArgument{"global_buffer", PTR, 48, "lse_ptr"});
    }
    return arguments;
}

/// Both compiled kernels plus everything launch() needs, owning nothing that points back
/// into the MatchContext or BoundTokens it came from.
class PreparedGfx950AttentionUnified3D : public PreparedDispatch
{
public:
    PreparedGfx950AttentionUnified3D(IngestorKernelCode segment,
                                     IngestorKernelCode reduce,
                                     UnifiedAttentionBinding binding,
                                     UnifiedAttentionProblem problem,
                                     Gfx950Unified3DWorkspaceLayout layout,
                                     std::vector<unsigned char> workspaceImage,
                                     int64_t slidingWindow)
        : _segment(std::move(segment))
        , _reduce(std::move(reduce))
        , _binding(binding)
        , _problem(problem)
        , _layout(layout)
        , _workspaceImage(std::move(workspaceImage))
        , _slidingWindow(slidingWindow)
    {
    }

    compilation::IRunnableKernel& segmentForStream(hipStream_t stream) const
    {
        return _segment.kernelForStream(stream);
    }

    compilation::IRunnableKernel& reduceForStream(hipStream_t stream) const
    {
        return _reduce.kernelForStream(stream);
    }

    const UnifiedAttentionBinding& binding() const
    {
        return _binding;
    }

    const UnifiedAttentionProblem& problem() const
    {
        return _problem;
    }

    const Gfx950Unified3DWorkspaceLayout& layout() const
    {
        return _layout;
    }

    const std::vector<unsigned char>& workspaceImage() const
    {
        return _workspaceImage;
    }

    /// The variant's window, which the kernel reads as a kernarg.
    int64_t slidingWindow() const
    {
        return _slidingWindow;
    }

private:
    IngestorKernelCode _segment;
    IngestorKernelCode _reduce;
    UnifiedAttentionBinding _binding;
    UnifiedAttentionProblem _problem;
    Gfx950Unified3DWorkspaceLayout _layout;
    std::vector<unsigned char> _workspaceImage;
    int64_t _slidingWindow = 0;
};

/// The graph's problem, read from the bound Q and K uids. A null tensor means a bound uid
/// names no tensor: refused by name, since everything after indexes dims.
UnifiedAttentionProblem boundProblem(const MatchContext& context,
                                     const UnifiedAttentionBinding& binding)
{
    const auto* q = findTensor(context, binding.q);
    const auto* k = findTensor(context, binding.k);
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_3d: the bound Q or K tensor is missing or malformed");
    }
    return problemFor(*q, *k);
}

/// The candidate, or a refusal that names the kernel: kernel_match declines a candidate it
/// cannot launch, so this is defence in depth for a caller that reaches the handler without
/// having matched.
AttentionUnified3DCandidate requireCandidate(const KernelDefinition& kernel)
{
    const auto candidate = candidateFor(kernel);
    if(!candidate.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_3d: kernel '" + toString(kernel.kernelId)
                + "' declares no supported decode geometry, or its launches are not exactly '"
                + SEGMENT_LAUNCH + "' and '" + REDUCE_LAUNCH + "'");
    }
    return *candidate;
}

/// The layout for @p candidate over @p problem, or a refusal that says which rule failed.
Gfx950Unified3DWorkspaceLayout requireLayout(const AttentionUnified3DCandidate& candidate,
                                             const UnifiedAttentionProblem& problem,
                                             const KernelDefinition& kernel)
{
    if(problem.seqLenQ != 1)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_3d: kernel '" + toString(kernel.kernelId)
                + "' is a decode-grid kernel and the graph has seqlen_q "
                + std::to_string(problem.seqLenQ));
    }
    const auto layout = gfx950Unified3DWorkspaceLayout(problem.batch,
                                                       problem.seqLenQ,
                                                       problem.seqLenKv,
                                                       candidate.blockSize,
                                                       problem.numQueryHeads,
                                                       problem.headSize,
                                                       candidate.numSegments);
    if(!layout.has_value())
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_3d: kernel '" + toString(kernel.kernelId)
                + "' cannot lay out its workspace for this graph: block_size "
                + std::to_string(candidate.blockSize) + " must divide seqlen_kv "
                + std::to_string(problem.seqLenKv)
                + " and the segment scratch must fit 32-bit indexing");
    }
    return *layout;
}

/**
 * @brief The native dispatch behind this engine's UDD.
 */
class Gfx950AttentionUnified3DDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    explicit Gfx950AttentionUnified3DDispatchHandler(
        const compilation::KpackKernelLoader& kpackLoader)
        : _kpackLoader(kpackLoader)
    {
    }

    /// The segment kernel, then the reduce kernel: one module, two launches.
    std::set<std::string> launchNames() const override
    {
        return {SEGMENT_LAUNCH, REDUCE_LAUNCH};
    }

    /// The identity paged-KV metadata, then the segment scratch; each section 256-byte
    /// aligned.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto binding = unified_attention::readBinding(bound, TOKEN_NAMES);
        const auto problem = boundProblem(context, binding);
        const auto candidate = requireCandidate(kernel);
        return requireLayout(candidate, problem, kernel).totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = unified_attention::readBinding(bound, TOKEN_NAMES);

        // graph_match is the gate for this; a re-check here is defence in depth for a
        // caller that reaches the handler without having matched.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_3d: the output tensor is not dense BSHD; the reduce "
                "kernel bakes that layout and takes no stride arguments");
        }

        const auto candidate = requireCandidate(kernel);
        // kernelMatches pairs them; a caller that skipped it would launch the wrong reduce
        // kernarg list.
        if(candidate.emitLse != (binding.stats != unified_attention::NO_STATS))
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_3d: kernel '" + toString(kernel.kernelId)
                    + (candidate.emitLse ? "' writes stats, but the graph asks for none"
                                         : "' writes no stats, but the graph asks for them"));
        }
        const auto problem = boundProblem(context, binding);
        const auto layout = requireLayout(candidate, problem, kernel);

        // The host-side metadata is built before either code object is loaded, so a graph
        // the tables cannot describe is refused by name without touching an archive.
        auto image = gfx950Unified3DWorkspaceImage(
            problem.batch, problem.seqLenQ, problem.seqLenKv, candidate.blockSize);
        if(!image.has_value() || image->size() != layout.segmOutputOffset)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_3d: kernel '" + toString(kernel.kernelId)
                    + "' cannot stage paged-KV metadata for this graph");
        }

        // Each launch is loaded through the same adapter and checked against its own
        // signature; both resolve out of the one module the descriptor names. The reduce
        // launch's recorded signature is checked here as well, before either load, so a
        // descriptor whose reduce kernel disagrees with this pack is refused without opening
        // an archive -- the adapter would otherwise only reach it after the segment loaded.
        // A launch's definition keeps the kernel's name, so the labels add the launch.
        const auto segmentDefinition = launchKernelDefinition(kernel, SEGMENT_LAUNCH);
        const auto reduceDefinition = launchKernelDefinition(kernel, REDUCE_LAUNCH);
        if(reduceDefinition.source.kind == hipdnn_plugin_sdk::ingestor::KernelSourceKind::KPACK)
        {
            requireSignatureMatch(reduceDefinition.source.signature,
                                  unified3DReduceSignature(candidate.emitLse),
                                  reduceDefinition.source.symbol,
                                  hipdnn_plugin_sdk::ingestor::describeDescriptor(
                                      "kernel",
                                      reduceDefinition.name + " (launch '" + REDUCE_LAUNCH + "')",
                                      reduceDefinition.kernelId));
        }
        auto segment = buildIngestorKernelCode(
            _kpackLoader, context, segmentDefinition, unified3DSegmentSignature());
        auto reduce = buildIngestorKernelCode(
            _kpackLoader, context, reduceDefinition, unified3DReduceSignature(candidate.emitLse));

        const auto geometry = gfx950Unified3DLaunchGeometry(problem.numQueryHeads,
                                                            problem.numKvHeads,
                                                            candidate.numSegments,
                                                            problem.batch,
                                                            toString(kernel.kernelId));

        segment.setBlockSize(geometry.blockX, 1, 1);
        segment.setGridSize(geometry.segmentGridX, geometry.segmentGridY, geometry.segmentGridZ);
        reduce.setBlockSize(geometry.blockX, 1, 1);
        reduce.setGridSize(geometry.reduceGridX, geometry.reduceGridY, 1);

        return std::make_unique<PreparedGfx950AttentionUnified3D>(std::move(segment),
                                                                  std::move(reduce),
                                                                  binding,
                                                                  problem,
                                                                  layout,
                                                                  std::move(*image),
                                                                  candidate.slidingWindow);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedUnified
            = dynamic_cast<const PreparedGfx950AttentionUnified3D&>(prepared);
        const auto& binding = preparedUnified.binding();
        const auto& layout = preparedUnified.layout();
        const auto& image = preparedUnified.workspaceImage();

        if(workspace == nullptr)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                "gfx950 attention_unified_3d: launched without the "
                    + std::to_string(layout.totalBytes)
                    + "-byte workspace workspaceBytes() asked for");
        }

        const auto q
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.q, deviceBuffers, numDeviceBuffers);
        const auto k
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.k, deviceBuffers, numDeviceBuffers);
        const auto v
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.v, deviceBuffers, numDeviceBuffers);
        const auto o
            = hipdnn_plugin_sdk::findDeviceBuffer(binding.o, deviceBuffers, numDeviceBuffers);

        // One copy of the identity metadata, on the handle's stream so both kernels launched
        // after it see it. The scratch after it is written by the segment kernel before the
        // reduce reads it, all in stream order.
        hipStream_t stream = handle.getStream();
        const hipError_t copied
            = hipMemcpyAsync(workspace, image.data(), image.size(), hipMemcpyHostToDevice, stream);
        if(copied != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                std::string("gfx950 attention_unified_3d: staging the paged-KV metadata failed: ")
                    + hipGetErrorString(copied));
        }

        auto* const base = static_cast<unsigned char*>(workspace);
        void* const blockTables = base + layout.identity.blockTablesOffset;
        void* const seqLens = base + layout.identity.seqLensOffset;
        void* const queryStartLen = base + layout.identity.queryStartLenOffset;
        void* const segmOutput = base + layout.segmOutputOffset;
        void* const segmMax = base + layout.segmMaxOffset;
        void* const segmExpsum = base + layout.segmExpsumOffset;

        // Unused pointers are null and the unused scalars neutral: the shipped variants have
        // no sinks, ALiBi, qq-bias, softcap or fp8, so the kernel never reads them.
        void* const unused = nullptr;
        const float neutralScale = 1.0F;
        const float noSoftcap = 0.0F;
        const int32_t noQqBiasStride = 0;

        const auto& p = preparedUnified.problem();
        preparedUnified.segmentForStream(stream).launch(
            stream,
            segmOutput,
            segmMax,
            segmExpsum,
            q.ptr,
            k.ptr,
            v.ptr,
            unused,
            blockTables,
            seqLens,
            unused,
            unused,
            queryStartLen,
            binding.scale,
            neutralScale,
            neutralScale,
            noSoftcap,
            static_cast<int32_t>(p.batch),
            static_cast<int32_t>(layout.identity.blocksPerSeq),
            noQqBiasStride,
            static_cast<int32_t>(p.numQueryHeads),
            static_cast<int32_t>(p.numKvHeads),
            static_cast<int32_t>(preparedUnified.slidingWindow()));
        if(binding.stats != unified_attention::NO_STATS)
        {
            // The LSE-on reduce: the same arguments, then the stats output.
            const auto stats = hipdnn_plugin_sdk::findDeviceBuffer(
                binding.stats, deviceBuffers, numDeviceBuffers);
            preparedUnified.reduceForStream(stream).launch(stream,
                                                           o.ptr,
                                                           segmOutput,
                                                           segmMax,
                                                           segmExpsum,
                                                           seqLens,
                                                           static_cast<int32_t>(p.numQueryHeads),
                                                           stats.ptr);
            return;
        }
        preparedUnified.reduceForStream(stream).launch(stream,
                                                       o.ptr,
                                                       segmOutput,
                                                       segmMax,
                                                       segmExpsum,
                                                       seqLens,
                                                       static_cast<int32_t>(p.numQueryHeads));
    }

private:
    const compilation::KpackKernelLoader& _kpackLoader;
};

} // namespace

compilation::KpackModuleCache& gfx950AttentionUnified3DKpackModuleCache()
{
    static compilation::KpackModuleCache s_moduleCache;
    return s_moduleCache;
}

void resetGfx950AttentionUnified3DModuleCache()
{
    gfx950AttentionUnified3DKpackModuleCache().clear();
}

namespace
{

/// This engine's dispatch handler, process-lifetime.
const Gfx950AttentionUnified3DDispatchHandler& gfx950AttentionUnified3DDispatchHandler()
{
    static const compilation::KpackKernelLoader s_kpackLoader(
        gfx950AttentionUnified3DKpackModuleCache());
    static const Gfx950AttentionUnified3DDispatchHandler s_dispatchHandler(s_kpackLoader);
    return s_dispatchHandler;
}

} // namespace

void registerGfx950AttentionUnified3DSymbols(SymbolScope<Handle>& scope)
{
    scope.add(std::string(GRAPH_MATCHER_SYMBOL), &gfx950AttentionUnified3DGraphMatches);
    scope.add(std::string(KERNEL_MATCHER_SYMBOL), &kernelMatches);
    scope.add(std::string(SCORE_SYMBOL), &scoreKernel);
    scope.add(std::string(DISPATCH_SYMBOL), &gfx950AttentionUnified3DDispatchHandler());
}

} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
