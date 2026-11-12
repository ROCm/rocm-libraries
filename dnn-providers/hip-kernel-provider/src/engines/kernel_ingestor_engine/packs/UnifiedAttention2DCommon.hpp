// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <hip/hip_runtime_api.h>

#include <hipdnn_plugin_sdk/PluginDeviceBuffers.hpp>
#include <hipdnn_plugin_sdk/PluginException.hpp>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "compilation/KpackKernelLoader.hpp"
#include "core/Handle.hpp"
#include "engines/kernel_ingestor_engine/IngestorKernelCode.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGraphMatch.hpp"

/**
 * @file UnifiedAttention2DCommon.hpp
 * @brief What the unified-attention 2D packs share (hipkernel:Gfx950AttentionUnified2D and
 *        hipkernel:Gfx942AttentionUnified2D): the graph match, the kernel match against a
 *        candidate's baked metadata, and the native dispatch, parameterised by one
 *        descriptor per architecture.
 *
 * Both packs serve a dense BSHD SDPA-forward node through rocKE's tiled 2D paged-KV kernel,
 * with the same ABI, the same identity block table in the plan's workspace and the same grid
 * formula (UnifiedAttentionGeometry.hpp), so one copy of that code serves both. What differs
 * per architecture reaches it through Unified2DGraphArch and Unified2DArch: the traits, the
 * engine and its tokens, the decode hand-off rule, which GQA ratios the catalog ships, whether
 * the mask is restricted, how the candidate's record is read (`candidateFor`), which head
 * facts a candidate bakes (`headsMatch`) and the extra checks prepare() makes. Scoring, the
 * KMD field names and the candidate record stay in each pack's own native file, which also
 * owns its module cache and registration.
 *
 * Header-only: the candidate type is the pack's own, so the kernel match and the handler are
 * templates over it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention
{

using hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler;
using hipdnn_plugin_sdk::ingestor::KernelArgument;
using hipdnn_plugin_sdk::ingestor::PreparedDispatch;

/// The KMD `dtype` field both packs' variants carry.
inline constexpr std::string_view DTYPE_FIELD = "dtype";

/// The GQA group (query heads per key/value head) of a head pair, or nullopt when the pair has
/// none: no key/value heads, or a query head count they do not divide.
inline std::optional<int64_t> gqaRatio(int64_t numQueryHeads, int64_t numKvHeads)
{
    if(numKvHeads <= 0 || numQueryHeads % numKvHeads != 0)
    {
        return std::nullopt;
    }
    return numQueryHeads / numKvHeads;
}

/// What the graph match asks of one architecture's engine.
struct Unified2DGraphArch
{
    /// "gfx942" or "gfx950": the head of every error this module throws for the arch.
    std::string_view label;
    /// The engine name its decline lines carry.
    std::string_view engineName;
    const UnifiedAttentionArchTraits* traits = nullptr;
    BoundTokenNames tokens;
    /// Whether decode (seqlen_q 1) of a head pair is the split-KV engine's.
    bool (*decodeHandedOff)(int64_t numQueryHeads, int64_t numKvHeads) = nullptr;
    /// The split-KV engine's name as its decline line spells it.
    std::string_view decodeEngineLabel;
    /// Whether a graph asking for a mask other than the bottom-right causal band is declined
    /// by name: the catalog carries only that kernel. False serves every mask kind.
    bool bottomRightOnly = false;
    /// The GQA groups the catalog ships, or nullptr when no group is judged: a catalog keyed
    /// by head pair rather than by ratio lists pairs a ratio list cannot name.
    const int64_t* shippedGqaRatios = nullptr;
    std::size_t numShippedGqaRatios = 0;
};

/// Whether the catalog ships a kernel for GQA group @p ratio: the one ratio rule. graph_match
/// declines a group this refuses by name, so it cannot empty the catalog at kernel_match.
inline bool servesGqaRatio(const Unified2DGraphArch& arch, int64_t ratio)
{
    if(arch.shippedGqaRatios == nullptr)
    {
        return true;
    }
    for(std::size_t i = 0; i < arch.numShippedGqaRatios; ++i)
    {
        if(arch.shippedGqaRatios[i] == ratio)
        {
            return true;
        }
    }
    return false;
}

/// The shipped groups as a decline line spells them: "1, 2, 4, 8 and 16".
inline std::string describeShippedGqaRatios(const Unified2DGraphArch& arch)
{
    std::string text;
    for(std::size_t i = 0; i < arch.numShippedGqaRatios; ++i)
    {
        if(i > 0)
        {
            text += i + 1 == arch.numShippedGqaRatios ? " and " : ", ";
        }
        text += std::to_string(arch.shippedGqaRatios[i]);
    }
    return text;
}

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * The shared unified-attention match (UnifiedAttentionGraphMatch.hpp) around the 2D engines'
 * rules: a whole GQA group the catalog ships (up to the traits' widest), head_size 64 or 128,
 * q_len <= kv_len, decode handed to the split-KV engine, and the arch's mask restriction. Each
 * decline of a graph with an SDPA-forward node logs its cause at INFO under the engine's name;
 * any other graph declines silently.
 *
 * @warning Returning std::nullopt empties this engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
inline std::optional<BoundTokens> graphMatches(const MatchContext& context,
                                               const Unified2DGraphArch& arch)
{
    const auto operands = matchOperands(context, arch.engineName);
    if(!operands.has_value())
    {
        return std::nullopt;
    }
    const auto& problem = operands->problem;

    // GQA: the kernel derives its group size by integer division, so a non-divisible pair
    // silently drops heads, and a group over the traits' widest is a shape it is not built
    // for.
    const auto ratio = gqaRatio(problem.numQueryHeads, problem.numKvHeads);
    if(!ratio.has_value() || *ratio > arch.traits->maxQueriesPerKv)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            (DeclinedBy{arch.engineName, "gqa"})
            << "query heads " << problem.numQueryHeads << " must be a whole multiple, at most "
            << arch.traits->maxQueriesPerKv << ", of key/value heads " << problem.numKvHeads);
        return std::nullopt;
    }

    // A group the catalog ships no kernel for would pass every check below and then empty the
    // catalog at kernel_match, with no line to say why.
    if(!servesGqaRatio(arch, *ratio))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{arch.engineName, "gqa"})
                               << "a GQA group of " << *ratio
                               << " is not supported; the catalog ships groups of "
                               << describeShippedGqaRatios(arch));
        return std::nullopt;
    }

    // head_size 64 or 128: the catalog is built for those two.
    if(problem.headSize != 64 && problem.headSize != 128)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{arch.engineName, "head_size"})
                               << "head size " << problem.headSize
                               << " is not supported; this catalog is built for 64 and 128");
        return std::nullopt;
    }

    // The kernel's seq_lens contract (vLLM's) is q_len <= kv_len per sequence: its causal
    // limit is seq_len - q_len, which a longer query makes negative.
    if(problem.seqLenQ > problem.seqLenKv)
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{arch.engineName, "shape"})
                               << "seqlen_q " << problem.seqLenQ << " exceeds seqlen_kv "
                               << problem.seqLenKv << "; the kernel needs q_len <= kv_len");
        return std::nullopt;
    }

    // Decode (one query token per sequence) is the split-KV engine's wherever it can be
    // built: there the 2D kernel runs one under-filled Q block per sequence while the
    // split-KV kernel splits the keys across CTAs. No select_path crossover is replayed: the
    // boundary is this static rule.
    if(problem.seqLenQ == 1 && arch.decodeHandedOff(problem.numQueryHeads, problem.numKvHeads))
    {
        HIPDNN_PLUGIN_LOG_INFO((DeclinedBy{arch.engineName, "decode"})
                               << "decode (seqlen_q 1) with a GQA group of " << *ratio
                               << " is served by the split-KV " << arch.decodeEngineLabel
                               << " engine");
        return std::nullopt;
    }

    const auto matched = matchAttributes(
        context, *operands->attributes, problem, arch.engineName, *arch.traits);
    if(!matched.has_value())
    {
        return std::nullopt;
    }

    // The tiled 2D body of some archs computes only the bottom-right causal band, so the
    // catalog carries only that mask. A graph asking for another is declined by name here
    // rather than left to empty the catalog at kernel_match. Top-left at Sq == Skv is the same
    // computation; declining it too is a scope choice, not a kernel limit.
    if(arch.bottomRightOnly && matched->mask != MaskType::BOTTOM_RIGHT_CAUSAL)
    {
        HIPDNN_PLUGIN_LOG_INFO(
            (DeclinedBy{arch.engineName, "mask"})
            << (matched->mask == MaskType::NO_MASK ? "no causal mask" : "a top-left causal mask")
            << " is not supported; the " << arch.label
            << " kernel computes the bottom-right causal band only");
        return std::nullopt;
    }
    return bindGraph(
        *operands->attributes, matched->mask, matched->scale, matched->stats, arch.tokens);
}

/// What the kernel match and the handler ask of one architecture's engine, over its candidate
/// record @p Candidate (which must carry `headSize`, `blockSize`, `maskType`, `slidingWindow`,
/// `numWarps`, `blockMPerWarp` and `emitLse`: whether the kernel also writes the stats output,
/// always false where no LSE-on kernel ships).
template <typename Candidate>
struct Unified2DArch
{
    Unified2DGraphArch graph;
    /// The candidate's baked configuration, or nullopt when it has none this engine can
    /// launch. Total: an absent or malformed field declines, never throws.
    std::optional<Candidate> (*candidateFor)(const KernelDefinition& kernel) = nullptr;
    /// Whether @p candidate's baked head facts fit @p problem: the head pair itself or only
    /// its ratio. Must be total over a non-dividing pair.
    bool (*headsMatch)(const Candidate& candidate, const SdpaShape& problem) = nullptr;
    /// The checks prepare() makes beyond the shared ones, once the candidate and the problem
    /// are known: throws HipdnnPluginException with HIPDNN_PLUGIN_STATUS_BAD_PARAM for a
    /// launch the candidate cannot serve. nullptr for none.
    void (*prepareChecks)(const Candidate& candidate,
                          const SdpaShape& problem,
                          const KernelDefinition& kernel)
        = nullptr;
};

/// The head of every error the dispatch throws for @p label's engine.
inline std::string errorPrefix(std::string_view label)
{
    return std::string(label) + " attention_unified_2d: ";
}

/// Throws HIPDNN_PLUGIN_STATUS_BAD_PARAM with @p text under @p label's engine prefix.
[[noreturn]] inline void refuse(std::string_view label, const std::string& text)
{
    throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                   errorPrefix(label) + text);
}

/**
 * @brief Kernel-scoped applicability: does THIS candidate's baked metadata fit?
 *
 * One binary serves any sequence length and batch of its configuration: batch, seqlen_q and
 * seqlen_kv reach the kernel as runtime arguments and in the staged tables, so the metadata
 * carries no shape and none is compared. What a candidate does bake, and what this checks
 * against the graph:
 *
 *   - dtype, head_size, the arch's head facts (Unified2DArch::headsMatch), the mask kind and
 *     the sliding window;
 *   - the stats output: a candidate that writes it (emit_lse) serves only a graph that asks
 *     for it, and every other candidate only a graph that does not;
 *   - block_size, which must divide seqlen_kv (the identity cache is whole blocks).
 */
template <typename Candidate>
bool kernelMatches(const MatchContext& context,
                   const BoundTokens& bound,
                   const KernelDefinition& kernel,
                   const Unified2DArch<Candidate>& arch)
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
    const auto candidate = arch.candidateFor(kernel);
    if(!candidate.has_value())
    {
        return false;
    }

    const auto dataTypeName = supportedDataTypeName(problem.dataType);
    if(!dataTypeName.has_value() || stringMetadata(kernel, DTYPE_FIELD) != dataTypeName)
    {
        return false;
    }

    if(candidate->headSize != problem.headSize || !arch.headsMatch(*candidate, problem))
    {
        return false;
    }

    const auto mask
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.graph.tokens.maskType);
    if(!mask.has_value() || candidate->maskType != *mask)
    {
        return false;
    }

    // Sliding window: the kernel's mask structure differs with a window, so a windowed
    // variant is a different binary.
    const auto slidingWindow
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.graph.tokens.slidingWindow);
    if(!slidingWindow.has_value() || candidate->slidingWindow != *slidingWindow)
    {
        return false;
    }

    // Only an LSE-on kernel serves a graph that asks for stats, and only such a graph: the
    // LSE store is baked in, with its trailing lse_ptr.
    const auto stats
        = hipdnn_plugin_sdk::ingestor::tryGetBoundInt(bound, arch.graph.tokens.stats);
    if(!stats.has_value() || candidate->emitLse != (*stats != NO_STATS))
    {
        return false;
    }

    // The identity cache must be whole blocks of THIS candidate's size, and small enough for
    // the table the kernel indexes in 32 bits.
    return unifiedAttentionWorkspaceLayout(problem.batch, problem.seqLenKv, candidate->blockSize)
        .has_value();
}

/// The kernel signature for all variants of the 2D engines: 10 pointers, 6 scalars, the two
/// strides (`_attn_signature(include_bt_stride=True, include_qq_bias_stride=True)`) and the
/// runtime shape kernargs (`RUNTIME_KERNARG_FIELDS`: head counts and window).
///
/// NAMES ARE LOAD-BEARING. requireSignatureMatch compares kind and size always, but names
/// only when BOTH sides carry one. Kind and size alone cannot tell the ten pointers apart, nor
/// the f32 scalars from the i32 ones, so without names an operand permutation passes the check
/// and the kernel reads the wrong buffers with no error and no status code. hkp_pack lowers
/// the recorded names out of the rocKE builder's own parameter list, so keep these spellings
/// identical to the Python (kernels/<arch>/attention_tiled_2d.py,
/// build_unified_attention_2d_tiled's `b.param` calls); a divergence here fails every
/// dispatch rather than silently weakening the check.
///
/// Offsets mirror the packed kernarg layout. They are not compared; they exist so the
/// mismatch diagnostic prints the real layout beside the recorded one. An LSE-on kernel
/// (@p emitLse) appends lse_ptr.
inline std::vector<KernelArgument> unified2DKernelSignature(bool emitLse = false)
{
    constexpr auto PTR = static_cast<uint32_t>(sizeof(void*));
    constexpr auto I32 = static_cast<uint32_t>(sizeof(int32_t));
    constexpr auto F32 = static_cast<uint32_t>(sizeof(float));

    std::vector<KernelArgument> arguments{
        KernelArgument{"global_buffer", PTR, 0, "output_ptr"},
        KernelArgument{"global_buffer", PTR, 8, "query_ptr"},
        KernelArgument{"global_buffer", PTR, 16, "key_cache_ptr"},
        KernelArgument{"global_buffer", PTR, 24, "value_cache_ptr"},
        KernelArgument{"global_buffer", PTR, 32, "sink_ptr"},
        KernelArgument{"global_buffer", PTR, 40, "block_tables_ptr"},
        KernelArgument{"global_buffer", PTR, 48, "seq_lens_ptr"},
        KernelArgument{"global_buffer", PTR, 56, "alibi_slopes_ptr"},
        KernelArgument{"global_buffer", PTR, 64, "qq_bias_ptr"},
        KernelArgument{"global_buffer", PTR, 72, "query_start_len_ptr"},
        KernelArgument{"by_value", F32, 80, "scale"},
        KernelArgument{"by_value", F32, 84, "k_scale"},
        KernelArgument{"by_value", F32, 88, "v_scale"},
        KernelArgument{"by_value", F32, 92, "out_scale"},
        KernelArgument{"by_value", F32, 96, "softcap"},
        KernelArgument{"by_value", I32, 100, "num_seqs"},
        KernelArgument{"by_value", I32, 104, "block_table_stride"},
        KernelArgument{"by_value", I32, 108, "qq_bias_stride_0"},
        KernelArgument{"by_value", I32, 112, "num_query_heads"},
        KernelArgument{"by_value", I32, 116, "num_kv_heads"},
        KernelArgument{"by_value", I32, 120, "sliding_window"}};
    if(emitLse)
    {
        // After the runtime kernargs, 8-byte aligned (read off the compiled LSE kernel).
        arguments.push_back(KernelArgument{"global_buffer", PTR, 128, "lse_ptr"});
    }
    return arguments;
}

/// The compiled kernel plus everything launch() needs, owning nothing that points back into
/// the MatchContext or BoundTokens it came from.
class PreparedUnified2D : public PreparedDispatch
{
public:
    PreparedUnified2D(IngestorKernelCode code,
                      UnifiedAttentionBinding binding,
                      SdpaShape problem,
                      UnifiedAttentionWorkspaceLayout layout,
                      std::vector<unsigned char> workspaceImage,
                      int64_t slidingWindow)
        : _code(std::move(code))
        , _binding(binding)
        , _problem(problem)
        , _layout(layout)
        , _workspaceImage(std::move(workspaceImage))
        , _slidingWindow(slidingWindow)
    {
    }

    compilation::IRunnableKernel& kernelForStream(hipStream_t stream) const
    {
        return _code.kernelForStream(stream);
    }

    const UnifiedAttentionBinding& binding() const
    {
        return _binding;
    }

    const SdpaShape& problem() const
    {
        return _problem;
    }

    const UnifiedAttentionWorkspaceLayout& layout() const
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
    IngestorKernelCode _code;
    UnifiedAttentionBinding _binding;
    SdpaShape _problem;
    UnifiedAttentionWorkspaceLayout _layout;
    std::vector<unsigned char> _workspaceImage;
    int64_t _slidingWindow = 0;
};

/// The graph's problem, read from the bound Q and K uids. A null tensor means a bound uid
/// names no tensor: refused by name, since everything after indexes dims.
inline SdpaShape boundProblem(const MatchContext& context,
                              const UnifiedAttentionBinding& binding,
                              std::string_view label)
{
    const auto* q = findTensor(context, binding.q);
    const auto* k = findTensor(context, binding.k);
    if(q == nullptr || k == nullptr || !isWellFormedOperand(*q) || !isWellFormedOperand(*k))
    {
        refuse(label, "the bound Q or K tensor is missing or malformed");
    }
    return problemFor(*q, *k);
}

/// The layout for a candidate's @p blockSize over @p problem, or a refusal that says which rule
/// failed.
inline UnifiedAttentionWorkspaceLayout requireLayout(int64_t blockSize,
                                                     const SdpaShape& problem,
                                                     const KernelDefinition& kernel,
                                                     std::string_view label)
{
    const auto layout = unifiedAttentionWorkspaceLayout(problem.batch, problem.seqLenKv, blockSize);
    if(!layout.has_value())
    {
        refuse(label,
               "kernel '" + hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId)
                   + "' has block_size " + std::to_string(blockSize)
                   + ", which does not divide seqlen_kv " + std::to_string(problem.seqLenKv)
                   + " into a table the kernel can index");
    }
    return *layout;
}

/**
 * @brief The native dispatch behind an engine's UDD.
 *
 * Process-lifetime: the caller keeps @p arch and @p kpackLoader alive for as long as the
 * handler is registered.
 */
template <typename Candidate>
class Unified2DDispatchHandler : public IKernelDispatchHandler<Handle>
{
public:
    Unified2DDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                             const Unified2DArch<Candidate>& arch)
        : _kpackLoader(kpackLoader)
        , _arch(arch)
    {
    }

    /// The identity block table, seq_lens and query_start_len, each section 256-byte aligned.
    size_t workspaceBytes(const MatchContext& context,
                          const BoundTokens& bound,
                          const KernelDefinition& kernel) const override
    {
        const auto binding = readBinding(bound, _arch.graph.tokens);
        const auto problem = boundProblem(context, binding, _arch.graph.label);
        const auto candidate = requireCandidate(kernel);
        return requireLayout(candidate.blockSize, problem, kernel, _arch.graph.label).totalBytes;
    }

    std::unique_ptr<PreparedDispatch> prepare(const MatchContext& context,
                                              const BoundTokens& bound,
                                              const KernelDefinition& kernel) const override
    {
        const auto binding = readBinding(bound, _arch.graph.tokens);

        // graph_match is the gate for this: a non-BSHD output declines there and never
        // reaches prepare(). This re-check is defence in depth for a caller that reaches the
        // handler without having matched, which is why it faults rather than declines: by
        // this point the engine has been chosen and there is no one left to defer to.
        const auto* o = findTensor(context, binding.o);
        if(o == nullptr || !isWellFormedOperand(*o) || !hasBshdStrides(*o))
        {
            refuse(_arch.graph.label,
                   "the output tensor is not dense BSHD; the kernel bakes that layout and "
                   "takes no stride arguments");
        }

        const auto candidate = requireCandidate(kernel);
        // kernelMatches pairs them; a caller that skipped it would launch the wrong kernarg
        // list.
        if(candidate.emitLse != (binding.stats != NO_STATS))
        {
            refuse(_arch.graph.label,
                   "kernel '" + hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId)
                       + (candidate.emitLse ? "' writes stats, but the graph asks for none"
                                            : "' writes no stats, but the graph asks for them"));
        }
        const auto problem = boundProblem(context, binding, _arch.graph.label);
        if(_arch.prepareChecks != nullptr)
        {
            _arch.prepareChecks(candidate, problem, kernel);
        }
        const auto layout = requireLayout(candidate.blockSize, problem, kernel, _arch.graph.label);

        // The host-side metadata is built before the code object is loaded, so a graph the
        // tables cannot describe is refused by name without touching an archive.
        auto image = unifiedAttentionWorkspaceImage(
            problem.batch, problem.seqLenQ, problem.seqLenKv, candidate.blockSize);
        if(!image.has_value() || image->size() != layout.totalBytes)
        {
            refuse(_arch.graph.label,
                   "kernel '" + hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId)
                       + "' cannot stage paged-KV metadata for this graph");
        }

        auto code = buildIngestorKernelCode(
            _kpackLoader, context, kernel, unified2DKernelSignature(candidate.emitLse));

        // Grid from the SELECTED CANDIDATE'S geometry and the GRAPH PROBLEM, not from
        // descriptor shape metadata: the binary takes its shape at runtime.
        const auto geometry = unifiedAttention2DLaunchGeometry(
            *_arch.graph.traits,
            candidate.numWarps,
            candidate.blockMPerWarp,
            problem.numQueryHeads,
            problem.numKvHeads,
            problem.seqLenQ,
            problem.batch,
            hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId));

        code.setBlockSize(geometry.blockX, 1, 1);
        code.setGridSize(geometry.gridX, geometry.gridY, geometry.gridZ);

        return std::make_unique<PreparedUnified2D>(
            std::move(code), binding, problem, layout, std::move(*image), candidate.slidingWindow);
    }

    void launch(const Handle& handle,
                const PreparedDispatch& prepared,
                const hipdnnPluginDeviceBuffer_t* deviceBuffers,
                uint32_t numDeviceBuffers,
                void* workspace) const override
    {
        const auto& preparedUnified = dynamic_cast<const PreparedUnified2D&>(prepared);
        const auto& binding = preparedUnified.binding();
        const auto& layout = preparedUnified.layout();
        const auto& image = preparedUnified.workspaceImage();

        if(workspace == nullptr)
        {
            refuse(_arch.graph.label,
                   "launched without the " + std::to_string(image.size())
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

        // One copy of the whole image, on the handle's stream so the kernel launched next
        // sees it. The image is immutable, and a pageable source is staged by the runtime
        // before the call returns, so concurrent launches of one plan share it safely.
        hipStream_t stream = handle.getStream();
        const hipError_t copied
            = hipMemcpyAsync(workspace, image.data(), image.size(), hipMemcpyHostToDevice, stream);
        if(copied != hipSuccess)
        {
            throw hipdnn_plugin_sdk::HipdnnPluginException(
                HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
                errorPrefix(_arch.graph.label)
                    + "staging the paged-KV metadata failed: " + hipGetErrorString(copied));
        }

        auto* const base = static_cast<unsigned char*>(workspace);
        void* const blockTables = base + layout.blockTablesOffset;
        void* const seqLens = base + layout.seqLensOffset;
        void* const queryStartLen = base + layout.queryStartLenOffset;

        // Unused pointers are null and the unused scalars neutral: the shipped variants have
        // no sinks, ALiBi, qq-bias, softcap or fp8, so the kernel never reads them.
        void* const unused = nullptr;
        const float neutralScale = 1.0F;
        const float noSoftcap = 0.0F;
        const int32_t noQqBiasStride = 0;

        const auto& p = preparedUnified.problem();
        // The kernarg list, which an LSE-on kernel ends with the stats output.
        const auto launchWith = [&](auto... statsArgument) {
            preparedUnified.kernelForStream(stream).launch(
                stream,
                o.ptr,
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
                neutralScale,
                noSoftcap,
                static_cast<int32_t>(p.batch),
                static_cast<int32_t>(layout.blocksPerSeq),
                noQqBiasStride,
                static_cast<int32_t>(p.numQueryHeads),
                static_cast<int32_t>(p.numKvHeads),
                static_cast<int32_t>(preparedUnified.slidingWindow()),
                statsArgument...);
        };
        if(binding.stats != NO_STATS)
        {
            const auto stats = hipdnn_plugin_sdk::findDeviceBuffer(
                binding.stats, deviceBuffers, numDeviceBuffers);
            launchWith(stats.ptr);
            return;
        }
        launchWith();
    }

private:
    /// The candidate, or a refusal that names the kernel: kernel_match declines a candidate it
    /// cannot launch, so this is defence in depth for a caller that reaches the handler without
    /// having matched. The grid and CTA come from it, and there is no geometry to substitute
    /// that would not launch this binary with another binary's.
    Candidate requireCandidate(const KernelDefinition& kernel) const
    {
        const auto candidate = _arch.candidateFor(kernel);
        if(!candidate.has_value())
        {
            refuse(_arch.graph.label,
                   "kernel '" + hipdnn_plugin_sdk::ingestor::toString(kernel.kernelId)
                       + "' declares no supported geometry");
        }
        return *candidate;
    }

    const compilation::KpackKernelLoader& _kpackLoader;
    const Unified2DArch<Candidate>& _arch;
};

} // namespace hip_kernel_provider::kernel_ingestor_engine::unified_attention

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
