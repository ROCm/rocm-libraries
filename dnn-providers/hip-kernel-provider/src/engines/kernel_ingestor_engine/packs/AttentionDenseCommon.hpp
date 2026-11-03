// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/ingestor/Descriptors.hpp>
#include <hipdnn_plugin_sdk/ingestor/IKernelDispatchHandler.hpp>
#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>
#include <hipdnn_plugin_sdk/ingestor/SavedDispatch.hpp>

/**
 * @file AttentionDenseCommon.hpp
 * @brief What the gfx942 and gfx950 dense-attention ingestor packs share: the launch state
 *        and its saved form, and the matching and dispatch both engines run, parameterised
 *        by an AttentionDenseArch descriptor.
 *
 * The two kernels are `rocke/library/kernels/gfx942/attention_dense.py` and
 * `rocke/library/kernels/gfx950/attention_dense.py`. They take the same ABI, launch on the
 * same grid formula and are bound from the same graph contract, so one copy of that code
 * serves both engines. What differs per arch reaches it through the descriptor: the tile
 * rules and launch geometry, the scale range, whether a causal sliding window is served,
 * whether the candidate's head counts are compared, and the launch contracts. Scoring and
 * every read of a KMD field only one engine declares stay in that engine's own
 * `Gfx9xxAttentionDenseNative.cpp`, which also owns the module cache and registration.
 */

// The provider's plugin handle (core/Handle.hpp), named here only as the handler's handle type.
struct Handle;

namespace hip_kernel_provider::compilation
{
class KpackKernelLoader;
}

namespace hip_kernel_provider::kernel_ingestor_engine
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// The tensor uids and derived scalars a matched dense-attention graph binds.
struct AttentionDenseBinding
{
    int64_t q = 0;
    int64_t k = 0;
    int64_t v = 0;
    int64_t o = 0;
    int64_t causal = 0;
    int64_t slidingWindow = 0;
    float scale = 0.0F;
};

/// The graph facts the matcher and prepare() both need, derived once from the tensors.
struct AttentionDenseProblem
{
    int64_t batch = 0;
    int64_t seqLenQ = 0;
    int64_t seqLenKv = 0;
    int64_t numQueryHeads = 0;
    int64_t numKvHeads = 0;
    int64_t headSize = 0;
    data_objects::DataType dataType = data_objects::DataType::UNSET;
};

/// Everything a dense-attention launch is computed from.
struct AttentionDenseLaunchInputs
{
    AttentionDenseBinding binding;
    AttentionDenseProblem problem;
    int64_t blockM = 0;
};

namespace attention_dense
{

/// The launch values both engines' contracts store. `scale` is stored as a double, which
/// holds every float exactly. `data_type` is the DataType enumerator name.
inline constexpr std::string_view Q_UID_VALUE = "q.uid";
inline constexpr std::string_view K_UID_VALUE = "k.uid";
inline constexpr std::string_view V_UID_VALUE = "v.uid";
inline constexpr std::string_view O_UID_VALUE = "o.uid";
inline constexpr std::string_view CAUSAL_VALUE = "causal";
inline constexpr std::string_view SLIDING_WINDOW_VALUE = "sliding_window";
inline constexpr std::string_view BATCH_VALUE = "batch";
inline constexpr std::string_view SEQLEN_Q_VALUE = "seqlen_q";
inline constexpr std::string_view SEQLEN_KV_VALUE = "seqlen_kv";
inline constexpr std::string_view NUM_QUERY_HEADS_VALUE = "num_query_heads";
inline constexpr std::string_view NUM_KV_HEADS_VALUE = "num_kv_heads";
inline constexpr std::string_view HEAD_SIZE_VALUE = "head_size";
inline constexpr std::string_view BLOCK_M_VALUE = "block_m";
inline constexpr std::string_view SCALE_VALUE = "scale";
inline constexpr std::string_view DATA_TYPE_VALUE = "data_type";
inline constexpr std::string_view STRIDE_LAYOUT_VALUE = "stride_layout";

/// The one stride layout the kernels support. prepare() refuses every other layout.
inline constexpr std::string_view BSHD_LAYOUT = "bshd";

/// The kernel arguments a launch contract passes.
enum class AttentionDenseAbi
{
    /// q, k, v, o, scale, batch, seqlen_q, seqlen_kv: kernels that bake the head counts.
    BAKED_HEAD_COUNTS,
    /// The same, then num_query_heads and num_kv_heads: kernels that read them at runtime.
    RUNTIME_HEAD_COUNTS
};

/// A versioned dispatch name, under which a plan's launch values are saved, and the kernel
/// arguments a launch under it passes.
struct AttentionDenseContract
{
    std::string_view symbol;
    AttentionDenseAbi abi = AttentionDenseAbi::RUNTIME_HEAD_COUNTS;
};

/// The tokens an engine's graph match binds, spelled per engine.
struct AttentionDenseTokens
{
    std::string_view q;
    std::string_view k;
    std::string_view v;
    std::string_view o;
    /// The mask type graphMatches derived, so kernelMatches does not re-derive it.
    std::string_view causal;
    /// The sliding window the candidate must have been built with: the number of keys a
    /// query attends, counting the diagonal (the graph's left_bound), or 0 for none.
    std::string_view slidingWindow;
    /// The softmax scale, f32, bound as its bit pattern (BoundTokens carry int64_t).
    std::string_view scaleBits;
};

/// The grid and block a variant must launch with.
struct AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;
};

/**
 * @brief Everything the shared matching and dispatch ask of one arch.
 *
 * Each engine defines one at namespace scope in its own native file. The function pointers
 * are that arch's restatement of its own Python (its Gfx9xxAttentionDenseGeometry.hpp).
 */
struct AttentionDenseArch
{
    /// "gfx942" or "gfx950": the head of every exception this module throws for the arch.
    std::string_view label;
    /// The engine name its decline lines carry.
    std::string_view engineName;
    AttentionDenseTokens tokens;
    /// Whether kernelMatches compares the candidate's num_query_heads/num_kv_heads with the
    /// graph's: true where the catalog lists each head configuration as its own entry.
    bool matchesHeadCounts = false;
    /// The softmax scales the kernel computes correctly, and the decline and restore text
    /// for any other.
    bool (*isSupportedScale)(float scale) = nullptr;
    std::string_view unsupportedScaleReason;
    /// Whether (blockM, blockN) is a tile the kernel is built with at headSize. Total over
    /// every int64 triple.
    bool (*isSupportedTile)(int64_t headSize, int64_t blockM, int64_t blockN) = nullptr;
    /// The launch geometry of a candidate's own block_m over a graph's shape. Throws
    /// HipdnnPluginException with HIPDNN_PLUGIN_STATUS_BAD_PARAM for a block_m the arch
    /// does not build or a non-positive extent.
    AttentionDenseGeometry (*geometry)(int64_t blockM,
                                       int64_t seqLenQ,
                                       int64_t numQueryHeads,
                                       int64_t batch,
                                       const std::string& kernelName)
        = nullptr;
    /// The sliding-window zero-trip guard for one candidate's tile, or nullptr for an arch
    /// that serves no window: its graph match declines every windowed mask.
    bool (*windowReachesKeys)(
        int64_t blockM, int64_t blockN, int64_t seqLenQ, int64_t seqLenKv, int64_t slidingWindow)
        = nullptr;
    /// Whether restore refuses saved values the graph and kernel match never bind: causal
    /// outside {0, 1}, a negative window, a window without causal, or one that leaves query
    /// rows no key; num_query_heads not a multiple of num_kv_heads, a head_size other than
    /// 64 or 128, K/V of 2^31 bytes or Q/O of 2^31 elements, or seqlen_q not a multiple of
    /// block_m.
    bool refusesUnservedSavedValues = false;
    /// The contract the engine's catalog dispatches through and new plans are saved under.
    AttentionDenseContract currentContract;
    /// A superseded contract whose plans still restore, or an empty symbol for none. It
    /// shares currentContract's launch values and differs only in its kernel arguments.
    AttentionDenseContract legacyContract;

    bool servesSlidingWindow() const
    {
        return windowReachesKeys != nullptr;
    }
};

/**
 * @brief Graph-scoped applicability for the whole engine.
 *
 * Each decline of a graph with an SDPA-forward node logs its cause at INFO; any other
 * graph declines silently.
 *
 * @warning Returning std::nullopt empties the engine's WHOLE catalog and skips EVERY
 *          remaining pack, not just this one.
 */
std::optional<hipdnn_plugin_sdk::ingestor::BoundTokens>
    graphMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                 const AttentionDenseArch& arch);

/// Kernel-scoped applicability: does THIS candidate's baked metadata fit the graph?
bool kernelMatches(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                   const hipdnn_plugin_sdk::ingestor::BoundTokens& bound,
                   const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                   const AttentionDenseArch& arch);

/// A candidate's integer metadata field, or nullopt when it is absent or holds another
/// type. Total: never throws, so a malformed record declines instead of faulting a match.
std::optional<int64_t> integerMetadata(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                                       std::string_view field);

/// A candidate's (block_m, block_n) tile, read from its completed metadata.
struct AttentionDenseTile
{
    int64_t blockM = 0;
    int64_t blockN = 0;
};

/**
 * @brief The candidate's tile, or nullopt when it has none the arch can launch.
 *
 * Validated against the arch's isSupportedTile. A record omitting block_m or block_n
 * reaches here completed to 256/64 through the KMD defaults, so an absent field means an
 * uncompleted or malformed record: it is declined rather than given a default here,
 * because a substituted tile launches a binary with another binary's grid.
 */
std::optional<AttentionDenseTile>
    candidateTile(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel,
                  const AttentionDenseArch& arch);

/**
 * @brief The native dispatch behind an engine's UDD, for one launch contract.
 *
 * The handler of a contract saves new plans under that contract's name and restores only
 * plans saved under it. Process-lifetime: the caller keeps it alive for as long as it is
 * registered, and @p kpackLoader and @p arch for as long as the handler lives.
 */
std::unique_ptr<hipdnn_plugin_sdk::ingestor::IKernelDispatchHandler<Handle>>
    makeDispatchHandler(const compilation::KpackKernelLoader& kpackLoader,
                        const AttentionDenseArch& arch,
                        const AttentionDenseContract& contract);

/// The launch values both engines save, one value set for every contract.
hipdnn_plugin_sdk::ingestor::MetadataValues launchValues(const AttentionDenseBinding& binding,
                                                         const AttentionDenseProblem& problem,
                                                         int64_t blockM);

/**
 * @brief Reads @p arch's launch values back into the launch state.
 *
 * The dispatch name must be the arch's current or legacy contract; anything else,
 * including the unversioned name, is refused against the current one. `batch`, `seqlen_q`
 * and `seqlen_kv`, the head counts and `head_size` must be in [1, INT32_MAX], because the
 * kernel takes them as 32-bit integers. `block_m` must be a tile the arch builds. `scale`
 * must convert to a float and back without change and be a scale the arch supports.
 * `data_type` must be a type the kernel is built for, and `stride_layout` must be `bshd`.
 * An arch with refusesUnservedSavedValues also refuses a mask or shape the graph and kernel
 * match never bind.
 *
 * @throws HipdnnPluginException with a load refusal when the dispatch name is neither
 *         contract, or when a value is missing, extra, of another type or out of range.
 */
AttentionDenseLaunchInputs
    readLaunchInputs(const hipdnn_plugin_sdk::ingestor::SavedLaunchInputs& inputs,
                     const AttentionDenseArch& arch);

} // namespace attention_dense
} // namespace hip_kernel_provider::kernel_ingestor_engine

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
