// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file UnifiedAttentionGeometry.hpp
 * @brief The arch-neutral half of the unified-attention (vLLM `unified_attention`) packs:
 *        launch grid, batch capacity and the identity paged-KV metadata a contiguous K/V
 *        graph is served with, parameterized by the target's UnifiedAttentionArchTraits.
 *
 * rocKE builds the unified-attention kernels per architecture, but the host contract around
 * them is shared: the kernel reads a paged K/V cache through `block_tables`, per-sequence
 * lengths from `seq_lens` and ragged Q offsets from `query_start_len`, and a contiguous BSHD
 * K/V tensor IS that cache with an identity block table. What differs per architecture is
 * the wave width, whether the grid is kv-major or q-major, and the geometry the builder
 * accepts. Those are the traits; a pack for another architecture (gfx942, AICK-2544)
 * supplies its own traits and its own builder-legality check and reuses everything here.
 *
 * Pure functions of KMD fields and graph dimensions: testable with no HIP context.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// What one architecture's unified-attention kernels fix about their launch and addressing.
struct UnifiedAttentionArchTraits
{
    /// Prefix for diagnostics, e.g. "gfx950".
    const char* arch = "";
    /// Lanes per wave; a CTA has `num_warps * waveLanes` threads (`launch_block`).
    int64_t waveLanes = 64;
    /// q-major grid `(qblocks, Hkv, 1)` instead of kv-major `(Hkv, qblocks, 1)`
    /// (`use_q_major_grid` in the dispatcher's launch meta).
    bool qMajorGrid = false;
    /// The K/V byte bound of the 32-bit addressing every shipped kernel is built with
    /// (`_I32_KV_BUFFER_BYTES`); past it buffer loads return zeros.
    int64_t kvBufferBytes = 0x7FFF0000;
    /// `block_size` values the paged cache may be cut at.
    std::array<int64_t, 3> blockSizes = {16, 32, 64};
    /// The most query heads one key/value head may serve.
    int64_t maxQueriesPerKv = 16;
    /// The largest KV tile, in pages (the dispatcher's 1x/2x/4x/8x tile policies).
    int64_t maxTileBlocks = 8;
    /// `num_warps` and `block_m_per_warp` values the 2D builder takes.
    std::array<int64_t, 4> numWarps = {1, 2, 4, 8};
    std::array<int64_t, 2> blockMPerWarp = {16, 32};
};

/// gfx950 (CDNA4): wave64, kv-major grid.
inline constexpr UnifiedAttentionArchTraits GFX950_UNIFIED_ATTENTION_TRAITS{"gfx950"};

/// Each workspace section starts on this boundary.
inline constexpr std::size_t UNIFIED_ATTENTION_WORKSPACE_ALIGNMENT = 256;

/// Bytes of one f32 element of the split-KV segment scratch.
inline constexpr std::size_t UNIFIED_ATTENTION_SCRATCH_ELEMENT = sizeof(float);

namespace detail
{

template <std::size_t N>
inline bool isOneOf(const std::array<int64_t, N>& values, int64_t value)
{
    for(const auto v : values)
    {
        if(v == value)
        {
            return true;
        }
    }
    return false;
}

inline std::size_t alignWorkspaceSection(std::size_t bytes)
{
    constexpr std::size_t A = UNIFIED_ATTENTION_WORKSPACE_ALIGNMENT;
    return (bytes + A - 1) / A * A;
}

} // namespace detail

inline bool isSupportedUnifiedAttentionBlockSize(const UnifiedAttentionArchTraits& traits,
                                                 int64_t blockSize)
{
    return detail::isOneOf(traits.blockSizes, blockSize);
}

/**
 * @brief Binary-search trip count a kernel was compiled with for a baked `num_seqs`.
 *
 * `binary_search_seq_idx_iters`: `max(1, ceil(log2(num_seqs + 1)))`, and 32 for a spec that
 * declares no `num_seqs`. The same on every architecture.
 */
inline int64_t unifiedAttentionBinarySearchIters(int64_t numSeqs)
{
    constexpr int64_t UNSPECIALIZED_ITERS = 32;
    if(numSeqs <= 0)
    {
        return UNSPECIALIZED_ITERS;
    }
    int64_t iters = 1;
    // Smallest iters with 2^iters >= numSeqs + 1; bounded by the shift width anyway.
    while(iters < UNSPECIALIZED_ITERS && (int64_t{1} << iters) < numSeqs + 1)
    {
        ++iters;
    }
    return iters;
}

/**
 * @brief The most sequences one launch of a kernel baked with @p numSeqs can address.
 *
 * The kernel bisects `[0, num_seqs_runtime]` in a fixed number of iterations (an unrolled
 * scf.for), which resolves at most `2^iters - 1` sequences. A larger batch leaves the search
 * unfinished: the last blocks read the wrong sequence's lengths and the launch still
 * succeeds. 511 -> 9 iterations -> 511.
 */
inline int64_t unifiedAttentionBatchCapacity(int64_t numSeqs)
{
    return (int64_t{1} << unifiedAttentionBinarySearchIters(numSeqs)) - 1;
}

/// The grid and block a candidate must launch with.
struct UnifiedAttentionLaunchGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const UnifiedAttentionLaunchGeometry& a,
                           const UnifiedAttentionLaunchGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The 2D launch geometry for one candidate over one graph.
 *
 * Mirrors `AttentionTuningSpec.launch_grid` for the tiled 2D path:
 *
 *     block_m = block_m_per_warp * num_warps
 *     block_q = block_m // num_queries_per_kv
 *     qblocks = total_q // block_q + num_seqs
 *     grid    = (num_kv_heads, qblocks, 1)      kv-major
 *             = (qblocks, num_kv_heads, 1)      q-major (traits.qMajorGrid)
 *
 * and `launch_block`, `(waveLanes * num_warps, 1, 1)`. `total_q` is `batch * seqLenQ` and
 * `num_seqs` the RUNTIME batch, not the baked capacity.
 *
 * Throws instead of returning a degenerate grid: an empty launch returns cleanly having
 * written nothing, and prepare() is the last place a named failure is cheap.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline UnifiedAttentionLaunchGeometry
    unifiedAttention2DLaunchGeometry(const UnifiedAttentionArchTraits& traits,
                                     int64_t numWarps,
                                     int64_t blockMPerWarp,
                                     int64_t numQueryHeads,
                                     int64_t numKvHeads,
                                     int64_t seqLenQ,
                                     int64_t batch,
                                     const std::string& kernelName)
{
    const auto refuse = [&traits, &kernelName](const std::string& why) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            std::string(traits.arch) + " attention_unified_2d: kernel '" + kernelName + "' " + why);
    };

    if(!detail::isOneOf(traits.numWarps, numWarps))
    {
        refuse("declares num_warps " + std::to_string(numWarps) + ", which " + traits.arch
               + " does not build");
    }
    if(!detail::isOneOf(traits.blockMPerWarp, blockMPerWarp))
    {
        refuse("declares block_m_per_warp " + std::to_string(blockMPerWarp) + ", which "
               + traits.arch + " does not build");
    }
    if(numQueryHeads <= 0 || numKvHeads <= 0 || seqLenQ <= 0 || batch <= 0)
    {
        refuse("is asked to launch a non-positive num_query_heads, num_kv_heads, seqlen_q or "
               "batch");
    }
    if(numQueryHeads % numKvHeads != 0 || numQueryHeads / numKvHeads > traits.maxQueriesPerKv)
    {
        refuse("is asked to launch " + std::to_string(numQueryHeads) + " query heads over "
               + std::to_string(numKvHeads)
               + " key/value heads; the kernel needs a whole group of 1 to "
               + std::to_string(traits.maxQueriesPerKv));
    }

    const int64_t blockM = blockMPerWarp * numWarps;
    const int64_t queriesPerKv = numQueryHeads / numKvHeads;
    const int64_t blockQ = blockM / queriesPerKv;
    int64_t totalQ = 0;
    if(__builtin_mul_overflow(batch, seqLenQ, &totalQ))
    {
        refuse("is asked to launch a batch * seqlen_q that overflows");
    }
    const int64_t qBlocks = totalQ / blockQ + batch;
    // One axis carries the kv head count, the other the Q blocks: both must fit the 32-bit
    // grid dims.
    if(qBlocks > INT32_MAX || numKvHeads > INT32_MAX)
    {
        refuse("is asked to launch a grid wider than 2^31 - 1 blocks");
    }

    UnifiedAttentionLaunchGeometry geometry;
    geometry.blockX = static_cast<unsigned>(numWarps * traits.waveLanes);
    geometry.gridX = static_cast<unsigned>(traits.qMajorGrid ? qBlocks : numKvHeads);
    geometry.gridY = static_cast<unsigned>(traits.qMajorGrid ? numKvHeads : qBlocks);
    geometry.gridZ = 1;
    return geometry;
}

/// Where each identity paged-KV metadata array sits in the workspace, in bytes.
struct UnifiedAttentionWorkspaceLayout
{
    int64_t blocksPerSeq = 0;
    std::size_t blockTablesOffset = 0;
    std::size_t blockTablesBytes = 0;
    std::size_t seqLensOffset = 0;
    std::size_t seqLensBytes = 0;
    std::size_t queryStartLenOffset = 0;
    std::size_t queryStartLenBytes = 0;
    std::size_t totalBytes = 0;

    friend bool operator==(const UnifiedAttentionWorkspaceLayout& a,
                           const UnifiedAttentionWorkspaceLayout& b)
    {
        return a.blocksPerSeq == b.blocksPerSeq && a.blockTablesOffset == b.blockTablesOffset
               && a.blockTablesBytes == b.blockTablesBytes && a.seqLensOffset == b.seqLensOffset
               && a.seqLensBytes == b.seqLensBytes && a.queryStartLenOffset == b.queryStartLenOffset
               && a.queryStartLenBytes == b.queryStartLenBytes && a.totalBytes == b.totalBytes;
    }
};

/**
 * @brief The workspace the identity paged-KV metadata occupies, or nullopt when it cannot
 *        be laid out.
 *
 * Three int32 arrays, each starting on a 256-byte boundary:
 *
 *     block_tables     [batch, Skv / block_size]   section 0
 *     seq_lens         [batch]                     section 1
 *     query_start_len  [batch + 1]                 section 2
 *
 * nullopt for a non-positive dimension, a @p blockSize that does not divide @p seqLenKv
 * (the kernel reads whole blocks, and a partial one is not a contiguous cache), or a
 * table the kernel's 32-bit `num_seqs * block_table_stride` bound cannot index.
 */
inline std::optional<UnifiedAttentionWorkspaceLayout>
    unifiedAttentionWorkspaceLayout(int64_t batch, int64_t seqLenKv, int64_t blockSize)
{
    if(batch <= 0 || seqLenKv <= 0 || blockSize <= 0 || seqLenKv % blockSize != 0)
    {
        return std::nullopt;
    }

    UnifiedAttentionWorkspaceLayout layout;
    layout.blocksPerSeq = seqLenKv / blockSize;

    int64_t entries = 0;
    if(__builtin_mul_overflow(batch, layout.blocksPerSeq, &entries) || entries > INT32_MAX)
    {
        return std::nullopt;
    }

    constexpr std::size_t ELEMENT = sizeof(int32_t);
    layout.blockTablesOffset = 0;
    layout.blockTablesBytes = static_cast<std::size_t>(entries) * ELEMENT;
    layout.seqLensOffset = detail::alignWorkspaceSection(layout.blockTablesBytes);
    layout.seqLensBytes = static_cast<std::size_t>(batch) * ELEMENT;
    layout.queryStartLenOffset
        = layout.seqLensOffset + detail::alignWorkspaceSection(layout.seqLensBytes);
    layout.queryStartLenBytes = static_cast<std::size_t>(batch + 1) * ELEMENT;
    layout.totalBytes
        = layout.queryStartLenOffset + detail::alignWorkspaceSection(layout.queryStartLenBytes);
    return layout;
}

/**
 * @brief The bytes launch() copies into the workspace: the identity paged-KV metadata.
 *
 *   block_tables[b][j] = b * blocksPerSeq + j        (the cache IS the K/V tensor)
 *   seq_lens[b]        = seqLenKv
 *   query_start_len[b] = b * seqLenQ                 (b = 0..batch)
 *
 * Padding between sections is zero. nullopt exactly when unifiedAttentionWorkspaceLayout()
 * is, and when `batch * seqLenQ` does not fit int32.
 */
inline std::optional<std::vector<unsigned char>> unifiedAttentionWorkspaceImage(int64_t batch,
                                                                                int64_t seqLenQ,
                                                                                int64_t seqLenKv,
                                                                                int64_t blockSize)
{
    const auto layout = unifiedAttentionWorkspaceLayout(batch, seqLenKv, blockSize);
    int64_t totalQ = 0;
    if(!layout.has_value() || seqLenQ <= 0 || __builtin_mul_overflow(batch, seqLenQ, &totalQ)
       || totalQ > INT32_MAX)
    {
        return std::nullopt;
    }

    std::vector<unsigned char> image(layout->totalBytes, 0);
    const auto put = [&image](std::size_t offset, std::size_t index, int64_t value) {
        const auto narrow = static_cast<int32_t>(value);
        std::memcpy(image.data() + offset + index * sizeof(int32_t), &narrow, sizeof(narrow));
    };

    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t j = 0; j < layout->blocksPerSeq; ++j)
        {
            put(layout->blockTablesOffset,
                static_cast<std::size_t>(b * layout->blocksPerSeq + j),
                b * layout->blocksPerSeq + j);
        }
        put(layout->seqLensOffset, static_cast<std::size_t>(b), seqLenKv);
    }
    for(int64_t b = 0; b <= batch; ++b)
    {
        put(layout->queryStartLenOffset, static_cast<std::size_t>(b), b * seqLenQ);
    }
    return image;
}

/// Where each workspace section sits, in bytes: the identity paged-KV metadata (exactly
/// the 2D layout), then the three split-KV segment scratch arrays.
struct UnifiedAttention3DWorkspaceLayout
{
    UnifiedAttentionWorkspaceLayout identity;
    std::size_t segmOutputOffset = 0;
    std::size_t segmOutputBytes = 0;
    std::size_t segmMaxOffset = 0;
    std::size_t segmMaxBytes = 0;
    std::size_t segmExpsumOffset = 0;
    std::size_t segmExpsumBytes = 0;
    std::size_t totalBytes = 0;

    friend bool operator==(const UnifiedAttention3DWorkspaceLayout& a,
                           const UnifiedAttention3DWorkspaceLayout& b)
    {
        return a.identity == b.identity && a.segmOutputOffset == b.segmOutputOffset
               && a.segmOutputBytes == b.segmOutputBytes && a.segmMaxOffset == b.segmMaxOffset
               && a.segmMaxBytes == b.segmMaxBytes && a.segmExpsumOffset == b.segmExpsumOffset
               && a.segmExpsumBytes == b.segmExpsumBytes && a.totalBytes == b.totalBytes;
    }
};

/**
 * @brief The workspace both kernels share, or nullopt when it cannot be laid out.
 *
 * Six sections, each starting on a 256-byte boundary:
 *
 *     block_tables     int32 [batch, Skv / block_size]          the identity metadata,
 *     seq_lens         int32 [batch]                            laid out as the 2D
 *     query_start_len  int32 [batch + 1]                        kernels read it
 *     segm_output      f32   [total_q, Hq, num_segments, D]
 *     segm_max         f32   [total_q, Hq, num_segments]
 *     segm_expsum      f32   [total_q, Hq, num_segments]
 *
 * The reduce kernel's descriptors index the scratch exactly so
 * (`build_unified_attention_reduce_tiled`). Only the identity prefix is host-written; the
 * segment kernel writes every scratch element the reduce reads, including a neutral entry
 * for a segment past the sequence's end, so the scratch needs no initialisation.
 *
 * nullopt whenever unifiedAttentionWorkspaceLayout() is, for a non-positive scratch
 * dimension, and for a scratch the kernels' 32-bit element indexing cannot address.
 */
inline std::optional<UnifiedAttention3DWorkspaceLayout>
    unifiedAttention3DWorkspaceLayout(int64_t batch,
                                      int64_t seqLenQ,
                                      int64_t seqLenKv,
                                      int64_t blockSize,
                                      int64_t numQueryHeads,
                                      int64_t headSize,
                                      int64_t numSegments)
{
    const auto identity = unifiedAttentionWorkspaceLayout(batch, seqLenKv, blockSize);
    if(!identity.has_value() || seqLenQ <= 0 || numQueryHeads <= 0 || headSize <= 0
       || numSegments <= 0)
    {
        return std::nullopt;
    }

    int64_t rows = 0;
    int64_t outputElements = 0;
    if(__builtin_mul_overflow(batch, seqLenQ, &rows)
       || __builtin_mul_overflow(rows, numQueryHeads, &rows)
       || __builtin_mul_overflow(rows, numSegments, &rows)
       || __builtin_mul_overflow(rows, headSize, &outputElements) || outputElements > INT32_MAX)
    {
        return std::nullopt;
    }

    UnifiedAttention3DWorkspaceLayout layout;
    layout.identity = *identity;
    layout.segmOutputOffset = identity->totalBytes;
    layout.segmOutputBytes
        = static_cast<std::size_t>(outputElements) * UNIFIED_ATTENTION_SCRATCH_ELEMENT;
    layout.segmMaxOffset
        = layout.segmOutputOffset + detail::alignWorkspaceSection(layout.segmOutputBytes);
    layout.segmMaxBytes = static_cast<std::size_t>(rows) * UNIFIED_ATTENTION_SCRATCH_ELEMENT;
    layout.segmExpsumOffset
        = layout.segmMaxOffset + detail::alignWorkspaceSection(layout.segmMaxBytes);
    layout.segmExpsumBytes = layout.segmMaxBytes;
    layout.totalBytes
        = layout.segmExpsumOffset + detail::alignWorkspaceSection(layout.segmExpsumBytes);
    return layout;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
