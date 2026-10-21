// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx950AttentionUnified2DGeometry.hpp
 * @brief What gfx950 unified-attention 2D restates from its Python builder: the geometry
 *        rules, the launch grid, the batch capacity and the paged-KV metadata the host
 *        derives for a contiguous K/V graph.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same binary
 * from C++. Nothing compares the two halves, and a mismatch does not fault: the kernel
 * computes something else. Everything here is a pure function of a candidate's KMD fields
 * and the graph's dimensions, so it is testable with no HIP context.
 *
 * The kernel is vLLM's paged `unified_attention`: Q is ragged (`query_start_len`), K/V live
 * in a paged cache addressed through `block_tables`, and per-sequence lengths come from
 * `seq_lens`. A hipDNN SDPA graph carries none of that. Its K and V are contiguous BSHD
 * tensors, which ARE a paged cache whose block table is the identity: sequence `b` owns
 * blocks `[b * blocksPerSeq, (b + 1) * blocksPerSeq)`. The host builds those three small
 * int32 arrays (workspaceImage) and the launch copies them into the plan's workspace.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave64 wave: a CTA has `num_warps * 64` threads
/// (`AttentionTuningSpec.launch_block`).
inline constexpr int64_t GFX950_UNIFIED_2D_WAVE_LANES = 64;

/// The kernel indexes K/V with a 32-bit byte offset when `use_i64_kv_addr` is off, and
/// bounds its buffer loads at this size (`_I32_KV_BUFFER_BYTES`); past it loads return
/// zeros. Every shipped kernel is built i32.
inline constexpr int64_t GFX950_UNIFIED_2D_KV_BUFFER_BYTES = 0x7FFF0000;

/// `block_size` values the tiled 2D kernel is built for (`supports_tiled_2d`).
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL = 16;
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_MEDIUM = 32;
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_LARGE = 64;

/// `1 <= num_queries_per_kv <= 16` (`supports_tiled_2d`).
inline constexpr int64_t GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV = 16;

/// The largest KV tile, in pages: the dispatcher's tile policies are 1x, 2x, 4x and 8x.
inline constexpr int64_t GFX950_UNIFIED_2D_MAX_TILE_BLOCKS = 8;

/// Each workspace section starts on this boundary.
inline constexpr std::size_t GFX950_UNIFIED_2D_WORKSPACE_ALIGNMENT = 256;

inline bool isSupportedGfx950Unified2DBlockSize(int64_t blockSize)
{
    return blockSize == GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL
           || blockSize == GFX950_UNIFIED_2D_BLOCK_SIZE_MEDIUM
           || blockSize == GFX950_UNIFIED_2D_BLOCK_SIZE_LARGE;
}

/**
 * @brief The geometry fields a candidate carries, as the Python spec validates them.
 *
 * `UnifiedAttention2DTiledSpec.__post_init__` and `supports_tiled_2d`, conjoined:
 *   - head_size in {64, 128, 256}, divisible by 32 -- the engine ships 64 and 128
 *     (@p headSize is range-checked by the matcher, not here)
 *   - block_size in {16, 32, 64}
 *   - num_warps in {1, 2, 4, 8}; block_m_per_warp in {16, 32}; 32 needs num_warps <= 4
 *   - tile_size a positive multiple of block_size, at most 8 * block_size (the dispatcher's
 *     1x/2x/4x/8x tile policies), with `tile_size * head_size >= threads * 8` and
 *     `512 / head_size <= block_size` (a wave must not straddle blocks)
 *   - use_mfma_32x32 needs block_m_per_warp 32, tile_size % 32 == 0 and head_size % 16 == 0;
 *     use_transposed_qk_32x32 needs use_mfma_32x32
 *
 * The LDS budget is NOT restated: hkp_pack compiles every shipped kernel, and rocKE's
 * registry (`_gfx950_2d_legality`) refuses an over-budget spec before it can ship.
 *
 * Total over every int64 input: each divisor is checked positive before it is used.
 */
struct Gfx950Unified2DGeometryFields
{
    int64_t headSize = 0;
    int64_t blockSize = 0;
    int64_t tileSize = 0;
    int64_t numWarps = 0;
    int64_t blockMPerWarp = 0;
    bool useMfma32x32 = false;
    bool useTransposedQk32x32 = false;
};

inline bool isSupportedGfx950Unified2DGeometry(const Gfx950Unified2DGeometryFields& g)
{
    if(g.headSize <= 0 || g.headSize % 32 != 0 || !isSupportedGfx950Unified2DBlockSize(g.blockSize))
    {
        return false;
    }
    if(g.numWarps != 1 && g.numWarps != 2 && g.numWarps != 4 && g.numWarps != 8)
    {
        return false;
    }
    if(g.blockMPerWarp != 16 && g.blockMPerWarp != 32)
    {
        return false;
    }
    if(g.blockMPerWarp == 32 && g.numWarps > 4)
    {
        return false;
    }
    if(g.tileSize <= 0 || g.tileSize % g.blockSize != 0
       || g.tileSize > GFX950_UNIFIED_2D_MAX_TILE_BLOCKS * g.blockSize)
    {
        return false;
    }
    const int64_t threads = g.numWarps * GFX950_UNIFIED_2D_WAVE_LANES;
    if(g.tileSize * g.headSize < threads * 8)
    {
        return false;
    }
    const int64_t perWaveTokens = (GFX950_UNIFIED_2D_WAVE_LANES * 8) / g.headSize;
    if(perWaveTokens > g.blockSize)
    {
        return false;
    }
    if(g.useMfma32x32 && (g.blockMPerWarp != 32 || g.tileSize % 32 != 0 || g.headSize % 16 != 0))
    {
        return false;
    }
    return !g.useTransposedQk32x32 || g.useMfma32x32;
}

/**
 * @brief Binary-search trip count the kernel was compiled with for a baked `num_seqs`.
 *
 * `UnifiedAttention2DTiledSpec.binary_search_iters`: `max(1, ceil(log2(num_seqs + 1)))`,
 * and 32 for a spec that declares no `num_seqs`.
 */
inline int64_t gfx950Unified2DBinarySearchIters(int64_t numSeqs)
{
    constexpr int64_t UNSPECIALIZED_ITERS = 32;
    if(numSeqs <= 0)
    {
        return UNSPECIALIZED_ITERS;
    }
    int64_t iters = 1;
    // Smallest iters with 2^iters >= numSeqs + 1; numSeqs is far below 2^62 in any record
    // the descriptor validator accepts, and the loop is bounded by the shift width anyway.
    while(iters < UNSPECIALIZED_ITERS && (int64_t{1} << iters) < numSeqs + 1)
    {
        ++iters;
    }
    return iters;
}

/**
 * @brief The most sequences one launch of a kernel baked with @p numSeqs can address.
 *
 * The kernel bisects `[0, num_seqs_runtime]` in a fixed number of iterations (it is an
 * unrolled scf.for, not a while loop), which resolves at most `2^iters - 1` sequences. A
 * larger batch leaves the search unfinished: the last blocks read the wrong sequence's
 * `seq_lens` and `query_start_len` and the launch still succeeds. 511 -> 9 iterations -> 511.
 */
inline int64_t gfx950Unified2DBatchCapacity(int64_t numSeqs)
{
    return (int64_t{1} << gfx950Unified2DBinarySearchIters(numSeqs)) - 1;
}

/// The grid and block a candidate must launch with.
struct Gfx950Unified2DLaunchGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx950Unified2DLaunchGeometry& a,
                           const Gfx950Unified2DLaunchGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one candidate over one graph.
 *
 * Mirrors `AttentionTuningSpec.launch_grid` for the tiled 2D path:
 *
 *     block_m = block_m_per_warp * num_warps
 *     block_q = block_m // num_queries_per_kv
 *     qblocks = total_q // block_q + num_seqs
 *     grid    = (num_kv_heads, qblocks, 1)
 *
 * and `launch_block`, `(64 * num_warps, 1, 1)`. `total_q` is `batch * seqLenQ` (the
 * ragged Q rows) and `num_seqs` the RUNTIME batch, not the baked capacity.
 *
 * Throws instead of returning a degenerate grid: an empty launch returns cleanly having
 * written nothing, and prepare() is the last place a named failure is cheap.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx950Unified2DLaunchGeometry gfx950Unified2DLaunchGeometry(int64_t numWarps,
                                                                   int64_t blockMPerWarp,
                                                                   int64_t numQueryHeads,
                                                                   int64_t numKvHeads,
                                                                   int64_t seqLenQ,
                                                                   int64_t batch,
                                                                   const std::string& kernelName)
{
    const auto refuse = [&kernelName](const std::string& why) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "gfx950 attention_unified_2d: kernel '"
                                                           + kernelName + "' " + why);
    };

    if(numWarps != 1 && numWarps != 2 && numWarps != 4 && numWarps != 8)
    {
        refuse("declares num_warps " + std::to_string(numWarps) + ", which gfx950 does not build");
    }
    if(blockMPerWarp != 16 && blockMPerWarp != 32)
    {
        refuse("declares block_m_per_warp " + std::to_string(blockMPerWarp)
               + ", which gfx950 does not build");
    }
    if(numQueryHeads <= 0 || numKvHeads <= 0 || seqLenQ <= 0 || batch <= 0)
    {
        refuse("is asked to launch a non-positive num_query_heads, num_kv_heads, seqlen_q or "
               "batch");
    }
    if(numQueryHeads % numKvHeads != 0
       || numQueryHeads / numKvHeads > GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV)
    {
        refuse("is asked to launch " + std::to_string(numQueryHeads) + " query heads over "
               + std::to_string(numKvHeads)
               + " key/value heads; the kernel needs a whole group of 1 to 16");
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
    // x carries the kv head count, y the Q blocks: both must fit the 32-bit grid dims.
    if(qBlocks > INT32_MAX || numKvHeads > INT32_MAX)
    {
        refuse("is asked to launch a grid wider than 2^31 - 1 blocks");
    }

    Gfx950Unified2DLaunchGeometry geometry;
    geometry.blockX = static_cast<unsigned>(numWarps * GFX950_UNIFIED_2D_WAVE_LANES);
    geometry.gridX = static_cast<unsigned>(numKvHeads);
    geometry.gridY = static_cast<unsigned>(qBlocks);
    geometry.gridZ = 1;
    return geometry;
}

/// Where each paged-KV metadata array sits in the workspace, in bytes.
struct Gfx950Unified2DWorkspaceLayout
{
    int64_t blocksPerSeq = 0;
    std::size_t blockTablesOffset = 0;
    std::size_t blockTablesBytes = 0;
    std::size_t seqLensOffset = 0;
    std::size_t seqLensBytes = 0;
    std::size_t queryStartLenOffset = 0;
    std::size_t queryStartLenBytes = 0;
    std::size_t totalBytes = 0;

    friend bool operator==(const Gfx950Unified2DWorkspaceLayout& a,
                           const Gfx950Unified2DWorkspaceLayout& b)
    {
        return a.blocksPerSeq == b.blocksPerSeq && a.blockTablesOffset == b.blockTablesOffset
               && a.blockTablesBytes == b.blockTablesBytes && a.seqLensOffset == b.seqLensOffset
               && a.seqLensBytes == b.seqLensBytes && a.queryStartLenOffset == b.queryStartLenOffset
               && a.queryStartLenBytes == b.queryStartLenBytes && a.totalBytes == b.totalBytes;
    }
};

namespace detail
{

inline std::size_t alignWorkspaceSection(std::size_t bytes)
{
    constexpr std::size_t A = GFX950_UNIFIED_2D_WORKSPACE_ALIGNMENT;
    return (bytes + A - 1) / A * A;
}

} // namespace detail

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
inline std::optional<Gfx950Unified2DWorkspaceLayout>
    gfx950Unified2DWorkspaceLayout(int64_t batch, int64_t seqLenKv, int64_t blockSize)
{
    if(batch <= 0 || seqLenKv <= 0 || blockSize <= 0 || seqLenKv % blockSize != 0)
    {
        return std::nullopt;
    }

    Gfx950Unified2DWorkspaceLayout layout;
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
 * Padding between sections is zero. nullopt exactly when
 * gfx950Unified2DWorkspaceLayout() is, and when `batch * seqLenQ` does not fit int32.
 */
inline std::optional<std::vector<unsigned char>> gfx950Unified2DWorkspaceImage(int64_t batch,
                                                                               int64_t seqLenQ,
                                                                               int64_t seqLenKv,
                                                                               int64_t blockSize)
{
    const auto layout = gfx950Unified2DWorkspaceLayout(batch, seqLenKv, blockSize);
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

} // namespace hip_kernel_provider::kernel_ingestor_engine
