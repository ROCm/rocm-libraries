// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGeometry.hpp"

/**
 * @file Gfx950AttentionUnified2DGeometry.hpp
 * @brief What gfx950 unified-attention 2D restates from its Python builder: the geometry
 *        rules, and the launch grid, batch capacity and paged-KV metadata of
 *        UnifiedAttentionGeometry.hpp bound to the gfx950 traits.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same binary
 * from C++. Nothing compares the two halves, and a mismatch does not fault: the kernel
 * computes something else. Everything here is a pure function of a candidate's KMD fields
 * and the graph's dimensions, so it is testable with no HIP context.
 *
 * The arch-neutral contract (identity block table over a contiguous BSHD K/V, the batch
 * capacity a baked `num_seqs` gives, the grid) lives in UnifiedAttentionGeometry.hpp; this
 * header adds only what gfx950's builder fixes: its traits and its geometry legality.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

inline constexpr const UnifiedAttentionArchTraits& GFX950_UNIFIED_2D_TRAITS
    = GFX950_UNIFIED_ATTENTION_TRAITS;

/// Lanes per wave64 wave: a CTA has `num_warps * 64` threads
/// (`AttentionTuningSpec.launch_block`).
inline constexpr int64_t GFX950_UNIFIED_2D_WAVE_LANES = GFX950_UNIFIED_ATTENTION_TRAITS.waveLanes;

/// The kernel indexes K/V with a 32-bit byte offset when `use_i64_kv_addr` is off, and
/// bounds its buffer loads at this size (`_I32_KV_BUFFER_BYTES`); past it loads return
/// zeros. Every shipped kernel is built i32.
inline constexpr int64_t GFX950_UNIFIED_2D_KV_BUFFER_BYTES
    = GFX950_UNIFIED_ATTENTION_TRAITS.kvBufferBytes;

/// `block_size` values the tiled 2D kernel is built for (`supports_tiled_2d`).
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_SMALL
    = GFX950_UNIFIED_ATTENTION_TRAITS.blockSizes[0];
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_MEDIUM
    = GFX950_UNIFIED_ATTENTION_TRAITS.blockSizes[1];
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_LARGE
    = GFX950_UNIFIED_ATTENTION_TRAITS.blockSizes[2];

/// `1 <= num_queries_per_kv <= 16` (`supports_tiled_2d`).
inline constexpr int64_t GFX950_UNIFIED_2D_MAX_QUERIES_PER_KV
    = GFX950_UNIFIED_ATTENTION_TRAITS.maxQueriesPerKv;

/// The largest KV tile, in pages: the dispatcher's tile policies are 1x, 2x, 4x and 8x.
inline constexpr int64_t GFX950_UNIFIED_2D_MAX_TILE_BLOCKS
    = GFX950_UNIFIED_ATTENTION_TRAITS.maxTileBlocks;

/// Each workspace section starts on this boundary.
inline constexpr std::size_t GFX950_UNIFIED_2D_WORKSPACE_ALIGNMENT
    = UNIFIED_ATTENTION_WORKSPACE_ALIGNMENT;

inline bool isSupportedGfx950Unified2DBlockSize(int64_t blockSize)
{
    return isSupportedUnifiedAttentionBlockSize(GFX950_UNIFIED_ATTENTION_TRAITS, blockSize);
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
    if(!detail::isOneOf(GFX950_UNIFIED_ATTENTION_TRAITS.numWarps, g.numWarps))
    {
        return false;
    }
    if(!detail::isOneOf(GFX950_UNIFIED_ATTENTION_TRAITS.blockMPerWarp, g.blockMPerWarp))
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

/// `unifiedAttentionBinarySearchIters`, named for this pack.
inline int64_t gfx950Unified2DBinarySearchIters(int64_t numSeqs)
{
    return unifiedAttentionBinarySearchIters(numSeqs);
}

/// `unifiedAttentionBatchCapacity`: 511 -> 9 iterations -> 511 sequences.
inline int64_t gfx950Unified2DBatchCapacity(int64_t numSeqs)
{
    return unifiedAttentionBatchCapacity(numSeqs);
}

using Gfx950Unified2DLaunchGeometry = UnifiedAttentionLaunchGeometry;

/// `unifiedAttention2DLaunchGeometry` with the gfx950 traits: kv-major grid
/// `(num_kv_heads, total_q // block_q + batch, 1)`, `64 * num_warps` threads.
inline Gfx950Unified2DLaunchGeometry gfx950Unified2DLaunchGeometry(int64_t numWarps,
                                                                   int64_t blockMPerWarp,
                                                                   int64_t numQueryHeads,
                                                                   int64_t numKvHeads,
                                                                   int64_t seqLenQ,
                                                                   int64_t batch,
                                                                   const std::string& kernelName)
{
    return unifiedAttention2DLaunchGeometry(GFX950_UNIFIED_ATTENTION_TRAITS,
                                            numWarps,
                                            blockMPerWarp,
                                            numQueryHeads,
                                            numKvHeads,
                                            seqLenQ,
                                            batch,
                                            kernelName);
}

using Gfx950Unified2DWorkspaceLayout = UnifiedAttentionWorkspaceLayout;

/// `unifiedAttentionWorkspaceLayout`, named for this pack.
inline std::optional<Gfx950Unified2DWorkspaceLayout>
    gfx950Unified2DWorkspaceLayout(int64_t batch, int64_t seqLenKv, int64_t blockSize)
{
    return unifiedAttentionWorkspaceLayout(batch, seqLenKv, blockSize);
}

/// `unifiedAttentionWorkspaceImage`, named for this pack.
inline std::optional<std::vector<unsigned char>> gfx950Unified2DWorkspaceImage(int64_t batch,
                                                                               int64_t seqLenQ,
                                                                               int64_t seqLenKv,
                                                                               int64_t blockSize)
{
    return unifiedAttentionWorkspaceImage(batch, seqLenQ, seqLenKv, blockSize);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
