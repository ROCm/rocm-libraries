// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "engines/kernel_ingestor_engine/packs/UnifiedAttentionGeometry.hpp"

/**
 * @file Gfx942AttentionUnified2DGeometry.hpp
 * @brief What gfx942 unified-attention 2D restates from its Python builder: the geometry
 *        rules, and the launch grid and paged-KV metadata of UnifiedAttentionGeometry.hpp
 *        bound to the gfx942 traits.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same binary
 * from C++. Nothing compares the two halves, and a mismatch does not fault: the kernel
 * computes something else. Everything here is a pure function of a candidate's KMD fields
 * and the graph's dimensions, so it is testable with no HIP context.
 *
 * The arch-neutral contract (identity block table over a contiguous BSHD K/V, the grid)
 * lives in UnifiedAttentionGeometry.hpp; this header adds only what gfx942's builder fixes:
 * its traits and its geometry legality.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave64 wave: a CTA has `num_warps * 64` threads
/// (`AttentionTuningSpec.launch_block`).
inline constexpr int64_t GFX942_UNIFIED_2D_WAVE_LANES = GFX942_UNIFIED_ATTENTION_TRAITS.waveLanes;

/// `1 <= num_queries_per_kv <= 16` (`supports_tiled_2d`).
inline constexpr int64_t GFX942_UNIFIED_2D_MAX_QUERIES_PER_KV
    = GFX942_UNIFIED_ATTENTION_TRAITS.maxQueriesPerKv;

/// The GQA groups (query heads per key/value head) the catalog ships kernels for, one per
/// (dtype, head size, group) cell. The kernel runs any group up to the maximum above, so this
/// is the catalog's limit, not the kernel's: graph_match declines every other group by name,
/// and the catalog census holds this list to the groups the shipped kernels carry.
inline constexpr std::array<int64_t, 5> GFX942_UNIFIED_2D_SHIPPED_RATIOS{1, 2, 4, 8, 16};

/// The largest KV tile, in pages: the dispatcher's tile policies are 1x, 2x, 4x and 8x.
inline constexpr int64_t GFX942_UNIFIED_2D_MAX_TILE_BLOCKS
    = GFX942_UNIFIED_ATTENTION_TRAITS.maxTileBlocks;

/// fp16/bf16 halves one lane moves per async K/V copy: CDNA3 caps load-to-LDS at one dword.
inline constexpr int64_t GFX942_UNIFIED_2D_DMA_HALVES_PER_LANE = 2;

inline bool isSupportedGfx942Unified2DBlockSize(int64_t blockSize)
{
    return isSupportedUnifiedAttentionBlockSize(GFX942_UNIFIED_ATTENTION_TRAITS, blockSize);
}

/**
 * @brief The geometry fields a candidate carries, as the Python spec validates them.
 *
 * `UnifiedAttention2DTiledSpec.__post_init__` and `supports_tiled_2d` (gfx942), conjoined:
 *   - head_size in {64, 128, 192, 256}; the engine ships 64 and 128 (@p headSize is
 *     range-checked by the matcher, not here)
 *   - block_size in {16, 32, 64}
 *   - num_warps in {1, 2, 4, 8}; block_m_per_warp in {16, 32}. 8 warps x 32 rows is legal:
 *     the builder then needs direct-Q and conflict-free V stores, which the catalog's spec
 *     carries and the KMD does not
 *   - tile_size a positive multiple of block_size, at most 8 * block_size (the dispatcher's
 *     1x/2x/4x/8x tile policies), with `tile_size * head_size >= threads * 2` (one async
 *     copy of two halves per lane must fit the tile) and `128 / head_size <= block_size`
 *     (a wave must not straddle blocks)
 *   - use_mfma_32x32x8 needs block_m_per_warp 32 and tile_size % 32 == 0;
 *     use_transposed_qk_32x32 needs use_mfma_32x32x8
 *
 * The LDS budget is NOT restated: hkp_pack compiles every shipped kernel, and rocKE's
 * `supports_tiled_2d` refuses an over-budget spec before it can ship.
 *
 * Total over every int64 input: each divisor is checked positive before it is used.
 */
struct Gfx942Unified2DGeometryFields
{
    int64_t headSize = 0;
    int64_t blockSize = 0;
    int64_t tileSize = 0;
    int64_t numWarps = 0;
    int64_t blockMPerWarp = 0;
    bool useMfma32x32x8 = false;
    bool useTransposedQk32x32 = false;
};

inline bool isSupportedGfx942Unified2DGeometry(const Gfx942Unified2DGeometryFields& g)
{
    if(g.headSize != 64 && g.headSize != 128 && g.headSize != 192 && g.headSize != 256)
    {
        return false;
    }
    if(!isSupportedGfx942Unified2DBlockSize(g.blockSize))
    {
        return false;
    }
    if(!detail::isOneOf(GFX942_UNIFIED_ATTENTION_TRAITS.numWarps, g.numWarps))
    {
        return false;
    }
    if(!detail::isOneOf(GFX942_UNIFIED_ATTENTION_TRAITS.blockMPerWarp, g.blockMPerWarp))
    {
        return false;
    }
    if(g.tileSize <= 0 || g.tileSize % g.blockSize != 0
       || g.tileSize > GFX942_UNIFIED_2D_MAX_TILE_BLOCKS * g.blockSize)
    {
        return false;
    }
    const int64_t threads = g.numWarps * GFX942_UNIFIED_2D_WAVE_LANES;
    if(g.tileSize * g.headSize < threads * GFX942_UNIFIED_2D_DMA_HALVES_PER_LANE)
    {
        return false;
    }
    const int64_t perWaveTokens
        = (GFX942_UNIFIED_2D_WAVE_LANES * GFX942_UNIFIED_2D_DMA_HALVES_PER_LANE) / g.headSize;
    if(perWaveTokens > g.blockSize)
    {
        return false;
    }
    if(g.useMfma32x32x8 && (g.blockMPerWarp != 32 || g.tileSize % 32 != 0))
    {
        return false;
    }
    return !g.useTransposedQk32x32 || g.useMfma32x32x8;
}

using Gfx942Unified2DLaunchGeometry = UnifiedAttentionLaunchGeometry;

/// `unifiedAttention2DLaunchGeometry` with the gfx942 traits: kv-major grid
/// `(num_kv_heads, total_q // block_q + batch, 1)`, `64 * num_warps` threads.
inline Gfx942Unified2DLaunchGeometry gfx942Unified2DLaunchGeometry(int64_t numWarps,
                                                                   int64_t blockMPerWarp,
                                                                   int64_t numQueryHeads,
                                                                   int64_t numKvHeads,
                                                                   int64_t seqLenQ,
                                                                   int64_t batch,
                                                                   const std::string& kernelName)
{
    return unifiedAttention2DLaunchGeometry(GFX942_UNIFIED_ATTENTION_TRAITS,
                                            numWarps,
                                            blockMPerWarp,
                                            numQueryHeads,
                                            numKvHeads,
                                            seqLenQ,
                                            batch,
                                            kernelName);
}

using Gfx942Unified2DWorkspaceLayout = UnifiedAttentionWorkspaceLayout;

/// `unifiedAttentionWorkspaceLayout`, named for this pack.
inline std::optional<Gfx942Unified2DWorkspaceLayout>
    gfx942Unified2DWorkspaceLayout(int64_t batch, int64_t seqLenKv, int64_t blockSize)
{
    return unifiedAttentionWorkspaceLayout(batch, seqLenKv, blockSize);
}

/// `unifiedAttentionWorkspaceImage`, named for this pack.
inline std::optional<std::vector<unsigned char>> gfx942Unified2DWorkspaceImage(int64_t batch,
                                                                               int64_t seqLenQ,
                                                                               int64_t seqLenKv,
                                                                               int64_t blockSize)
{
    return unifiedAttentionWorkspaceImage(batch, seqLenQ, seqLenKv, blockSize);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
