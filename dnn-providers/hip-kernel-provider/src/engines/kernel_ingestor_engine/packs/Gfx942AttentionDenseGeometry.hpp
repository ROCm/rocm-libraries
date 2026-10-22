// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx942AttentionDenseGeometry.hpp
 * @brief The tile rules, launch geometry and window guard gfx942 attention_dense restates
 *        from its Python builder.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same binary
 * from C++. The engine must recompute what the Python launch path computes identically:
 * nothing compares the two halves, and a mismatch does not fault -- the kernel computes
 * something else.
 *
 * Header-only and dependency-light on purpose: pure functions of a candidate's KMD tile
 * and the graph's dimensions, testable with no HIP context and nothing to mock.
 *
 * The Python is `rocke/library/kernels/gfx942/attention_dense.py`:
 * `supports_attention_dense`, `attention_dense_grid` and `attention_dense_block`.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave64 wave, and the divisor in `num_waves = block_m // 32`.
inline constexpr int64_t GFX942_WAVE_LANES = 64;
inline constexpr int64_t GFX942_ROWS_PER_WAVE = 32;

/// The hardware workgroup limit `block_m // 32 * 64` must stay within.
inline constexpr int64_t GFX942_MAX_WORKGROUP_THREADS = 1024;

/// The tile granularity: `block_m` and `block_n` are positive multiples of 32.
inline constexpr int64_t GFX942_ATTENTION_DENSE_TILE_QUANTUM = 32;

/// Elements one async-DMA instruction moves into LDS: 64 lanes x 4 bytes / 2-byte element
/// (`_DMA_ELEMS_PER_INSTR`). K/V rows per instruction are this over the head size.
inline constexpr int64_t GFX942_DMA_ELEMENTS_PER_INSTRUCTION = 128;

/**
 * @brief Is (blockM, blockN) a tile the gfx942 dense kernel can be built with at
 *        @p headSize?
 *
 * The tile rules `supports_attention_dense` enforces that this engine's launch depends on:
 *   - head_size in {64, 128}                              (_SUPPORTED_HEAD_SIZES)
 *   - block_m a positive multiple of 32, at most a 1024-thread CTA
 *   - block_n a positive multiple of 32                   (spec __post_init__)
 *   - block_m % block_n == 0                              (check_dense_spec_preflight)
 *   - block_n % num_waves == 0 and the rows per wave a multiple of the rows one async
 *     DMA moves (1 at D128, 2 at D64)
 *
 * The LDS budget is NOT restated: it depends on lds_row_pad and on the resolved V pad and
 * conflict-free-V policy, and a tile over it cannot be packed, so no shipped descriptor
 * carries one. This predicate exists so the engine never divides by, or launches with, a
 * tile it does not understand.
 *
 * Total over every int64 triple: the sign and membership tests run before any modulo, so
 * no divisor reaching one is zero or negative.
 */
inline bool isSupportedGfx942AttentionDenseTile(int64_t headSize, int64_t blockM, int64_t blockN)
{
    if(headSize != 64 && headSize != 128)
    {
        return false;
    }
    if(blockM <= 0 || blockM % GFX942_ATTENTION_DENSE_TILE_QUANTUM != 0
       || blockM / GFX942_ROWS_PER_WAVE * GFX942_WAVE_LANES > GFX942_MAX_WORKGROUP_THREADS)
    {
        return false;
    }
    if(blockN <= 0 || blockN % GFX942_ATTENTION_DENSE_TILE_QUANTUM != 0 || blockM % blockN != 0)
    {
        return false;
    }
    const int64_t numWaves = blockM / GFX942_ROWS_PER_WAVE;
    const int64_t rowsPerInstruction = GFX942_DMA_ELEMENTS_PER_INSTRUCTION / headSize;
    return blockN % numWaves == 0 && (blockN / numWaves) % rowsPerInstruction == 0;
}

/**
 * @brief Does every query block of a causal sliding-window launch see at least one key
 *        tile?
 *
 * Mirrors the last guard of `supports_attention_dense`: the last query block's window can
 * start past the final key tile, which gives a zero-trip KV loop, a zero softmax
 * denominator and NaN output. The Python rejects such a spec; this engine declines the
 * candidate, since the guard reads the candidate's own tile.
 *
 *     n_q = seqlen_q // block_m;  n_ktiles = seqlen_kv // block_n
 *     n_per = block_m // block_n; swt = sliding_window // block_n
 *     reject when (n_q - 1) * n_per - swt >= n_ktiles
 *
 * A zero window (no window) always passes. Callers must have validated the tile with
 * isSupportedGfx942AttentionDenseTile and the extents as positive.
 */
inline bool gfx942AttentionDenseWindowReachesKeys(int64_t blockM,
                                                  int64_t blockN,
                                                  int64_t seqLenQ,
                                                  int64_t seqLenKv,
                                                  int64_t slidingWindow)
{
    if(slidingWindow == 0)
    {
        return true;
    }
    const int64_t queryBlocks = seqLenQ / blockM;
    const int64_t keyTiles = seqLenKv / blockN;
    const int64_t tilesPerQueryBlock = blockM / blockN;
    const int64_t windowTiles = slidingWindow / blockN;
    return (queryBlocks - 1) * tilesPerQueryBlock - windowTiles < keyTiles;
}

/// The grid and block a variant must launch with.
struct Gfx942AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx942AttentionDenseGeometry& a,
                           const Gfx942AttentionDenseGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one selected candidate over one graph.
 *
 * Mirrors the non-persistent branch of `attention_dense_grid`:
 *
 *     nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
 *     return (nqb, spec.num_query_heads, spec.batch)
 *
 * and `attention_dense_block`, `(spec.num_waves * 64, 1, 1)`.
 *
 * @p blockM is the selected candidate's own completed `block_m`; @p seqLenQ,
 * @p numQueryHeads and @p batch are the graph's, since the binary takes its shape at
 * runtime and its metadata carries only canonical build inputs.
 *
 * Every shipped variant is non-persistent; the Python's persistent grid,
 * `(spec.num_persistent, 1, 1)`, has no counterpart in this catalog.
 *
 * Throws instead of returning a degenerate grid: an empty or negative launch returns
 * cleanly having written nothing, and prepare() is the last place a named failure is
 * cheap.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx942AttentionDenseGeometry gfx942AttentionDenseGeometry(int64_t blockM,
                                                                 int64_t seqLenQ,
                                                                 int64_t numQueryHeads,
                                                                 int64_t batch,
                                                                 const std::string& kernelName)
{
    if(blockM <= 0 || blockM % GFX942_ATTENTION_DENSE_TILE_QUANTUM != 0
       || blockM / GFX942_ROWS_PER_WAVE * GFX942_WAVE_LANES > GFX942_MAX_WORKGROUP_THREADS)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 attention_dense: kernel '" + kernelName + "' declares block_m "
                + std::to_string(blockM) + ", which gfx942 does not build");
    }

    if(seqLenQ <= 0 || numQueryHeads <= 0 || batch <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 attention_dense: kernel '" + kernelName
                + "' declares a non-positive seqlen_q, num_query_heads or batch");
    }

    Gfx942AttentionDenseGeometry geometry;
    geometry.blockX = static_cast<unsigned>(blockM / GFX942_ROWS_PER_WAVE * GFX942_WAVE_LANES);
    // CEIL, as the Python writes it. Exact for every graph kernel_match serves
    // (Sq % block_m == 0).
    geometry.gridX = static_cast<unsigned>((seqLenQ + blockM - 1) / blockM);
    geometry.gridY = static_cast<unsigned>(numQueryHeads);
    geometry.gridZ = static_cast<unsigned>(batch);
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
