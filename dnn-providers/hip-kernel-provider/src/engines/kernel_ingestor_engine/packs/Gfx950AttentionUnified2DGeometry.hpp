// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

/**
 * @file Gfx950AttentionUnified2DGeometry.hpp
 * @brief The launch geometry, derived workspace layout and contents, and binary-search
 *        capacity bound for hipkernel:Gfx950AttentionUnified2D.
 *
 * Every quantity here restates one the rocKE builder computes in Python. The Python side
 * is authoritative; this header is the mirror a C++ dispatch has to carry because the
 * runtime never imports rocKE. The profile's `launch_surface:` block links each function
 * to its Python source and its test
 * (projects/hipdnn/tools/IngestorGenerator/configs/gfx950_attention_unified_2d.profile.yaml).
 *
 * Sources mirrored, all under dnn-providers/hip-kernel-provider/rocke:
 *   - grid / block  : library/dispatch/attention/tuning_specs.py
 *                     AttentionTuningSpec.launch_grid / .launch_block, the `2d` path
 *   - block_m, block_q : library/kernels/gfx950/attention_tiled_2d.py:786,808
 *   - trip count    : library/kernels/gfx950/attention_tiled_2d.py:827
 *                     UnifiedAttention2DTiledSpec.binary_search_iters
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave on gfx950. One warp is one wave64, and the CTA is num_warps of them.
inline constexpr int64_t GFX950_UNIFIED_2D_WAVE_LANES = 64;

/// The alignment every workspace section starts on. The kernel reads each of the three
/// arrays through its own base pointer, so they only have to be individually aligned.
inline constexpr int64_t GFX950_UNIFIED_2D_WORKSPACE_ALIGNMENT = 256;

/// Trip count the kernel falls back to when a spec names no batch (`num_seqs <= 0`).
/// Mirrors the Python fallback, which calls 32 "conservative": it is wide enough for any
/// batch a 32-bit num_seqs can express.
inline constexpr int64_t GFX950_UNIFIED_2D_FALLBACK_SEARCH_ITERS = 32;

/// Head sizes the unified 2D family builds for. Measured against the candidate
/// capability, which allows all three under one entry -- which is why D256 is served by
/// this engine rather than by a separate one.
inline constexpr int64_t GFX950_UNIFIED_2D_HEAD_SIZE_SMALL = 64;
inline constexpr int64_t GFX950_UNIFIED_2D_HEAD_SIZE_MEDIUM = 128;
inline constexpr int64_t GFX950_UNIFIED_2D_HEAD_SIZE_LARGE = 256;

/// KV page sizes the family builds for.
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_MIN = 16;
inline constexpr int64_t GFX950_UNIFIED_2D_BLOCK_SIZE_MAX = 64;

/// @brief True when @p headSize is one this engine's kernels are built for.
inline bool isSupportedGfx950AttentionUnified2DHeadSize(int64_t headSize)
{
    return headSize == GFX950_UNIFIED_2D_HEAD_SIZE_SMALL
           || headSize == GFX950_UNIFIED_2D_HEAD_SIZE_MEDIUM
           || headSize == GFX950_UNIFIED_2D_HEAD_SIZE_LARGE;
}

/// @brief True when @p blockSize is a KV page size the family builds for: a power of two
///        in [16, 64].
inline bool isSupportedGfx950AttentionUnified2DBlockSize(int64_t blockSize)
{
    if(blockSize < GFX950_UNIFIED_2D_BLOCK_SIZE_MIN || blockSize > GFX950_UNIFIED_2D_BLOCK_SIZE_MAX)
    {
        return false;
    }
    return (blockSize & (blockSize - 1)) == 0;
}

/**
 * @brief The fixed trip count the kernel bakes for its sequence binary search.
 *
 * Mirrors `UnifiedAttention2DTiledSpec.binary_search_iters`:
 * `ceil(log2(num_seqs + 1))`, floored at 1, with 32 when the spec names no batch.
 *
 * Computed with an integer doubling loop rather than std::log2: a double log of 2^k is
 * allowed to land a hair below k, and ceil() would then return k instead of k+1 and
 * silently under-count the iterations -- the one direction that is not safe.
 */
inline int64_t gfx950AttentionUnified2DSearchIterations(int64_t numSeqs)
{
    if(numSeqs <= 0)
    {
        return GFX950_UNIFIED_2D_FALLBACK_SEARCH_ITERS;
    }
    // Smallest i with 2^i >= numSeqs + 1.
    int64_t iterations = 0;
    int64_t reach = 1;
    while(reach < numSeqs + 1)
    {
        reach *= 2;
        ++iterations;
    }
    return iterations < 1 ? 1 : iterations;
}

/**
 * @brief The largest runtime batch a kernel baked at @p numSeqs still resolves correctly.
 *
 * THIS IS A CORRECTNESS BOUND, NOT A PERFORMANCE HINT. The kernel finds a Q block's
 * sequence with a fixed-trip binary search whose `right` bound is the RUNTIME num_seqs
 * kernarg but whose trip count is baked. The loop carries `done = left >= right` and
 * yields `select(done, left, ...)`, so it is idempotent once converged
 * (rocke/platform/python/rocke/helpers/attention.py:685). Therefore:
 *
 *   - more iterations than the batch needs  -> harmless, the loop has already converged
 *   - fewer                                 -> the search has NOT converged and returns
 *                                              the wrong sequence index, with nothing
 *                                              faulting and every output wrong
 *
 * Converging a search over `numSeqs + 1` candidates takes ceil(log2(numSeqs + 1))
 * iterations, so a kernel baked with `i` iterations serves any batch up to 2^i - 1.
 */
inline int64_t gfx950AttentionUnified2DMaxBatch(int64_t numSeqs)
{
    const int64_t iterations = gfx950AttentionUnified2DSearchIterations(numSeqs);
    // Saturate rather than shift past the width of the type: the fallback's 32
    // iterations already exceed any batch a 32-bit kernarg can carry.
    constexpr int64_t SATURATION_ITERS = 62;
    if(iterations >= SATURATION_ITERS)
    {
        return INT64_MAX;
    }
    return (int64_t{1} << iterations) - 1;
}

/// The launch geometry for one prepared dispatch.
struct Gfx950AttentionUnified2DGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx950AttentionUnified2DGeometry& a,
                           const Gfx950AttentionUnified2DGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief Grid and block for a unified 2D launch, mirroring `launch_grid`/`launch_block`.
 *
 * Python, for the `2d` path:
 *
 *     block_m = block_m_per_warp * num_warps
 *     block_q = block_m // nqk   if nqk <= block_m   else 1
 *     qblocks = total_q // block_q + num_seqs
 *     grid    = (num_kv_heads, qblocks, 1)
 *     block   = (64 * num_warps, 1, 1)
 *
 * `qblocks` deliberately uses FLOOR division and then adds one block per sequence, which
 * over-counts rather than under-counts when a sequence does not fill its last Q block.
 * The surplus blocks exit at the kernel's own `qb_start_pos >= cur_batch_q_len` guard
 * (attention_tiled_2d.py:1265). Rounding this to a tight ceil would drop the final
 * partial block of every sequence but the last.
 *
 * @param blockMPerWarp Rows of Q per warp, from the candidate's metadata.
 * @param numWarps      Warps per CTA, from the candidate's metadata.
 * @param numQueryHeads The graph's query heads.
 * @param numKvHeads    The graph's KV heads; must divide @p numQueryHeads.
 * @param seqLenQ       Query tokens per sequence.
 * @param batch         Sequences in the graph; the kernel's `num_seqs` at runtime.
 * @param kernelName    Named in any thrown message so a bad candidate is identifiable.
 */
inline Gfx950AttentionUnified2DGeometry
    gfx950AttentionUnified2DGeometry(int64_t blockMPerWarp,
                                     int64_t numWarps,
                                     int64_t numQueryHeads,
                                     int64_t numKvHeads,
                                     int64_t seqLenQ,
                                     int64_t batch,
                                     const std::string& kernelName)
{
    if(blockMPerWarp <= 0 || numWarps <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + kernelName
                + "' declares a non-positive block_m_per_warp or num_warps");
    }
    if(seqLenQ <= 0 || numQueryHeads <= 0 || numKvHeads <= 0 || batch <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + kernelName
                + "' declares a non-positive seqlen_q, num_query_heads, num_kv_heads or "
                  "batch");
    }
    if(numQueryHeads % numKvHeads != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + kernelName
                + "' was given num_query_heads that num_kv_heads does not divide");
    }

    const int64_t numQueriesPerKv = numQueryHeads / numKvHeads;
    const int64_t blockM = blockMPerWarp * numWarps;
    const int64_t blockQ = (numQueriesPerKv <= blockM) ? (blockM / numQueriesPerKv) : 1;
    const int64_t totalQ = batch * seqLenQ;
    const int64_t qBlocks = totalQ / blockQ + batch;

    Gfx950AttentionUnified2DGeometry geometry;
    geometry.gridX = static_cast<unsigned>(numKvHeads);
    geometry.gridY = static_cast<unsigned>(qBlocks);
    geometry.gridZ = 1U;
    geometry.blockX = static_cast<unsigned>(GFX950_UNIFIED_2D_WAVE_LANES * numWarps);
    return geometry;
}

/**
 * @brief Byte offsets and total size of the synthesized paged-attention workspace.
 *
 * The hipDNN graph is dense BSHD SDPA; the kernel's ABI is the paged/varlen one. The gap
 * is three int32 arrays the provider derives from the graph's own shape, with no new
 * graph inputs:
 *
 *   block_tables[batch][blocksPerSeq] -- the identity table. Sequence b's page j is the
 *                                        physical page b * blocksPerSeq + j, which is
 *                                        exactly what a dense contiguous K/V already is.
 *   seq_lens[batch]                   -- every sequence is seqLenKv long.
 *   query_start_len[batch + 1]        -- the cumulative query count, b * seqLenQ.
 *
 * Contents depend only on (batch, seqLenQ, seqLenKv, blockSize), all known at prepare(),
 * so the handler stages them host-side and issues one copy.
 *
 * The identity table is only valid because K and V are contiguous: graph_match declines a
 * non-contiguous K or V for precisely this reason.
 */
struct Gfx950AttentionUnified2DWorkspace
{
    int64_t blockTablesOffset = 0;
    int64_t blockTablesBytes = 0;
    int64_t seqLensOffset = 0;
    int64_t seqLensBytes = 0;
    int64_t queryStartLenOffset = 0;
    int64_t queryStartLenBytes = 0;
    int64_t totalBytes = 0;
    /// Pages per sequence, and the block_tables row stride the kernel is handed.
    int64_t blocksPerSequence = 0;

    friend bool operator==(const Gfx950AttentionUnified2DWorkspace& a,
                           const Gfx950AttentionUnified2DWorkspace& b)
    {
        return a.blockTablesOffset == b.blockTablesOffset
               && a.blockTablesBytes == b.blockTablesBytes && a.seqLensOffset == b.seqLensOffset
               && a.seqLensBytes == b.seqLensBytes && a.queryStartLenOffset == b.queryStartLenOffset
               && a.queryStartLenBytes == b.queryStartLenBytes && a.totalBytes == b.totalBytes
               && a.blocksPerSequence == b.blocksPerSequence;
    }
};

/// @brief Round @p value up to the next multiple of @p alignment.
inline int64_t gfx950AttentionUnified2DAlignUp(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

/**
 * @brief Lay out the derived workspace for one (batch, seqLenQ, seqLenKv, blockSize).
 *
 * @note @p seqLenKv must be a whole number of pages. A partial trailing page would make
 *       the identity table address tokens past the end of K/V, so graph_match declines
 *       `seqLenKv % blockSize != 0` and this function treats it as a programming error.
 */
inline Gfx950AttentionUnified2DWorkspace
    gfx950AttentionUnified2DWorkspace(int64_t batch,
                                      int64_t seqLenQ,
                                      int64_t seqLenKv,
                                      int64_t blockSize,
                                      const std::string& kernelName)
{
    if(batch <= 0 || seqLenQ <= 0 || seqLenKv <= 0 || blockSize <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + kernelName
                + "' was given a non-positive batch, seqlen_q, seqlen_kv or block_size "
                  "to size its workspace for");
    }
    if(seqLenKv % blockSize != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_unified_2d: kernel '" + kernelName
                + "' was given a seqlen_kv that block_size does not divide, so the "
                  "identity block table would address tokens past the end of K/V");
    }

    constexpr int64_t INT32_BYTES = 4;
    constexpr int64_t ALIGN = GFX950_UNIFIED_2D_WORKSPACE_ALIGNMENT;

    Gfx950AttentionUnified2DWorkspace workspace;
    workspace.blocksPerSequence = seqLenKv / blockSize;

    workspace.blockTablesOffset = 0;
    workspace.blockTablesBytes = batch * workspace.blocksPerSequence * INT32_BYTES;

    workspace.seqLensOffset = gfx950AttentionUnified2DAlignUp(
        workspace.blockTablesOffset + workspace.blockTablesBytes, ALIGN);
    workspace.seqLensBytes = batch * INT32_BYTES;

    workspace.queryStartLenOffset
        = gfx950AttentionUnified2DAlignUp(workspace.seqLensOffset + workspace.seqLensBytes, ALIGN);
    workspace.queryStartLenBytes = (batch + 1) * INT32_BYTES;

    workspace.totalBytes = gfx950AttentionUnified2DAlignUp(
        workspace.queryStartLenOffset + workspace.queryStartLenBytes, ALIGN);

    // seqLenQ is not a term in any size above; it is taken so the signature states the
    // full set of inputs the CONTENTS depend on, which the handler fills from the same
    // four values.
    (void)seqLenQ;
    return workspace;
}

/**
 * @brief The workspace contents for one dispatch, as int32 words over the whole workspace.
 *
 * block_tables[b][j] = b * blocksPerSequence + j   -- the identity table. This is what a
 *                                                     contiguous dense K/V already is,
 *                                                     which is why graph_match insists on
 *                                                     BSHD K and V.
 * seq_lens[b]        = seqLenKv                    -- every sequence is the same length;
 *                                                     varlen graphs were declined.
 * query_start_len[b] = b * seqLenQ                 -- the cumulative query count, and
 *                                                     [batch] = batch * seqLenQ closes it.
 *
 * Each section starts at the byte offset @p workspace gives it. Padding words between
 * sections are zero; the kernel never reads them.
 *
 * @param workspace The layout gfx950AttentionUnified2DWorkspace returned for the same
 *                  @p batch, @p seqLenQ and @p seqLenKv.
 */
inline std::vector<int32_t>
    gfx950AttentionUnified2DStageWorkspace(int64_t batch,
                                           int64_t seqLenQ,
                                           int64_t seqLenKv,
                                           const Gfx950AttentionUnified2DWorkspace& workspace)
{
    constexpr int64_t INT32_BYTES = 4;
    std::vector<int32_t> staged(static_cast<size_t>(workspace.totalBytes / INT32_BYTES), 0);

    const auto wordAt = [&staged](int64_t byteOffset, int64_t index) -> int32_t& {
        return staged[static_cast<size_t>(byteOffset / INT32_BYTES + index)];
    };

    for(int64_t b = 0; b < batch; ++b)
    {
        for(int64_t j = 0; j < workspace.blocksPerSequence; ++j)
        {
            wordAt(workspace.blockTablesOffset, b * workspace.blocksPerSequence + j)
                = static_cast<int32_t>(b * workspace.blocksPerSequence + j);
        }
        wordAt(workspace.seqLensOffset, b) = static_cast<int32_t>(seqLenKv);
        wordAt(workspace.queryStartLenOffset, b) = static_cast<int32_t>(b * seqLenQ);
    }
    wordAt(workspace.queryStartLenOffset, batch) = static_cast<int32_t>(batch * seqLenQ);
    return staged;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
