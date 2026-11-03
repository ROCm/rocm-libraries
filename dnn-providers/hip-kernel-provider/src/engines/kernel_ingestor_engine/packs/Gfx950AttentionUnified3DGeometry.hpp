// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_plugin_sdk/PluginException.hpp>

#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionDenseGeometry.hpp"
#include "engines/kernel_ingestor_engine/packs/Gfx950AttentionUnified2DGeometry.hpp"

/**
 * @file Gfx950AttentionUnified3DGeometry.hpp
 * @brief What gfx950 split-KV unified attention restates from its Python builders: the
 *        geometry rules, the two launch grids, and the workspace the two kernels share.
 *
 * The engine launches two code objects per call. The segment kernel
 * (`build_unified_attention_3d_tiled`) splits each sequence's keys into `num_segments`
 * ranges and writes one partial softmax state per (token, query head, segment) into
 * workspace scratch; the reduce kernel (`build_unified_attention_reduce_tiled`) folds those
 * into the output. Both read the identity paged-KV metadata Unified2D derives for a
 * contiguous K/V graph, laid out identically here, so the two engines cannot disagree on it.
 *
 * Every shipped kernel is built with the decode grid (one CTA per sequence along x, no
 * `query_start_len` search), which is legal only when every sequence has one query token.
 * Everything here is a pure function of a candidate's KMD fields and the graph's dimensions.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// One wave64 CTA for both kernels (`SEGMENT_THREADS`; the reduce builder's `THREADS`).
inline constexpr unsigned GFX950_UNIFIED_3D_CTA_THREADS
    = static_cast<unsigned>(GFX950_UNIFIED_ATTENTION_TRAITS.waveLanes);

/// The segment kernel fixes BLOCK_M at 16 query rows, so a GQA group must divide 16
/// (`supports_tiled_3d`).
inline constexpr int64_t GFX950_UNIFIED_3D_BLOCK_M = 16;

/**
 * @brief Whether the segment kernel serves `numQueryHeads` query heads over `numKvHeads`
 * key/value heads: a whole GQA group that divides GFX950_UNIFIED_3D_BLOCK_M.
 *
 * The one statement of Unified3D's GQA rule: its graph match, geometry and launch check it,
 * and Unified2D hands a decode graph to Unified3D exactly when it holds. Total over every
 * int64 input.
 */
inline bool gfx950Unified3DServesGqaGroup(int64_t numQueryHeads, int64_t numKvHeads)
{
    if(numQueryHeads <= 0 || numKvHeads <= 0 || numQueryHeads % numKvHeads != 0)
    {
        return false;
    }
    return GFX950_UNIFIED_3D_BLOCK_M % (numQueryHeads / numKvHeads) == 0;
}

/// The reduce kernel's LDS per segment: one f32 factor (`split_kv_reduce_lds_bytes`, held
/// to the compiled code object's group_segment_fixed_size by
/// test_attention_tiled_3d_reduce_lds).
inline constexpr int64_t GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT = 4;

/// The most segments whose factor cache fits gfx950's LDS (`reduce_max_segments`): the
/// bound `supports_tiled_reduce_spec` gates with ArchTarget.fits_lds. The Python size rounds
/// the pool to 16 bytes, which does not change the bound while the capacity is a multiple
/// of 16.
inline constexpr int64_t GFX950_UNIFIED_3D_MAX_SEGMENTS
    = GFX950_LDS_CAPACITY_BYTES / GFX950_UNIFIED_3D_REDUCE_LDS_BYTES_PER_SEGMENT;

/// Bytes of one f32 element of the segment scratch.
inline constexpr std::size_t GFX950_UNIFIED_3D_SCRATCH_ELEMENT = UNIFIED_ATTENTION_SCRATCH_ELEMENT;

/**
 * @brief Whether gfx950 builds the segment and reduce kernels for these fields.
 *
 * `supports_tiled_3d` and `supports_tiled_reduce_spec`, conjoined:
 *   - head_size in {64, 128, 256} (the reduce kernel also needs it divisible by 64)
 *   - block_size in {16, 32, 64}
 *   - 1 <= num_queries_per_kv <= 16 and dividing 16
 *   - 1 <= num_segments <= GFX950_UNIFIED_3D_MAX_SEGMENTS (gfx950 LDS / 4 bytes)
 *
 * Total over every int64 input.
 */
inline bool isSupportedGfx950Unified3DGeometry(int64_t headSize,
                                               int64_t blockSize,
                                               int64_t numQueryHeads,
                                               int64_t numKvHeads,
                                               int64_t numSegments)
{
    if(headSize != 64 && headSize != 128 && headSize != 256)
    {
        return false;
    }
    if(!isSupportedGfx950Unified2DBlockSize(blockSize))
    {
        return false;
    }
    if(!gfx950Unified3DServesGqaGroup(numQueryHeads, numKvHeads))
    {
        return false;
    }
    return numSegments >= 1 && numSegments <= GFX950_UNIFIED_3D_MAX_SEGMENTS;
}

/// The grids and CTA both kernels launch with.
struct Gfx950Unified3DLaunchGeometry
{
    unsigned segmentGridX = 0;
    unsigned segmentGridY = 0;
    unsigned segmentGridZ = 0;
    unsigned reduceGridX = 0;
    unsigned reduceGridY = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx950Unified3DLaunchGeometry& a,
                           const Gfx950Unified3DLaunchGeometry& b)
    {
        return a.segmentGridX == b.segmentGridX && a.segmentGridY == b.segmentGridY
               && a.segmentGridZ == b.segmentGridZ && a.reduceGridX == b.reduceGridX
               && a.reduceGridY == b.reduceGridY && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one decode-grid candidate over one graph.
 *
 * Mirrors the Python launcher (`kernels/common/attention_unified.py`, the 3D pipeline):
 *
 *     segment grid = (num_q_blocks, num_kv_heads, num_segments), num_q_blocks = num_seqs on
 *                    the decode grid
 *     reduce grid  = (total_q, num_query_heads, 1)
 *     CTA          = (64, 1, 1) for both
 *
 * With one query token per sequence, `num_seqs` and `total_q` are both the batch.
 *
 * Throws instead of returning a degenerate grid: an empty launch returns cleanly having
 * written nothing.
 */
inline Gfx950Unified3DLaunchGeometry gfx950Unified3DLaunchGeometry(int64_t numQueryHeads,
                                                                   int64_t numKvHeads,
                                                                   int64_t numSegments,
                                                                   int64_t batch,
                                                                   const std::string& kernelName)
{
    const auto refuse = [&kernelName](const std::string& why) {
        throw hipdnn_plugin_sdk::HipdnnPluginException(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                       "gfx950 attention_unified_3d: kernel '"
                                                           + kernelName + "' " + why);
    };

    if(numQueryHeads <= 0 || numKvHeads <= 0 || batch <= 0)
    {
        refuse("is asked to launch a non-positive num_query_heads, num_kv_heads or batch");
    }
    if(numSegments < 1 || numSegments > GFX950_UNIFIED_3D_MAX_SEGMENTS)
    {
        refuse("declares num_segments " + std::to_string(numSegments)
               + ", which gfx950 does not build");
    }
    if(!gfx950Unified3DServesGqaGroup(numQueryHeads, numKvHeads))
    {
        refuse("is asked to launch " + std::to_string(numQueryHeads) + " query heads over "
               + std::to_string(numKvHeads)
               + " key/value heads; the kernel needs a group dividing 16");
    }
    if(batch > INT32_MAX || numQueryHeads > INT32_MAX)
    {
        refuse("is asked to launch a grid wider than 2^31 - 1 blocks");
    }

    Gfx950Unified3DLaunchGeometry geometry;
    geometry.segmentGridX = static_cast<unsigned>(batch);
    geometry.segmentGridY = static_cast<unsigned>(numKvHeads);
    geometry.segmentGridZ = static_cast<unsigned>(numSegments);
    geometry.reduceGridX = static_cast<unsigned>(batch);
    geometry.reduceGridY = static_cast<unsigned>(numQueryHeads);
    geometry.blockX = GFX950_UNIFIED_3D_CTA_THREADS;
    return geometry;
}

using Gfx950Unified3DWorkspaceLayout = UnifiedAttention3DWorkspaceLayout;

/// `unifiedAttention3DWorkspaceLayout`, named for this pack.
inline std::optional<Gfx950Unified3DWorkspaceLayout>
    gfx950Unified3DWorkspaceLayout(int64_t batch,
                                   int64_t seqLenQ,
                                   int64_t seqLenKv,
                                   int64_t blockSize,
                                   int64_t numQueryHeads,
                                   int64_t headSize,
                                   int64_t numSegments)
{
    return unifiedAttention3DWorkspaceLayout(
        batch, seqLenQ, seqLenKv, blockSize, numQueryHeads, headSize, numSegments);
}

/**
 * @brief The bytes launch() copies into the head of the workspace: the identity paged-KV
 *        metadata, exactly as gfx950Unified2DWorkspaceImage() builds it.
 *
 * Its size is `layout.identity.totalBytes`, which is `layout.segmOutputOffset`: the scratch
 * after it is not copied.
 */
inline std::optional<std::vector<unsigned char>> gfx950Unified3DWorkspaceImage(int64_t batch,
                                                                               int64_t seqLenQ,
                                                                               int64_t seqLenKv,
                                                                               int64_t blockSize)
{
    return gfx950Unified2DWorkspaceImage(batch, seqLenQ, seqLenKv, blockSize);
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
