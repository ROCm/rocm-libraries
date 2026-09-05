// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx950AttentionTiledGeometry.hpp
 * @brief The launch geometry gfx950 2D tiled attention restates from its Python
 *        dispatcher, plus the paged-KV geometry the matcher derives from the graph.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same binary
 * from C++. Everything the Python launch path computes, the engine must recompute
 * IDENTICALLY, and nothing in the build, the packer, the validator or the test suite
 * compares the two halves. The kernel does not fail on a mismatch -- it computes
 * something else, so the whole class is found by differential testing or not at all.
 *
 * Header-only and dependency-light on purpose: pure functions of KMD metadata and graph
 * dims, testable on any machine, with no HIP context and nothing to mock.
 *
 * THE DENSE TWIN IS A REFERENCE, NOT A TEMPLATE. Three differences, each verified
 * against source rather than assumed from the sibling:
 *
 *  1. **The grid carries a `+ num_seqs` VARLEN SLACK TERM, which dense has no analogue
 *     for.** `_get_2d_launch_meta` (kernels/common/attention_unified.py:4079-4102)
 *     computes `total_num_q_blocks = total_q // block_q + num_seqs`: one reserved
 *     padding q-block PER SEQUENCE, so a ragged batch needs no exact-division block
 *     count. Drop the term and the kernel UNDER-LAUNCHES, leaving tail blocks
 *     unwritten. The harness NaN-sentinel-fills outputs precisely so that shows up --
 *     as `allClose=false` with ZERO finite mismatches, which reads like a tolerance
 *     problem and is not one.
 *
 *  2. **grid.x is num_kv_heads, not num_query_heads.** The CTA owns one KV head and
 *     the `num_queries_per_kv` query heads that map to it (GQA is folded into
 *     `block_q`, below). Copying dense's `numQueryHeads` would over-launch by the GQA
 *     ratio -- up to 16x the CTAs, every extra one indexing past the KV cache.
 *
 *  3. **The CTA size is `wave_size * num_warps`, and num_warps VARIES per shape**
 *     (measured over {1,2,4} across the 52 dispatcher-resolved shapes). Dense derives
 *     its CTA from `block_m // 32`; here `block_m` is an OUTPUT of num_warps rather
 *     than an input. `wave_size` is 64 on gfx950 -- the 32-lane case is gfx1250 only,
 *     which this engine does not ship, so it is asserted rather than parameterised.
 *
 * And the whole reason this header has a SECOND function, which dense does not need:
 * the tiled kernel is structurally PAGED. `block_size` is the KV cache's page size, it
 * is required with no default on every tiled spec, and hipDNN has no page-size scalar
 * field -- so it must be DERIVED from the graph. Getting that derivation wrong indexes
 * the KV cache with the wrong stride and returns silently wrong numbers rather than
 * faulting, which makes it the highest-risk arithmetic in the integration.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave on gfx950. `_get_2d_launch_meta` selects 32 only for gfx1250
/// (attention_unified.py:4096); this engine's `arch` list is gfx950 alone, and packs
/// arch-prune before the matcher runs, so 64 is exact rather than a default.
inline constexpr int64_t GFX950_TILED_WAVE_LANES = 64;

/// The KV-cache page sizes `supports_tiled_2d` admits
/// (kernels/gfx950/attention_tiled_2d.py:946). A graph whose page size is anything
/// else is DECLINED -- never rounded, never clamped.
inline constexpr int64_t GFX950_TILED_BLOCK_SIZE_MIN = 16;
inline constexpr int64_t GFX950_TILED_BLOCK_SIZE_MID = 32;
inline constexpr int64_t GFX950_TILED_BLOCK_SIZE_MAX = 64;

/// `num_warps` and `block_m_per_warp` domains (attention_tiled_2d.py:958, :964), plus
/// the 1024-thread CTA cap that couples them (:970): block_m_per_warp=32 admits only
/// num_warps in {1,2,4}, because 8 warps x 32 rows would exceed it.
inline constexpr int64_t GFX950_TILED_MAX_CTA_THREADS = 1024;

/// True for a `block_size` the kernel can be built for.
inline constexpr bool gfx950TiledBlockSizeIsLegal(int64_t blockSize)
{
    return blockSize == GFX950_TILED_BLOCK_SIZE_MIN || blockSize == GFX950_TILED_BLOCK_SIZE_MID
           || blockSize == GFX950_TILED_BLOCK_SIZE_MAX;
}

/// True for a `num_warps` in the predicate's own set.
inline constexpr bool gfx950TiledNumWarpsIsLegal(int64_t numWarps)
{
    return numWarps == 1 || numWarps == 2 || numWarps == 4 || numWarps == 8;
}

/// The grid and block a variant must launch with.
struct Gfx950AttentionTiledGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx950AttentionTiledGeometry& a,
                           const Gfx950AttentionTiledGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one variant. Mirrors `_get_2d_launch_meta`.
 *
 * The Python, branch B (kernels/common/attention_unified.py:4079-4102):
 *
 *     block_m = num_warps * block_m_per_warp
 *     block_q = (block_m // num_queries_per_kv
 *                if num_queries_per_kv <= block_m else 1)
 *     total_num_q_blocks = total_q // block_q + num_seqs
 *     grid  = (num_kv_heads, total_num_q_blocks, 1)
 *     block = (wave_size * num_warps, 1, 1)
 *
 * **`total_q` and `num_seqs` come from the GRAPH, not the descriptor**, and that is
 * structural rather than a choice: `UnifiedAttention2DTiledSpec` has no `total_q`,
 * `max_seqlen_q`, `max_seqlen_k` or `batch` field at all -- verified by enumerating
 * `dataclasses.fields()`. The tiled kernel generalises over sequence length at
 * runtime where the dense kernel specialises at compile time. (`num_seqs` IS a spec
 * field, but only because it drives `binary_search_iters`, a compile-time trip count;
 * the grid's `num_seqs` is the graph's actual batch.)
 *
 * That is why 48 servable corpus shapes resolved to 39 distinct binaries: nine of them
 * differ only in sequence length, which the binary does not bake.
 *
 * **The `+ num_seqs` term is the varlen slack.** The kernel's half of the contract is
 * a binary search over `cu_q` with an early `ret()` for blocks landing in a padding
 * slot (attention_tiled_2d.py:1247-1266), whose invariant
 * `cu_q[i] // BLOCK_Q + i <= target` exactly inverts this construction
 * (rocke/helpers/attention.py:685-733). Restating one half without the other
 * under-launches or over-runs.
 *
 * Throws instead of returning a degenerate grid: an empty or negative launch returns
 * cleanly having written nothing, which is exactly the silent failure this file
 * exists to prevent, and prepare() is the last place a named failure is cheap.
 *
 * @param numKvHeads        grid.x. NOT num_query_heads -- see the file header.
 * @param numQueryHeads     Only to derive `num_queries_per_kv`.
 * @param totalQ            The GRAPH's flattened query-row count (sum of per-sequence
 *                          query lengths), not a descriptor field.
 * @param numSeqs           The GRAPH's sequence count. The slack term.
 * @param numWarps          The variant's own, from the KMD.
 * @param blockMPerWarp     The variant's own, from the KMD.
 * @param kernelName        Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx950AttentionTiledGeometry gfx950AttentionTiledGeometry(int64_t numKvHeads,
                                                                 int64_t numQueryHeads,
                                                                 int64_t totalQ,
                                                                 int64_t numSeqs,
                                                                 int64_t numWarps,
                                                                 int64_t blockMPerWarp,
                                                                 const std::string& kernelName)
{
    if(numKvHeads <= 0 || numQueryHeads <= 0 || totalQ <= 0 || numSeqs <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName
                + "' would launch with a non-positive num_kv_heads, num_query_heads, "
                  "total_q or num_seqs");
    }
    // num_warps divides into the CTA size and multiplies into block_m, so an illegal
    // value is a wrong CTA shape rather than a wrong answer -- but it is still wrong
    // silently. kernelMatches refuses these too; this is the last-resort guard for a
    // descriptor that reached prepare() with neither.
    if(!gfx950TiledNumWarpsIsLegal(numWarps))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName
                + "' declares a num_warps outside {1,2,4,8}");
    }
    if(blockMPerWarp != 16 && blockMPerWarp != 32)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName
                + "' declares a block_m_per_warp outside {16,32}");
    }
    // The 1024-thread CTA cap (attention_tiled_2d.py:970-975). 8 warps of 32 rows
    // would exceed it, and the predicate refuses that pair -- so a descriptor carrying
    // it names a binary the builder never emitted.
    if(blockMPerWarp == 32 && numWarps == 8)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName
                + "' pairs block_m_per_warp=32 with num_warps=8, which exceeds the "
                  "1024-thread CTA cap");
    }
    // GQA: the kernel derives its group size by integer division, so a non-divisible
    // pair silently drops the remainder heads. graph_match refuses it too.
    if(numQueryHeads % numKvHeads != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName
                + "' has num_query_heads not divisible by num_kv_heads");
    }

    const int64_t numQueriesPerKv = numQueryHeads / numKvHeads;
    const int64_t blockM = numWarps * blockMPerWarp;
    // The guard is `num_queries_per_kv <= block_m`, NOT a division-by-zero check: when
    // the GQA ratio exceeds the whole M tile a single query block cannot hold one
    // group, so the Python falls back to one query row per block.
    const int64_t blockQ = numQueriesPerKv <= blockM ? blockM / numQueriesPerKv : 1;
    if(blockQ <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_tiled: kernel '" + kernelName + "' resolved a non-positive block_q");
    }

    Gfx950AttentionTiledGeometry geometry;
    // FLOOR division, then `+ num_seqs`. Written as the expression the Python
    // evaluates rather than a simplified equivalent, so the two halves diff term for
    // term. It is deliberately NOT a ceil: the per-sequence padding block is what
    // covers the partial tail, and a ceil PLUS the slack would over-launch.
    geometry.gridX = static_cast<unsigned>(numKvHeads);
    geometry.gridY = static_cast<unsigned>(totalQ / blockQ + numSeqs);
    geometry.gridZ = 1;
    geometry.blockX = static_cast<unsigned>(GFX950_TILED_WAVE_LANES * numWarps);
    return geometry;
}

/// The paged KV cache's geometry, as the kernel indexes it.
struct Gfx950TiledPagedKvGeometry
{
    /// The KV cache page size -- `block_size` on the spec. Derived from the K tensor's
    /// sequence axis; see gfx950TiledPagedKvGeometry.
    int64_t blockSize = 0;
    /// Element count separating consecutive sequences' rows of the block table.
    /// **Elements, never bytes.**
    int64_t blockTableStride = 0;
};

/**
 * @brief The paged-KV geometry a graph implies. THE HIGHEST-RISK DERIVATION HERE.
 *
 * `block_size` is required with no default on every tiled spec, and `SdpaAttributes`
 * has **no page-size scalar field** -- all 41 of its fields were enumerated. So it
 * must be derived, and a wrong derivation indexes the KV cache with the wrong stride
 * and returns silently wrong numbers rather than faulting.
 *
 * **The answer: `block_size = K.dims[SEQ_AXIS]`. The K/V tensor IS the paged
 * container.** The page table resolves *which* block, never *how large* a block is.
 * Three independent sources agree:
 *
 *  1. **The kernel's own layout, stated and stride-proven.**
 *     attention_tiled_2d.py:1936-1938 states the cache layout as
 *     `[num_blocks, BS, NUM_KV, HD]`, and the byte strides at :1884-1887 prove the
 *     dim order:
 *         kv_stride_blk_b = BS * NUM_KV * HD * KV_BYTES
 *         kv_stride_tok_b =      NUM_KV * HD * KV_BYTES
 *         kv_stride_h_b   =               HD * KV_BYTES
 *     descending `blk > tok > kv_head > dim`, so the container's axis 1 -- the axis
 *     hipDNN calls S_kv -- IS block_size. The block-table lookup is a separate,
 *     orthogonal transform resolving axis 0 only:
 *         physical_block = block_tables[seq_idx * block_table_stride + tile_idx]
 *
 *  2. **The cuDNN convention hipDNN mirrors.** cuDNN's paged K container is
 *     `[num_blocks, page_size, num_heads, head_dim]` with the access rule
 *     `Kcache[b,h,s,d] = K[page_table_k[b, s / bs_k], h, s % bs_k, d]` -- the divisor
 *     is a property of the CONTAINER, not of the table. hipDNN carries the same two
 *     tensor UIDs, the same `max_seq_len_kv` scalar and the same setter names
 *     (`set_paged_attention_k_table`, `set_paged_attention_max_seq_len_kv`).
 *
 *  3. **hipDNN's own node validator, by what it does NOT exempt.**
 *     `SdpaFwdNode::pre_validate_node` enforces rank-4 Q/K/V unconditionally and
 *     `K.dim[2] == V.dim[2]`, with NO paged exemption anywhere in the file
 *     (`grep -c` for paged/page_table returns 0 there, against 6 in
 *     SdpaAttributes.hpp -- so the pattern works and the zero is real). A paged graph
 *     is therefore still a rank-4 K with a meaningful axis 2, and rank-4 with
 *     `[num_blocks, page_size, H_k, D]` is precisely the cuDNN container.
 *
 * Residual caveat, recorded honestly: no shipped hipDNN artifact exercises a paged
 * graph end to end -- no in-tree bundle populates `page_table_*`, both reference
 * executors decline paged, and the incumbent ASM engine rejects it. The contract is
 * established by the schema, the kernel and the cuDNN convention, and is confirmed
 * executably by this integration's own paged bundles.
 *
 * @param kSeqAxisExtent   `K.dims[SEQ_AXIS]` -- the container's page size.
 * @param pageTableInnerExtent `page_table_k.dims[1]`, the max blocks per sequence.
 *                         Doubles as the block-table row stride in ELEMENTS.
 * @return nullopt-equivalent: callers check `blockSize` legality themselves via
 *         gfx950TiledBlockSizeIsLegal, because a decline is not an error here.
 */
inline Gfx950TiledPagedKvGeometry gfx950TiledPagedKvGeometry(int64_t kSeqAxisExtent,
                                                             int64_t pageTableInnerExtent)
{
    Gfx950TiledPagedKvGeometry geometry;
    geometry.blockSize = kSeqAxisExtent;
    // ELEMENTS, not bytes. The host side computes this as a torch `.stride(0)`
    // (attention_unified.py:4181-4184), which is an element count, falling back to
    // `shape[1]`. A byte stride here is a 2x-4x indexing error into the KV cache --
    // silently wrong numbers, not a fault. The tensor is dense i32 rows, so the row
    // stride IS the inner extent.
    geometry.blockTableStride = pageTableInnerExtent;
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
