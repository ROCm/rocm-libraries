// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx950AttentionDenseGeometry.hpp
 * @brief The tile rules and launch geometry gfx950 attention_dense restates from its
 *        Python builder.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same
 * binary from C++. Everything the Python launch path computes, the engine must
 * recompute IDENTICALLY, and nothing in the build, the packer, the validator or the
 * test suite compares the two. The kernel does not fail on a mismatch -- it computes
 * something else, so the whole class is found by differential testing or not at all.
 *
 * Header-only and dependency-light on purpose: pure functions of a candidate's KMD tile
 * and the graph's dimensions, testable on any machine, with no HIP context and nothing
 * to mock.
 *
 * THE DIFFERENCE FROM THE gfx942 TWIN, verified against the source rather than
 * assumed from the sibling:
 *
 *  **block_m is a per-candidate KMD field, not a module constant.** The gfx942 twin
 *  carries a module-level `_BLOCK_M` (kernels/gfx942/attention_dense.py:211-213).
 *  gfx950 reads the value off the spec: `attention_dense_grid` divides by
 *  `spec.block_m` and `attention_dense_block` is `(spec.num_waves * 64, 1, 1)` with
 *  `num_waves = block_m // 32` (kernels/gfx950/attention_dense.py:2046-2059,
 *  kernels/common/attention_dense_spec.py:213-214). The catalog ships more than one
 *  block_m, so the engine reads the candidate's completed `block_m` metadata and
 *  passes it here; a binary launched with another candidate's block_m runs the wrong
 *  number of lanes over the wrong number of query blocks. `block_n` changes which
 *  graphs a candidate can serve (Skv % block_n) but not the launch, so it is an
 *  applicability input only and has no parameter below.
 *
 * The query-block count is a CEILING, as in the Python (`attention_dense_grid`,
 * :2046-2055). kernel_match only serves graphs with `Sq % block_m == 0`, where the
 * ceiling equals the quotient; it is kept so the two halves diff term for term, and so
 * a call outside that contract launches every row rather than silently dropping the
 * last partial block.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave64 wave, and the divisor `num_waves` uses. `attention_dense_block`
/// is `(num_waves * 64, 1, 1)` with `num_waves = block_m // 32`.
inline constexpr int64_t GFX950_WAVE_LANES = 64;
inline constexpr int64_t GFX950_ROWS_PER_WAVE = 32;

/// The two query tiles gfx950 builds: the block_m values of DENSE_TILE_GEOMETRIES
/// (kernels/common/attention_dense_spec.py:23-28), which gfx950's `supports()` admits
/// and nothing else (kernels/gfx950/attention_dense.py:301-308).
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M_SMALL = 128;
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M_LARGE = 256;

/// The key tile granularity: `block_n` is a positive multiple of 32
/// (AttentionDenseSpec.__post_init__, kernels/common/attention_dense_spec.py:88).
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_N_QUANTUM = 32;

/// gfx950 LDS per workgroup (ArchTarget("gfx950").lds_capacity_bytes).
inline constexpr int64_t GFX950_LDS_CAPACITY_BYTES = 163840;

/**
 * @brief Static LDS the non-persistent body allocates for one (head_size, block_n).
 *
 * Restates the K/V slabs of build_attention_dense (kernels/gfx950/attention_dense.py,
 * the "LDS allocation" block) at the layout every shipped variant is built with --
 * _NBUF = 2 buffers, K pad 8 (per row at D128, per packed row-group at D64), V pad 32
 * at D128 and none at D64, 2-byte elements:
 *
 *     D128: K [2, BN, 128 + 8] + V [2, BN, 128 + 32]      = 1184 * BN bytes
 *     D64:  K [2, BN / 2, 2 * 64 + 8] + V [2, BN, 64]     =  528 * BN bytes
 *
 * block_m does not enter it. Returns 0 for any other head size; callers have already
 * required 64 or 128.
 */
inline int64_t gfx950AttentionDenseStaticLdsBytes(int64_t headSize, int64_t blockN)
{
    constexpr int64_t BUFFERS = 2;
    constexpr int64_t ELEMENT_BYTES = 2;
    constexpr int64_t K_PAD = 8;
    constexpr int64_t V_PAD = 32;
    if(headSize == 128)
    {
        return BUFFERS * blockN * ((headSize + K_PAD) + (headSize + V_PAD)) * ELEMENT_BYTES;
    }
    if(headSize == 64)
    {
        constexpr int64_t ROWS_PER_GROUP = 2;
        const int64_t kBytes
            = BUFFERS * (blockN / ROWS_PER_GROUP) * (ROWS_PER_GROUP * headSize + K_PAD);
        const int64_t vBytes = BUFFERS * blockN * headSize;
        return (kBytes + vBytes) * ELEMENT_BYTES;
    }
    return 0;
}

/**
 * @brief Is (blockM, blockN) a tile the gfx950 dense kernel is built with at @p headSize?
 *
 * The conjunction of every tile rule the Python enforces, plus the LDS budget the
 * lowering enforces:
 *   - head_size in {64, 128}                        (spec __post_init__)
 *   - block_m in {128, 256}                         (gfx950 supports(), :301-308)
 *   - block_n > 0 and block_n % 32 == 0             (spec __post_init__, :88)
 *   - block_m % block_n == 0                        (check_dense_spec_preflight, :399)
 *   - block_n % (block_m / 32) == 0, i.e. num_waves (gfx950 supports(), :309-313)
 *   - static LDS fits GFX950_LDS_CAPACITY_BYTES     (excludes D128 block_n 256, which
 *                                                    passes every check above)
 *
 * Total over every int64 triple: the membership and sign tests run before any modulo,
 * so no divisor reaching one is zero or negative, and block_n is at most 256 by the
 * time it is multiplied. Anything that divides by a candidate's tile must ask this first.
 */
inline bool isSupportedGfx950AttentionDenseTile(int64_t headSize, int64_t blockM, int64_t blockN)
{
    if(headSize != 64 && headSize != 128)
    {
        return false;
    }
    if(blockM != GFX950_ATTENTION_DENSE_BLOCK_M_SMALL
       && blockM != GFX950_ATTENTION_DENSE_BLOCK_M_LARGE)
    {
        return false;
    }
    if(blockN <= 0 || blockN % GFX950_ATTENTION_DENSE_BLOCK_N_QUANTUM != 0)
    {
        return false;
    }
    const int64_t numWaves = blockM / GFX950_ROWS_PER_WAVE;
    if(blockM % blockN != 0 || blockN % numWaves != 0)
    {
        return false;
    }
    return gfx950AttentionDenseStaticLdsBytes(headSize, blockN) <= GFX950_LDS_CAPACITY_BYTES;
}

/// The grid and block a variant must launch with.
struct Gfx950AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const Gfx950AttentionDenseGeometry& a,
                           const Gfx950AttentionDenseGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one selected candidate over one graph.
 *
 * Mirrors `attention_dense_grid` (kernels/gfx950/attention_dense.py:2046-2055):
 *
 *     nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m
 *     return (nqb, spec.num_query_heads, spec.batch)
 *
 * and `attention_dense_block` (:2057-2059), `(spec.num_waves * 64, 1, 1)`.
 *
 * @p blockM is the selected candidate's own completed `block_m`; @p seqLenQ,
 * @p numQueryHeads and @p batch are the graph's, since the binary takes its shape at
 * runtime and its metadata carries only canonical build inputs.
 *
 * Every shipped variant is non-persistent, so only the `else` arm above is mirrored
 * here; the Python's persistent arm -- `(spec.num_persistent, 1, 1)` -- has no
 * counterpart in this catalog. A count is deliberately not quoted: the census test
 * owns the inventory, and a number here would drift.
 *
 * Throws instead of returning a degenerate grid. An empty or negative launch returns
 * cleanly having written nothing, which is the silent-wrong-answer case this file
 * defends against; prepare() is the last place a named failure is cheap. A block_m
 * gfx950 does not build is refused before it divides anything.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx950AttentionDenseGeometry gfx950AttentionDenseGeometry(int64_t blockM,
                                                                 int64_t seqLenQ,
                                                                 int64_t numQueryHeads,
                                                                 int64_t batch,
                                                                 const std::string& kernelName)
{
    if(blockM != GFX950_ATTENTION_DENSE_BLOCK_M_SMALL
       && blockM != GFX950_ATTENTION_DENSE_BLOCK_M_LARGE)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_dense: kernel '" + kernelName + "' declares block_m "
                + std::to_string(blockM) + ", which gfx950 does not build");
    }

    // gridY/gridZ index heads and batch directly; a non-positive value launches zero
    // CTAs and returns having written nothing.
    if(seqLenQ <= 0 || numQueryHeads <= 0 || batch <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_dense: kernel '" + kernelName
                + "' declares a non-positive seqlen_q, num_query_heads or batch");
    }

    Gfx950AttentionDenseGeometry geometry;
    // Written as the same expression the Python evaluates, so the two halves can be
    // diffed term for term: 256 lanes at block_m 128, 512 at block_m 256.
    geometry.blockX = static_cast<unsigned>(blockM / GFX950_ROWS_PER_WAVE * GFX950_WAVE_LANES);
    // CEIL, as the Python writes it. Exact for every graph kernel_match serves
    // (Sq % block_m == 0); on any other input it keeps the partial final block.
    geometry.gridX = static_cast<unsigned>((seqLenQ + blockM - 1) / blockM);
    geometry.gridY = static_cast<unsigned>(numQueryHeads);
    geometry.gridZ = static_cast<unsigned>(batch);
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
