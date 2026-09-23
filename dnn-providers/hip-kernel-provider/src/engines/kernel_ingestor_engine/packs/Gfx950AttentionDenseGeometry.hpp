// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx950AttentionDenseGeometry.hpp
 * @brief The launch geometry gfx950 attention_dense restates from its Python builder.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same
 * binary from C++. Everything the Python launch path computes, the engine must
 * recompute IDENTICALLY, and nothing in the build, the packer, the validator or the
 * test suite compares the two. The kernel does not fail on a mismatch -- it computes
 * something else, so the whole class is found by differential testing or not at all.
 *
 * Header-only and dependency-light on purpose: a pure function of KMD metadata,
 * testable on any machine, with no HIP context and nothing to mock.
 *
 * TWO DIFFERENCES FROM THE gfx942 TWIN, each verified against the source rather
 * than assumed from the sibling:
 *
 *  1. **block_m is a spec field that this engine pins, not one the kernel bakes.**
 *     Unlike the gfx942 twin -- which really does carry a module-level `_BLOCK_M`
 *     (kernels/gfx942/attention_dense.py:211-213) and varies it across its shipped
 *     set -- gfx950 reads the value off the spec: `attention_dense_grid` divides by
 *     `spec.block_m` and `attention_dense_block` is `(spec.num_waves * 64, 1, 1)`
 *     with `num_waves = block_m // 32` (kernels/gfx950/attention_dense.py:2046-2059).
 *     The preflight accepts every geometry in DENSE_TILE_GEOMETRIES, which includes
 *     `bm128` (kernels/common/attention_dense_spec.py:23-26).
 *
 *     Every variant this engine ships is block_m 256, so the constant below is a
 *     pin on the SHIPPED SET, not a property of the binary. It is deliberately not
 *     a KMD field because it does not vary -- the KMD carries only what varies.
 *     If a bm128 variant is ever added, this constant stops being true: block_m
 *     must become a KMD field, be threaded through gfx950AttentionDenseGeometry(),
 *     and be compared in kernelMatches(). Until then, varying it here would launch
 *     512 lanes against a 256-lane binary and halve the query grid.
 *
 *  2. **The ceiling is LIVE, not defensive.** On gfx942 `Sq % block_m == 0` is
 *     enforced by the predicate, so the ceil is exact and written only for
 *     term-by-term comparison with the Python. gfx950 serves RAGGED shapes, where
 *     `seqlen_q % block_m != 0` is legal and the last query block is partial. The
 *     Python carries the same ceil wherever it counts query blocks -- the spec's own
 *     block accounting (attention_dense.py:153, :170, :191) and `attention_dense_grid`
 *     (:2046-2055). Truncating here would drop the final block: the tail rows are
 *     never written, and nothing reports it.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The query-block tile every shipped gfx950 attention_dense variant is built with.
/// `block_m` IS a spec field (kernels/common/attention_dense_spec.py), so this is a
/// pin on the shipped set rather than a constant of the binary -- see note 1 above
/// for what must change if a variant with another block_m is ever shipped.
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M = 256;

/// Lanes per wave64 wave, and the divisor `num_waves` uses. `attention_dense_block`
/// is `(num_waves * 64, 1, 1)` with `num_waves = block_m // 32`.
inline constexpr int64_t GFX950_WAVE_LANES = 64;
inline constexpr int64_t GFX950_ROWS_PER_WAVE = 32;

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
 * @brief The launch geometry for one variant, from its KMD metadata alone.
 *
 * Mirrors `attention_dense_grid` (kernels/gfx950/attention_dense.py:2046-2055):
 *
 *     nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m  # ceil: ragged tail
 *     return (nqb, spec.num_query_heads, spec.batch)
 *
 * and `attention_dense_block` (:2057-2059), `(spec.num_waves * 64, 1, 1)`.
 *
 * Every shipped variant is non-persistent, so only the `else` arm above is mirrored
 * here; the Python's persistent arm -- `(spec.num_persistent, 1, 1)` -- has no
 * counterpart in this catalog. A count is deliberately not quoted: the census test
 * owns the inventory, and a number here would drift.
 *
 * Throws instead of returning a degenerate grid. An empty or negative launch returns
 * cleanly having written nothing, which is the silent-wrong-answer case this file
 * defends against; prepare() is the last place a named failure is cheap.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx950AttentionDenseGeometry gfx950AttentionDenseGeometry(int64_t seqLenQ,
                                                                 int64_t numQueryHeads,
                                                                 int64_t batch,
                                                                 const std::string& kernelName)
{
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
    // Written as the same expression the Python evaluates rather than the literal 512,
    // so the two halves can be diffed term for term. The tile is pinned to the shipped
    // set, not baked into the binary -- see note 1 in the file header.
    geometry.blockX = static_cast<unsigned>(GFX950_ATTENTION_DENSE_BLOCK_M / GFX950_ROWS_PER_WAVE
                                            * GFX950_WAVE_LANES);
    // CEIL, and it is load-bearing here: a ragged shape has a partial final query
    // block, and truncating drops it.
    geometry.gridX = static_cast<unsigned>((seqLenQ + GFX950_ATTENTION_DENSE_BLOCK_M - 1)
                                           / GFX950_ATTENTION_DENSE_BLOCK_M);
    geometry.gridY = static_cast<unsigned>(numQueryHeads);
    geometry.gridZ = static_cast<unsigned>(batch);
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
