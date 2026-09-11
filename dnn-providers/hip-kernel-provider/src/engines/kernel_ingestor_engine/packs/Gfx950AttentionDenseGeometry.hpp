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
 * THREE DIFFERENCES FROM THE gfx942 TWIN, each verified against the source rather
 * than assumed from the sibling:
 *
 *  1. **block_m IS a parameter, as of ROCm/rocm-libraries#11627.** It was not before:
 *     the module baked `_BLOCK_M = 256` as a constant and this header pinned 256 to
 *     match. #11627 deleted that constant, replaced it with a tile-geometry table
 *     (`DENSE_TILE_GEOMETRIES` = {default: bm256, bm128: bm128}) and a real spec field
 *     `block_m` (attention_dense.py:168), and made `num_waves` the property
 *     `block_m // 32` (:483). So the CTA size is `block_m * 2` lanes, 512 only at
 *     block_m=256, and the grid's ceil divides by the variant's own tile.
 *
 *     Pinning 256 here after that change compiles, validates, desk-checks and passes
 *     every mechanical gate while launching a bm128 binary with the wrong grid and
 *     twice the threads it was built for. That is why block_m is now read from the
 *     descriptor and compared in kernelMatches like any other baked shape field.
 *
 *  2. **The ceiling is LIVE, not defensive.** On gfx942 `Sq % block_m == 0` is
 *     enforced by the predicate, so the ceil is exact and written only for
 *     term-by-term comparison with the Python. gfx950 serves RAGGED shapes, where
 *     `seqlen_q % block_m != 0` is legal and the last query block is partial
 *     (attention_dense.py:2307 keeps the ceil for exactly that reason). Truncating
 *     here would drop the final block: the tail rows are never written, and nothing
 *     reports it.
 *
 *  3. **num_persistent defaults to 256**, the MI355X CU count, where gfx942 uses
 *     304. That value is not used here -- it arrives from the KMD -- but it is why
 *     a gfx942 geometry test's expectations do not transfer.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// Lanes per wave64 wave, and the divisor `num_waves` uses. `attention_dense_block`
/// is `(num_waves * 64, 1, 1)` with `num_waves = block_m // 32`
/// (kernels/gfx950/attention_dense.py:483-484, 2313-2315).
inline constexpr int64_t GFX950_WAVE_LANES     = 64;
inline constexpr int64_t GFX950_ROWS_PER_WAVE  = 32;

/// The tile geometries the kernel's own `supports_attention_dense` admits
/// (attention_dense.py:591-598, derived from `DENSE_TILE_GEOMETRIES`). A descriptor
/// naming anything else describes a binary the builder refuses to emit.
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M_DEFAULT = 256;
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M_BM128   = 128;

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
 * Mirrors `attention_dense_grid` (kernels/gfx950/attention_dense.py:2302-2310):
 *
 *     if spec.persistent: return (spec.num_persistent, 1, 1)
 *     nqb = (spec.seqlen_q + spec.block_m - 1) // spec.block_m  # ceil: ragged partial
 *     return (nqb, spec.num_query_heads, spec.batch)
 *
 * and `attention_dense_block` (:2313-2315), `(num_waves * 64, 1, 1)` with
 * `num_waves = block_m // 32` (:483-484).
 *
 * BOTH ARMS MATTER. The persistent grid-stride variant is a different binary that
 * expects a 1-D grid of `num_persistent` CTAs. Launching it on the default 3-D grid
 * leaves output rows unwritten, with no error anywhere.
 *
 * Throws instead of returning a degenerate grid. An empty or negative launch returns
 * cleanly having written nothing, which is the silent-wrong-answer case this file
 * defends against; prepare() is the last place a named failure is cheap.
 *
 * @param blockM   The variant's own `block_m`, from the KMD. NOT a constant: see the
 *                 file header. A descriptor that omits it resolves to the KMD's
 *                 `default_value`, which is why that default must be 256 -- the
 *                 dispatcher's own geometry -- and why kernelMatches compares it.
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline Gfx950AttentionDenseGeometry gfx950AttentionDenseGeometry(int64_t seqLenQ,
                                                                 int64_t numQueryHeads,
                                                                 int64_t batch,
                                                                 int64_t persistent,
                                                                 int64_t numPersistent,
                                                                 int64_t blockM,
                                                                 const std::string& kernelName)
{
    // A persistent variant with no usable CTA count would launch an empty or negative
    // grid. Fail at prepare with a named reason rather than at the far end of a silent
    // miscompute.
    if(persistent != 0 && numPersistent <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_dense: kernel '" + kernelName
                + "' is persistent but declares a non-positive num_persistent");
    }
    // The default arm indexes gridY/gridZ directly, so a non-positive head count or
    // batch launches zero CTAs and returns having written nothing.
    if(persistent == 0 && (seqLenQ <= 0 || numQueryHeads <= 0 || batch <= 0))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_dense: kernel '" + kernelName
                + "' declares a non-positive seqlen_q, num_query_heads or batch");
    }
    // block_m divides in BOTH the block size and the default grid's ceil, so a
    // non-positive value is a divide-by-zero or a negative CTA count rather than a
    // wrong answer. Both tile geometries the kernel admits are checked in
    // kernelMatches; this is the last-resort guard for a descriptor that reached
    // prepare() with neither.
    if(blockM <= 0 || blockM % GFX950_ROWS_PER_WAVE != 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx950 attention_dense: kernel '" + kernelName
                + "' declares a block_m that is not a positive multiple of 32");
    }

    Gfx950AttentionDenseGeometry geometry;
    // `num_waves * 64` with `num_waves = block_m // 32`. Written as the expression the
    // Python evaluates rather than a literal, so the two halves diff term for term --
    // and so this stops being 512 the moment a bm128 variant ships.
    geometry.blockX =
        static_cast<unsigned>(blockM / GFX950_ROWS_PER_WAVE * GFX950_WAVE_LANES);
    if(persistent != 0)
    {
        geometry.gridX = static_cast<unsigned>(numPersistent);
        geometry.gridY = 1;
        geometry.gridZ = 1;
    }
    else
    {
        // CEIL, and it is load-bearing here: a ragged shape has a partial final query
        // block, and truncating drops it.
        geometry.gridX = static_cast<unsigned>((seqLenQ + blockM - 1) / blockM);
        geometry.gridY = static_cast<unsigned>(numQueryHeads);
        geometry.gridZ = static_cast<unsigned>(batch);
    }
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
