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
 *  1. **block_m is not a parameter.** gfx950 bakes `_BLOCK_M = 256` as a module
 *     constant (attention_dense.py:88) and `attention_dense_block` derives the CTA
 *     from the `num_waves` property, which is `_BLOCK_M // 32` -- not from a spec
 *     field. There is no block_m knob to pass, so the block is the CONSTANT 512
 *     lanes for every variant this engine ships. Taking a block_m argument here
 *     would invite a caller to vary something the binary cannot.
 *
 *  2. **The ceiling is LIVE, not defensive.** On gfx942 `Sq % block_m == 0` is
 *     enforced by the predicate, so the ceil is exact and written only for
 *     term-by-term comparison with the Python. gfx950 serves RAGGED shapes, where
 *     `seqlen_q % 256 != 0` is legal and the last query block is partial
 *     (attention_dense.py:1878 keeps the ceil for exactly that reason). Truncating
 *     here would drop the final block: the tail rows are never written, and nothing
 *     reports it.
 *
 *  3. **num_persistent defaults to 256**, the MI355X CU count, where gfx942 uses
 *     304. That value is not used here -- it arrives from the KMD -- but it is why
 *     a gfx942 geometry test's expectations do not transfer.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The query-block tile, baked into the gfx950 kernel as `_BLOCK_M`
/// (kernels/gfx950/attention_dense.py:88). Not a spec field and not a knob: the
/// causal mask and the P relayout both assume it.
inline constexpr int64_t GFX950_ATTENTION_DENSE_BLOCK_M = 256;

/// Lanes per wave64 wave, and the divisor `num_waves` uses. `attention_dense_block`
/// is `(num_waves * 64, 1, 1)` with `num_waves = _BLOCK_M // 32`.
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
 * Mirrors `attention_dense_grid` (kernels/gfx950/attention_dense.py:1874-1881):
 *
 *     if spec.persistent: return (spec.num_persistent, 1, 1)
 *     nqb = (spec.seqlen_q + _BLOCK_M - 1) // _BLOCK_M   # ceil: ragged partial block
 *     return (nqb, spec.num_query_heads, spec.batch)
 *
 * and `attention_dense_block` (:1883-1885), `(num_waves * 64, 1, 1)`.
 *
 * BOTH ARMS MATTER. The persistent grid-stride variant is a different binary that
 * expects a 1-D grid of `num_persistent` CTAs. Launching it on the default 3-D grid
 * leaves output rows unwritten, with no error anywhere.
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
                                                                 int64_t persistent,
                                                                 int64_t numPersistent,
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

    Gfx950AttentionDenseGeometry geometry;
    // Constant across every shipped variant: block_m is baked, not a knob. Written as
    // the same expression the Python evaluates rather than the literal 512, so the two
    // halves can be diffed term for term.
    geometry.blockX = static_cast<unsigned>(GFX950_ATTENTION_DENSE_BLOCK_M / GFX950_ROWS_PER_WAVE
                                            * GFX950_WAVE_LANES);
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
        geometry.gridX = static_cast<unsigned>((seqLenQ + GFX950_ATTENTION_DENSE_BLOCK_M - 1)
                                               / GFX950_ATTENTION_DENSE_BLOCK_M);
        geometry.gridY = static_cast<unsigned>(numQueryHeads);
        geometry.gridZ = static_cast<unsigned>(batch);
    }
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
