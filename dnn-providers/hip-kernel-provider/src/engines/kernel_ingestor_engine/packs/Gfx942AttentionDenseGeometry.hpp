// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>
#include <string>

#include <hipdnn_plugin_sdk/PluginException.hpp>

/**
 * @file Gfx942AttentionDenseGeometry.hpp
 * @brief The launch geometry gfx942 attention_dense restates from its Python builder.
 *
 * A rocKE kernel is launched from Python; an ingestor engine relaunches the same
 * binary from C++. Everything the Python launch path computes, the engine must
 * recompute IDENTICALLY, and nothing in the build, the packer, the validator or the
 * test suite compares the two. The kernel does not fail on a mismatch -- it computes
 * something else, so the whole class is found by differential testing or not at all.
 * Two defects have already shipped through this restatement: the persistent grid
 * launched with the default grid size, and a windowed causal graph was served as
 * plain causal.
 *
 * This header exists so the arithmetic is REACHABLE BY A TEST. Inside prepare() it
 * sits behind a compiled kernel and a device, which is why it went unchecked while
 * most shipped shape-keys never executed on GPU -- "it worked on what we ran" covered
 * a minority of descriptors, and the persistent branch least of all.
 *
 * Header-only and dependency-light on purpose: a pure function of KMD metadata,
 * testable on any machine, with no HIP context and nothing to mock.
 */
namespace hip_kernel_provider::kernel_ingestor_engine
{

/// The grid and block a variant must launch with.
struct AttentionDenseGeometry
{
    unsigned gridX = 0;
    unsigned gridY = 0;
    unsigned gridZ = 0;
    unsigned blockX = 0;

    friend bool operator==(const AttentionDenseGeometry& a, const AttentionDenseGeometry& b)
    {
        return a.gridX == b.gridX && a.gridY == b.gridY && a.gridZ == b.gridZ
               && a.blockX == b.blockX;
    }
};

/**
 * @brief The launch geometry for one variant, from its KMD metadata alone.
 *
 * Mirrors `attention_dense_grid` (attention_dense.py:1803):
 *
 *     if spec.persistent: return (spec.num_persistent, 1, 1)
 *     return (ceil(Sq / block_m), num_query_heads, batch)
 *
 * and `attention_dense_block` (:1822), which is `(block_m // 32 * 64, 1, 1)` wave64
 * lanes. The default arm's ceiling is exact because `supports_attention_dense`
 * enforces `Sq % block_m == 0`; it is written as a ceiling anyway so the two
 * expressions match term for term and a reader can diff them against the Python.
 *
 * BOTH ARMS MATTER. The persistent grid-stride variant is a different binary that
 * expects a 1-D grid of `num_persistent` CTAs. Launching it on the default 3-D grid
 * leaves output rows unwritten -- in the builder's own words, "a mismatch writes some
 * rows twice and others never" -- with no error anywhere.
 *
 * Throws instead of returning a degenerate grid. An empty or negative launch returns
 * cleanly having written nothing, which is the silent-wrong-answer case this whole
 * file is defending against; prepare() is the last place a named failure is cheap.
 *
 * @param kernelName Only for the diagnostic, so a failure names the descriptor.
 */
inline AttentionDenseGeometry attentionDenseGeometry(int64_t blockM,
                                                     int64_t seqLenQ,
                                                     int64_t numQueryHeads,
                                                     int64_t batch,
                                                     int64_t persistent,
                                                     int64_t numPersistent,
                                                     const std::string& kernelName)
{
    if(blockM <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 attention_dense: kernel '" + kernelName + "' declares a non-positive block_m");
    }
    // A persistent variant with no usable CTA count would launch an empty or negative
    // grid. Fail at prepare with a named reason rather than at the far end of a silent
    // miscompute: this is the correspondence the old comment said nothing checked, and
    // it is what let the missing branch ship.
    if(persistent != 0 && numPersistent <= 0)
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 attention_dense: kernel '" + kernelName
                + "' is persistent but declares a non-positive num_persistent");
    }
    // The default arm indexes gridY/gridZ directly, so a non-positive head count or
    // batch launches zero CTAs and returns having written nothing.
    if(persistent == 0 && (seqLenQ <= 0 || numQueryHeads <= 0 || batch <= 0))
    {
        throw hipdnn_plugin_sdk::HipdnnPluginException(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "gfx942 attention_dense: kernel '" + kernelName
                + "' declares a non-positive seqlen_q, num_query_heads or batch");
    }

    AttentionDenseGeometry geometry;
    geometry.blockX = static_cast<unsigned>(blockM / 32 * 64);
    if(persistent != 0)
    {
        geometry.gridX = static_cast<unsigned>(numPersistent);
        geometry.gridY = 1;
        geometry.gridZ = 1;
    }
    else
    {
        geometry.gridX = static_cast<unsigned>((seqLenQ + blockM - 1) / blockM);
        geometry.gridY = static_cast<unsigned>(numQueryHeads);
        geometry.gridZ = static_cast<unsigned>(batch);
    }
    return geometry;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine
